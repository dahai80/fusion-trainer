"""Rollout-sink ingestion for RFT/GRPO (#86).

Consumes gateway rollout-tap sink records (JSONL, one record per
mirrored LLM call, schema in docs/rollout-tap.md):

    {"schema_version": 1, "rollout_id", "tenant_id", "source", "model",
     "created_at", "messages": [{"role","content"}, ...],
     "finish_reason", "latency_ms", "prompt",
     "outcome": {"signal": ..., "score": 0.0-1.0, "detail": ...}}

and turns them into trainer-consumable datasets:
    SFT  : positive rollouts (score >= min_score) -> {"messages": [...]}
    GRPO : records with a non-empty prompt -> {"prompt", "reward",
           "metadata"} (reward ref field survives DatasetLoader.load_grpo)

Outcome signals map to fixed scalar rewards (Rule 5, deterministic):
task_success/user_accepted -> 1.0, user_edited -> 0.5,
user_rejected/error -> 0.0, none/unobserved -> excluded from the
positive set. An explicit numeric outcome.score in [0, 1] overrides
the signal default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROLLOUT_SCHEMA_VERSION = 1
# Outcome sidecar contract (office-api #48): v2 adds top-level env_state /
# validator {name, version, status} fields, additively — v1 stays valid.
ROLLOUT_SIDECAR_SCHEMA_VERSIONS = (1, 2)
ROLLOUT_MANIFEST_SUFFIX = ".from_rollouts.json"
_REQUIRED_KEYS = ("schema_version", "rollout_id", "model", "messages")

SIGNAL_SCORES: dict[str, float | None] = {
    "task_success": 1.0,
    "user_accepted": 1.0,
    "user_edited": 0.5,
    "user_rejected": 0.0,
    "error": 0.0,
    "none": None,
}


class RolloutIngestError(Exception):
    pass


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in ("user", "assistant", "system", "tool"):
            return False
        # Issue #88: assistant tool-call turns may carry empty content; the
        # call payload lives in tool_calls. Structural validity is enforced
        # by formats.validate_chat_messages at conversion time.
        calls = m.get("tool_calls")
        if calls:
            continue
        if m.get("role") == "tool" and "tool_call_id" not in m:
            return False
        content = m.get("content")
        ok_content = isinstance(content, str) and bool(content.strip())
        if not ok_content and m.get("role") == "tool":
            # tool results may be empty when the tool returned nothing
            ok_content = True
        if not ok_content:
            return False
    return True


def reward_score(outcome: dict[str, Any]) -> float | None:
    """Deterministic outcome -> reward; None = no usable signal."""
    if not isinstance(outcome, dict):
        return None
    signal = outcome.get("signal")
    score = outcome.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return SIGNAL_SCORES.get(str(signal))
    if not 0.0 <= float(score) <= 1.0:
        logger.warning("outcome.score %r out of [0,1], ignoring", score)
        return SIGNAL_SCORES.get(str(signal))
    return float(score)


def load_rollout_sink(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load sink JSONL, skip malformed lines loudly, return (records, sha256)."""
    if not path.is_file():
        raise RolloutIngestError(f"rollout sink not found: {path}")
    digest = _sha256_file(path)
    records: list[dict[str, Any]] = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                logger.warning("skip sink line %d: not JSON", lineno)
                continue
            if not isinstance(rec, dict):
                bad += 1
                logger.warning("skip sink line %d: not an object", lineno)
                continue
            version = rec.get("schema_version")
            if version != ROLLOUT_SCHEMA_VERSION:
                bad += 1
                logger.warning(
                    "skip sink line %d: schema_version=%r != %d",
                    lineno,
                    version,
                    ROLLOUT_SCHEMA_VERSION,
                )
                continue
            missing = [k for k in _REQUIRED_KEYS if k not in rec]
            if missing:
                bad += 1
                logger.warning(
                    "skip sink line %d (rollout %s): missing keys %s",
                    lineno,
                    rec.get("rollout_id", "?"),
                    missing,
                )
                continue
            if not _valid_messages(rec.get("messages")):
                bad += 1
                logger.warning(
                    "skip sink line %d (rollout %s): invalid messages",
                    lineno,
                    rec.get("rollout_id", "?"),
                )
                continue
            records.append(rec)
    if not records:
        raise RolloutIngestError(
            f"no valid rollout records in {path} (skipped {bad} bad lines)"
        )
    if bad:
        logger.warning("loaded %d records, skipped %d bad lines", len(records), bad)
    logger.info(
        "loaded %d rollout records from %s (sha256=%s)",
        len(records),
        path,
        digest[:12],
    )
    return records, digest


def load_outcome_sidecar(path: Path) -> dict[str, dict[str, Any]]:
    """Load an outcome sidecar JSONL -> {rollout_id: outcome} (#97).

    Sidecar records (e.g. office-api rollout-outcomes.jsonl) carry
    schema_version/rollout_id/outcome and no `source` field — the
    gateway tap contract always writes `source` and never `outcome`
    (docs/rollout-tap.md). schema v2 (office-api #48) adds top-level
    env_state / validator / task-carrying fields additively; the join
    consumes only the outcome dict, unknown fields pass through
    harmlessly. The latest record per rollout_id wins (append-only
    file). Malformed lines are skipped loudly.
    """
    if not path.is_file():
        raise RolloutIngestError(f"outcome sidecar not found: {path}")
    outcomes: dict[str, dict[str, Any]] = {}
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                bad += 1
                logger.warning("skip sidecar line %d: not JSON", lineno)
                continue
            if not (
                isinstance(rec, dict)
                and rec.get("schema_version") in ROLLOUT_SIDECAR_SCHEMA_VERSIONS
                and rec.get("rollout_id")
                and isinstance(rec.get("outcome"), dict)
            ):
                bad += 1
                logger.warning(
                    "skip sidecar line %d (rollout %s): not an outcome record",
                    lineno,
                    rec.get("rollout_id", "?") if isinstance(rec, dict) else "?",
                )
                continue
            outcomes[str(rec["rollout_id"])] = rec["outcome"]
    if not outcomes:
        raise RolloutIngestError(
            f"no outcome records in {path} (skipped {bad} bad lines)"
        )
    if bad:
        logger.warning(
            "sidecar: %d outcomes loaded, %d bad lines skipped", len(outcomes), bad
        )
    logger.info("loaded %d outcomes from sidecar %s", len(outcomes), path)
    return outcomes


def join_outcomes(
    records: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Backfill tap records without an inline outcome from the sidecar
    map (#97). Inline outcomes always win; returns (records, joined)."""
    joined = 0
    for rec in records:
        if isinstance(rec.get("outcome"), dict):
            continue
        outcome = outcomes.get(str(rec.get("rollout_id")))
        if outcome is not None:
            rec["outcome"] = dict(outcome)
            joined += 1
    logger.info("sidecar join: %d/%d tap records backfilled", joined, len(records))
    return records, joined


def positive_rollouts(
    records: list[dict[str, Any]], min_score: float = 1.0
) -> list[dict[str, Any]]:
    """Records whose reward >= min_score (RFT positive set)."""
    positives = []
    for rec in records:
        score = reward_score(rec.get("outcome"))
        if score is not None and score >= min_score:
            positives.append(rec)
    logger.info(
        "%d/%d rollouts positive at min_score=%s",
        len(positives),
        len(records),
        min_score,
    )
    return positives


def failure_rollouts(
    records: list[dict[str, Any]], max_score: float = 1.0
) -> list[dict[str, Any]]:
    """Scored records whose reward < max_score (#111 failure channel).

    Feeds the resynthesis stage (weakness profile). Unobserved outcomes
    (reward None) are NOT failures — they carry no signal and belong to
    neither the positive nor the failure set.
    """
    failures = []
    for rec in records:
        score = reward_score(rec.get("outcome"))
        if score is not None and score < max_score:
            failures.append(rec)
    logger.info(
        "%d/%d rollouts negative at max_score=%s",
        len(failures),
        len(records),
        max_score,
    )
    return failures


_WEAKNESS_SAMPLE_CAP = 10


def weakness_profile(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Signal distribution over scored failures + bounded sample ids (#111).

    The profile is the resynthesis stage's input: which outcome signals
    dominate the failure set, with a few rollout_ids per signal to pull
    full trajectories from the tap for targeted resynthesis. When
    outcomes carry a `task_id` (office sidecar contract, #117), a
    per-task failure breakdown rides along for budget anchoring.
    """
    counts: dict[str, int] = {}
    samples: dict[str, list[str]] = {}
    by_task: dict[str, int] = {}
    has_task_id = False
    for rec in failures:
        outcome = rec.get("outcome")
        signal = (
            str(outcome.get("signal"))
            if isinstance(outcome, dict) and outcome.get("signal")
            else "unobserved"
        )
        counts[signal] = counts.get(signal, 0) + 1
        if isinstance(outcome, dict) and outcome.get("task_id"):
            has_task_id = True
            tid = str(outcome["task_id"])
            by_task[tid] = by_task.get(tid, 0) + 1
        bucket = samples.setdefault(signal, [])
        rid = rec.get("rollout_id")
        if len(bucket) < _WEAKNESS_SAMPLE_CAP and rid:
            bucket.append(str(rid))
    profile: dict[str, Any] = {"failure_signals": counts, "sample_rollout_ids": samples}
    if has_task_id:
        profile["failure_by_task"] = by_task
    return profile


def struggle_features(failures: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Deterministic per-rollout struggle features (#116 step-wise judge).

    Groups records by rollout_id (the tap mirrors one LLM call per
    record) and computes cheap struggle signals per group: redundant
    consecutive tool calls and error signals per step. The score is a
    deterministic first-pass judge — an LLM judge may layer on top
    later, but injection eligibility must be explainable from these
    features alone.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in failures:
        rid = str(rec.get("rollout_id") or "")
        if rid:
            groups.setdefault(rid, []).append(rec)
    out: dict[str, dict[str, Any]] = {}
    for rid, recs in groups.items():
        tool_calls = 0
        redundant = 0
        errors = 0
        prev_key: str | None = None
        for rec in recs:
            outcome = rec.get("outcome")
            if isinstance(outcome, dict) and outcome.get("signal") == "error":
                errors += 1
            for m in rec.get("messages") or []:
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "user":
                    prev_key = None
                for call in m.get("tool_calls") or []:
                    key = _tool_call_key(call)
                    if key is None:
                        continue
                    tool_calls += 1
                    if key == prev_key:
                        redundant += 1
                    prev_key = key
        # deterministic: each error counts double a redundant call,
        # normalized by step count so long trajectories don't dominate
        score = round((redundant + 2 * errors) / max(1, len(recs)), 4)
        out[rid] = {
            "steps": len(recs),
            "tool_calls": tool_calls,
            "redundant_consecutive": redundant,
            "error_signals": errors,
            "struggle_score": score,
        }
    return out


def _tool_call_key(call: Any) -> str | None:
    fn = call.get("function") if isinstance(call, dict) else None
    if not isinstance(fn, dict) or not fn.get("name"):
        return None
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return f"{fn['name']}|{args}"
    if isinstance(args, dict):
        args = json.dumps(args, sort_keys=True, ensure_ascii=False)
    return f"{fn['name']}|{args}"


def tool_call_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Offline efficiency pre-step (#111, CARE-PO): count removable calls.

    A call is redundant when it repeats the immediately preceding call
    (same tool name + normalized arguments) within the same assistant
    turn sequence; a new user turn resets the comparison anchor. The
    redundant-call rate is the BEFORE number for any future efficiency
    reward — the budget anchor must be fixed before training, never
    policy-relative.
    """
    total_calls = 0
    redundant_calls = 0
    records_with_calls = 0
    for rec in records:
        messages = rec.get("messages")
        if not isinstance(messages, list):
            continue
        prev_key: str | None = None
        calls_in_rec = 0
        redundant_in_rec = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                prev_key = None
            for call in m.get("tool_calls") or []:
                key = _tool_call_key(call)
                if key is None:
                    continue
                calls_in_rec += 1
                if key == prev_key:
                    redundant_in_rec += 1
                prev_key = key
        if calls_in_rec:
            records_with_calls += 1
            total_calls += calls_in_rec
            redundant_calls += redundant_in_rec
    rate = (redundant_calls / total_calls) if total_calls else 0.0
    return {
        "records_with_tool_calls": records_with_calls,
        "tool_calls": total_calls,
        "redundant_consecutive": redundant_calls,
        "redundant_rate": round(rate, 4),
    }


def rollouts_to_sft(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Positive rollouts -> D1 messages samples (multi-turn + tool calls
    preserved, issue #88)."""
    from .formats import copy_message, validate_chat_messages

    samples = []
    skipped = 0
    for rec in records:
        messages = [copy_message(m) for m in rec["messages"]]
        ok, reason = validate_chat_messages(messages)
        if not ok:
            skipped += 1
            logger.warning(
                "skip rollout %s: invalid message structure (%s)",
                rec.get("rollout_id", "?"),
                reason,
            )
            continue
        sample: dict[str, Any] = {"messages": messages}
        if isinstance(rec.get("tools"), list):
            sample["tools"] = rec["tools"]
        samples.append(sample)
    if skipped:
        logger.warning(
            "dropped %d/%d rollouts with invalid tool structure", skipped, len(records)
        )
    logger.info(
        "built %d SFT samples from %d positive rollouts", len(samples), len(records)
    )
    return samples


def rollouts_to_grpo(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records with a non-empty prompt -> GRPO prompt samples with reward ref."""
    samples = []
    skipped = 0
    for rec in records:
        prompt = str(rec.get("prompt") or "").strip()
        if not prompt:
            # Fall back to the first user turn when the tap did not set prompt
            for m in rec.get("messages", []):
                if m.get("role") == "user":
                    prompt = str(m.get("content", "")).strip()
                    break
        if not prompt:
            skipped += 1
            logger.warning(
                "skip rollout %s for GRPO: no prompt", rec.get("rollout_id", "?")
            )
            continue
        score = reward_score(rec.get("outcome"))
        entry: dict[str, Any] = {"prompt": prompt}
        if score is not None:
            entry["reward"] = score
        entry["metadata"] = {
            "rollout_id": rec.get("rollout_id", ""),
            "model": rec.get("model", ""),
            "source": rec.get("source", ""),
        }
        samples.append(entry)
    if skipped:
        logger.warning("%d rollouts skipped for GRPO (no prompt)", skipped)
    logger.info(
        "built %d GRPO prompts from %d rollout records", len(samples), len(records)
    )
    return samples


def _signal_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        outcome = rec.get("outcome")
        signal = (
            str(outcome.get("signal"))
            if isinstance(outcome, dict) and outcome.get("signal")
            else "unobserved"
        )
        counts[signal] = counts.get(signal, 0) + 1
    return counts


def build_from_rollouts(
    sink_path: Path,
    sft_output_path: Path | None = None,
    grpo_output_path: Path | None = None,
    min_score: float = 1.0,
    outcomes_path: Path | None = None,
    failures_output_path: Path | None = None,
) -> dict[str, Any]:
    """Full ingestion: sink -> SFT dataset and/or GRPO prompt set + manifest.

    The manifest is written next to the SFT dataset as
    `<output>.from_rollouts.json` (atomic) and carries the sink sha256 —
    SFTTrainer picks it up automatically and records the run's rollout
    source (#86 traceability, mirrors the from-eval sidecar of #76).
    `outcomes_path` (optional, #97) names an outcome sidecar JSONL whose
    records are joined by rollout_id to backfill tap records written
    without an inline outcome.
    `failures_output_path` (optional, #111) writes scored-negative
    rollouts (reward < min_score, full records with outcome) as the
    failure-trajectory channel feeding the resynthesis stage; the
    manifest carries the weakness profile and the offline redundant
    tool-call rate.
    """
    if sft_output_path is None and grpo_output_path is None and failures_output_path is None:
        raise RolloutIngestError(
            "nothing to write: pass --output (SFT) and/or --grpo-output "
            "(GRPO prompts) and/or --failures-output (failure channel)"
        )
    records, digest = load_rollout_sink(sink_path)
    joined = 0
    if outcomes_path is not None:
        outcomes = load_outcome_sidecar(outcomes_path)
        records, joined = join_outcomes(records, outcomes)

    sft_samples: list[dict[str, Any]] = []
    if sft_output_path is not None:
        sft_samples = rollouts_to_sft(positive_rollouts(records, min_score))
        if not sft_samples:
            raise RolloutIngestError(
                f"no positive rollouts at min_score={min_score} in {sink_path} "
                f"(signals: {_signal_counts(records)}) — nothing to train on; "
                "lower --min-score or wait for outcome backfill"
            )
        sft_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sft_output_path, "w", encoding="utf-8") as fh:
            for sample in sft_samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("wrote %d SFT samples to %s", len(sft_samples), sft_output_path)

    grpo_samples: list[dict[str, Any]] = []
    if grpo_output_path is not None:
        grpo_samples = rollouts_to_grpo(records)
        if not grpo_samples:
            raise RolloutIngestError(
                f"no prompts extractable from {sink_path} — GRPO output would be empty"
            )
        grpo_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grpo_output_path, "w", encoding="utf-8") as fh:
            for sample in grpo_samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info("wrote %d GRPO prompts to %s", len(grpo_samples), grpo_output_path)

    failure_records: list[dict[str, Any]] = []
    if failures_output_path is not None:
        # #111: an empty failure set is a GOOD result — write the empty
        # file (honest) rather than erroring.
        failure_records = failure_rollouts(records, min_score)
        failures_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(failures_output_path, "w", encoding="utf-8") as fh:
            for rec in failure_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(
            "wrote %d failure rollouts to %s",
            len(failure_records),
            failures_output_path,
        )

    efficiency = tool_call_stats(records)
    if efficiency["tool_calls"]:
        logger.info(
            "redundant tool calls: %d/%d (%.1f%%)",
            efficiency["redundant_consecutive"],
            efficiency["tool_calls"],
            efficiency["redundant_rate"] * 100,
        )

    manifest_anchor = sft_output_path or grpo_output_path or failures_output_path
    manifest = {
        "source": "gateway-rollout-tap",
        "sink_path": str(sink_path),
        "sink_sha256": digest,
        "outcomes_path": str(outcomes_path) if outcomes_path else "",
        "outcomes_joined": joined,
        "min_score": min_score,
        "sft_path": str(sft_output_path) if sft_output_path else "",
        "sft_samples": len(sft_samples),
        "grpo_path": str(grpo_output_path) if grpo_output_path else "",
        "grpo_prompts": len(grpo_samples),
        "failures_path": str(failures_output_path) if failures_output_path else "",
        "failures_records": len(failure_records),
        "weakness": weakness_profile(failure_records),
        "struggle": {
            rid: feat
            for rid, feat in struggle_features(failure_records).items()
            if feat["struggle_score"] > 0
        },
        "efficiency": efficiency,
        "records_total": len(records),
        "signals": _signal_counts(records),
        "lineage": {
            "source": "gateway-rollout-tap",
            "source_path": str(sink_path),
            "source_sha256": digest,
            "transforms": (
                (["join_outcomes"] if outcomes_path else [])
                + (["positive_rollouts", "rollouts_to_sft"] if sft_output_path else [])
                + (["failure_rollouts"] if failures_output_path else [])
            ),
            "output": str(manifest_anchor),
        },
    }
    manifest_path = Path(str(manifest_anchor) + ROLLOUT_MANIFEST_SUFFIX)
    tmp_path = Path(str(manifest_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    tmp_path.replace(manifest_path)
    logger.info("wrote manifest %s", manifest_path)
    return manifest


def find_rollout_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    """Return the from-rollouts manifest for a dataset file, or None."""
    if not dataset_path:
        return None
    cand = Path(str(dataset_path) + ROLLOUT_MANIFEST_SUFFIX)
    if not cand.is_file():
        return None
    try:
        data = json.loads(cand.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable rollout manifest %s: %s", cand, exc)
        return None
    if isinstance(data, dict) and data.get("sink_sha256"):
        logger.info(
            "dataset %s is rollout-fed (sink sha256=%s)",
            dataset_path,
            str(data["sink_sha256"])[:12],
        )
        return data
    return None
