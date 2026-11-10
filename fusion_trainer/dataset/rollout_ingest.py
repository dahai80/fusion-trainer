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
        if not isinstance(m.get("content"), str) or not m["content"].strip():
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


def rollouts_to_sft(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Positive rollouts -> D1 messages samples (multi-turn preserved)."""
    samples = []
    for rec in records:
        samples.append(
            {
                "messages": [
                    {"role": m["role"], "content": m["content"]}
                    for m in rec["messages"]
                ]
            }
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
) -> dict[str, Any]:
    """Full ingestion: sink -> SFT dataset and/or GRPO prompt set + manifest.

    The manifest is written next to the SFT dataset as
    `<output>.from_rollouts.json` (atomic) and carries the sink sha256 —
    SFTTrainer picks it up automatically and records the run's rollout
    source (#86 traceability, mirrors the from-eval sidecar of #76).
    """
    if sft_output_path is None and grpo_output_path is None:
        raise RolloutIngestError(
            "nothing to write: pass --output (SFT) and/or --grpo-output (GRPO prompts)"
        )
    records, digest = load_rollout_sink(sink_path)

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

    manifest_anchor = sft_output_path or grpo_output_path
    manifest = {
        "source": "gateway-rollout-tap",
        "sink_path": str(sink_path),
        "sink_sha256": digest,
        "min_score": min_score,
        "sft_path": str(sft_output_path) if sft_output_path else "",
        "sft_samples": len(sft_samples),
        "grpo_path": str(grpo_output_path) if grpo_output_path else "",
        "grpo_prompts": len(grpo_samples),
        "records_total": len(records),
        "signals": _signal_counts(records),
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
