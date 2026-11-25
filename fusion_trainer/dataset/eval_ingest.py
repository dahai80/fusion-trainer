"""Eval-trajectory ingestion for the flywheel loop (#76).

Consumes fusion-eval#6 trajectory exports (JSONL, one record per task run):
    {"schema_version": 1, "task_id", "task_type", "arm", "model", "passed",
     "prompt", "messages": [{"role","content"}, ...], "final_diff"}

and turns them into trainer-consumable datasets:
    SFT  : passed-only records -> {"messages": [...]} (D1 messages schema,
           the existing DatasetLoader.load_messages path trains on it)
    DPO  : tasks with BOTH a passed and a failed trajectory ->
           {"prompt", "chosen", "rejected"} (D1 dpo_pair schema)

Every produced SFT dataset gets a sidecar manifest `<output>.from_eval.json`
carrying the export sha256, so an SFT run fed by eval data can record its
source (requirement 3 of #76).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EVAL_EXPORT_SCHEMA_VERSION = 1
MANIFEST_SUFFIX = ".from_eval.json"
_REQUIRED_KEYS = ("task_id", "passed", "messages")


class EvalIngestError(Exception):
    pass


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_eval_export(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a fusion-eval trajectory export. Returns (records, sha256).

    Records failing the schema check are skipped with a loud logged count —
    fail-visible, never a silent partial conversion.
    """
    if not path.is_file():
        raise EvalIngestError(f"eval export not found: {path}")
    digest = sha256_file(path)
    records: list[dict[str, Any]] = []
    bad = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                bad += 1
                logger.warning(
                    "skip corrupt export line %d in %s: %s", lineno, path, exc
                )
                continue
            if not isinstance(rec, dict):
                bad += 1
                logger.warning("skip non-object export line %d in %s", lineno, path)
                continue
            version = rec.get("schema_version")
            if version != EVAL_EXPORT_SCHEMA_VERSION:
                bad += 1
                logger.warning(
                    "skip export line %d: schema_version=%r != %d",
                    lineno,
                    version,
                    EVAL_EXPORT_SCHEMA_VERSION,
                )
                continue
            missing = [k for k in _REQUIRED_KEYS if k not in rec]
            if missing:
                bad += 1
                logger.warning(
                    "skip export line %d (task %s): missing keys %s",
                    lineno,
                    rec.get("task_id", "?"),
                    missing,
                )
                continue
            records.append(rec)
    if not records:
        raise EvalIngestError(
            f"no valid trajectory records in {path} (skipped {bad} bad lines)"
        )
    if bad:
        logger.warning(
            "loaded %d records from %s, skipped %d bad lines", len(records), path, bad
        )
    logger.info(
        "loaded %d trajectory records from %s (sha256=%s)",
        len(records),
        path,
        digest[:12],
    )
    return records, digest


def _valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in ("user", "assistant", "system", "tool"):
            return False
        # Issue #88: assistant tool-call turns may carry empty content; the
        # call payload lives in tool_calls.
        if m.get("tool_calls"):
            continue
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            continue
        if m.get("role") == "tool" and "tool_call_id" in m:
            continue
        return False
    return True


def eval_records_to_sft(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Passed-only records -> D1 messages samples. Multi-turn + tool calls
    preserved (issue #88)."""
    from .formats import copy_message, validate_chat_messages

    samples = []
    skipped = 0
    for rec in records:
        if not rec.get("passed"):
            continue
        messages = rec.get("messages")
        if not _valid_messages(messages):
            skipped += 1
            logger.warning(
                "skip passed trajectory %s: empty/invalid messages",
                rec.get("task_id", "?"),
            )
            continue
        copied = [copy_message(m) for m in messages]
        ok, reason = validate_chat_messages(copied)
        if not ok and any(
            "tool_calls" in m or m.get("role") == "tool" for m in copied
        ):
            skipped += 1
            logger.warning(
                "skip passed trajectory %s: invalid tool structure (%s)",
                rec.get("task_id", "?"),
                reason,
            )
            continue
        sample: dict[str, Any] = {"messages": copied}
        if isinstance(rec.get("tools"), list):
            sample["tools"] = rec["tools"]
        samples.append(sample)
    if skipped:
        logger.warning("%d passed trajectories skipped (invalid messages)", skipped)
    logger.info(
        "built %d SFT samples from %d trajectory records", len(samples), len(records)
    )
    return samples


def _assistant_text(messages: list[dict[str, Any]]) -> str:
    """Flatten assistant turns to DPO-pair text; tool-call turns render as
    deterministic TOOL_CALL JSON lines (issue #88) instead of vanishing."""
    import json as _json

    parts = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        calls = m.get("tool_calls") or []
        for tc in calls:
            fn = (tc or {}).get("function") or {}
            parts.append(
                "TOOL_CALL "
                + _json.dumps(
                    {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content)
    return "\n\n".join(parts)


def mine_preference_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Failure mining: same task_id with BOTH passed and failed trajectories
    -> DPO pairs {prompt, chosen, rejected}.

    chosen = the passed trajectory's assistant turns joined; rejected = the
    failed one's. Records without a passed sibling are NOT mined — a rejected
    answer with no chosen reference is not a preference pair (fail-visible
    count in the returned log, not a fabricated pair).
    """
    by_task: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rec in records:
        by_task.setdefault(str(rec.get("task_id", "")), {"passed": [], "failed": []})
        slot = "passed" if rec.get("passed") else "failed"
        if _valid_messages(rec.get("messages")):
            by_task[str(rec.get("task_id", ""))][slot].append(rec)
    pairs = []
    unpaired_failures = 0
    for task_id, slots in sorted(by_task.items()):
        if not slots["passed"]:
            unpaired_failures += len(slots["failed"])
            continue
        chosen_rec = slots["passed"][0]
        prompt = str(chosen_rec.get("prompt") or "")
        if not prompt:
            unpaired_failures += len(slots["failed"])
            continue
        chosen = _assistant_text(chosen_rec["messages"])
        for failed_rec in slots["failed"]:
            rejected = _assistant_text(failed_rec["messages"])
            if not chosen.strip() or not rejected.strip():
                unpaired_failures += 1
                continue
            pairs.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "task_id": task_id,
                }
            )
    if unpaired_failures:
        logger.warning(
            "%d failed trajectories not mined (no passed sibling for their task)",
            unpaired_failures,
        )
    logger.info("mined %d DPO preference pairs from %d tasks", len(pairs), len(by_task))
    return pairs


def _per_type_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for rec in records:
        ttype = str(rec.get("task_type", "unknown"))
        slot = "passed" if rec.get("passed") else "failed"
        entry = counts.setdefault(ttype, {"total": 0, "passed": 0, "failed": 0})
        entry["total"] += 1
        entry[slot] += 1
    return counts


def export_arms(records: list[dict[str, Any]]) -> list[str]:
    """Distinct arm names in an export (sorted; missing arm -> 'unknown')."""
    return sorted({str(r.get("arm") or "unknown") for r in records})


def dpo_structurally_empty(records: list[dict[str, Any]]) -> bool:
    """True when the export can never yield a DPO pair: every task_id has
    exactly one trajectory, so no task has a passed+failed sibling (#78).
    This is the shape of every single-arm fusion-eval export — one run of
    one arm executes each task once.
    """
    counts: dict[str, int] = {}
    for rec in records:
        counts[str(rec.get("task_id", ""))] = (
            counts.get(str(rec.get("task_id", "")), 0) + 1
        )
    return bool(counts) and all(c == 1 for c in counts.values())


def build_from_eval_dataset(
    export_path: Path,
    output_path: Path,
    dpo_output_path: Path | None = None,
    dpo_allow_empty: bool = False,
) -> dict[str, Any]:
    """Full ingestion: export -> SFT dataset (+ optional DPO pairs) + manifest.

    The manifest is written next to the SFT dataset as `<output>.from_eval.json`
    and carries the export sha256 — SFTTrainer picks it up automatically so
    every eval-fed run records its source (#76 requirement 3).

    DPO gate (#78): a single-arm export (one trajectory per task) is
    structurally unable to produce preference pairs, so `dpo_output_path`
    on such an export raises — pass `dpo_allow_empty=True` (CLI
    `--dpo-anyway`) to write the empty output anyway.
    """
    records, digest = load_eval_export(export_path)
    sft_samples = eval_records_to_sft(records)
    if not sft_samples:
        raise EvalIngestError(
            f"no passed trajectories with valid messages in {export_path} — "
            "nothing to train on (use a passed-inclusive export)"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for sample in sft_samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("wrote %d SFT samples to %s", len(sft_samples), output_path)

    dpo_pairs: list[dict[str, Any]] = []
    dpo_empty = False
    if dpo_output_path is not None:
        if dpo_structurally_empty(records) and not dpo_allow_empty:
            raise EvalIngestError(
                "DPO output requested but this export is single-arm "
                f"(arms: {export_arms(records)}) — every task has exactly one "
                "trajectory, so no passed/failed siblings exist and mining is "
                "structurally empty. Merge several single-arm exports into "
                "one JSONL (tasks then have passed/failed siblings across "
                "runs), or run a multi-arm eval once fusion-eval supports "
                "it (fusion-eval#17), or pass --dpo-anyway "
                "to write the empty DPO file anyway."
            )
        dpo_empty = dpo_structurally_empty(records)
        if dpo_empty:
            logger.warning(
                "single-arm export: DPO output will be empty by construction "
                "(no task has a passed+failed sibling pair)"
            )
        dpo_pairs = mine_preference_pairs(records)
        dpo_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dpo_output_path, "w", encoding="utf-8") as fh:
            for pair in dpo_pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("wrote %d DPO pairs to %s", len(dpo_pairs), dpo_output_path)

    manifest = {
        "source": "fusion-eval",
        "export_path": str(export_path),
        "export_sha256": digest,
        "sft_path": str(output_path),
        "sft_samples": len(sft_samples),
        "dpo_path": str(dpo_output_path) if dpo_output_path else "",
        "dpo_pairs": len(dpo_pairs),
        "dpo_structurally_empty": dpo_empty,
        "arms": export_arms(records),
        "records_total": len(records),
        "records_passed": sum(1 for r in records if r.get("passed")),
        "records_failed": sum(1 for r in records if not r.get("passed")),
        "per_type": _per_type_counts(records),
        "lineage": {
            "source": "fusion-eval",
            "source_path": str(export_path),
            "source_sha256": digest,
            "transforms": ["eval_records_to_sft"],
            "output": str(output_path),
        },
    }
    manifest_path = Path(str(output_path) + MANIFEST_SUFFIX)
    tmp_path = Path(str(manifest_path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, manifest_path)
    logger.info("wrote manifest %s (export sha256=%s)", manifest_path, digest[:12])
    return manifest


def find_eval_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    """Return the from-eval manifest for a dataset file, or None.

    Checked locations: `<dataset>.from_eval.json` next to the file, and (for
    hub datasets resolved into `{hub}/datasets/<name>.jsonl`) the same suffix
    on the resolved file.
    """
    if not dataset_path:
        return None
    p = Path(dataset_path)
    candidates = [Path(str(p) + MANIFEST_SUFFIX)]
    if p.suffix == ".jsonl":
        candidates.append(p.with_name(p.name + MANIFEST_SUFFIX))
    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("unreadable eval manifest %s: %s", cand, exc)
                return None
            if isinstance(data, dict) and data.get("export_sha256"):
                logger.info(
                    "dataset %s is eval-fed (export sha256=%s)",
                    p,
                    str(data.get("export_sha256"))[:12],
                )
                return data
    return None
