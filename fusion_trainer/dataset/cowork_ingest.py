"""Cowork flow-export ingestion (#86 flywheel entry 3, capability plan §3.1).

Consumes fusion-cowork flow exports (JSONL, one record per DAG node run):

    {"schema_version": 1, "flow_id": "...", "node_id": "...",
     "task_type": "...", "arm": "...", "model": "...",
     "passed": true/false, "prompt": "...",
     "messages": [{"role","content"}, ...], "final_output": "..."}

Cowork exports are converted to the eval-trajectory record shape
(task_id = "<flow_id>/<node_id>") and mined by the SAME code path as
`ft datasets from-eval` — SFT from passed records, DPO pairs from
passed/failed siblings. DPO was shown data-thin, not algorithm-thin
(#83): cowork multi-node flows are the volume source for multi-arm
exports. Sidecar manifest carries the export sha256 + lineage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .eval_ingest import (
    EVAL_EXPORT_SCHEMA_VERSION,
    _valid_messages,
    build_from_eval_dataset,
    sha256_file,
)

logger = logging.getLogger(__name__)

COWORK_MANIFEST_SUFFIX = ".from_cowork.json"
_COWORK_REQUIRED_KEYS = ("schema_version", "flow_id", "node_id", "passed", "messages")


class CoworkIngestError(Exception):
    pass


def load_cowork_export(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load cowork flow export, validate, return (eval-shaped records, sha256)."""
    if not path.is_file():
        raise CoworkIngestError(f"cowork export not found: {path}")
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
            except ValueError:
                bad += 1
                logger.warning("skip cowork line %d: not JSON", lineno)
                continue
            if not isinstance(rec, dict):
                bad += 1
                logger.warning("skip cowork line %d: not an object", lineno)
                continue
            version = rec.get("schema_version")
            if version != EVAL_EXPORT_SCHEMA_VERSION:
                bad += 1
                logger.warning(
                    "skip cowork line %d: schema_version=%r != %d",
                    lineno,
                    version,
                    EVAL_EXPORT_SCHEMA_VERSION,
                )
                continue
            missing = [k for k in _COWORK_REQUIRED_KEYS if k not in rec]
            if missing:
                bad += 1
                logger.warning(
                    "skip cowork line %d (node %s/%s): missing keys %s",
                    lineno,
                    rec.get("flow_id", "?"),
                    rec.get("node_id", "?"),
                    missing,
                )
                continue
            if not _valid_messages(rec.get("messages")):
                bad += 1
                logger.warning(
                    "skip cowork line %d (node %s/%s): invalid messages",
                    lineno,
                    rec.get("flow_id", "?"),
                    rec.get("node_id", "?"),
                )
                continue
            records.append(rec)
    if not records:
        raise CoworkIngestError(
            f"no valid cowork records in {path} (skipped {bad} bad lines)"
        )
    if bad:
        logger.warning("loaded %d cowork records, skipped %d bad lines", len(records), bad)
    logger.info(
        "loaded %d cowork records from %s (sha256=%s)", len(records), path, digest[:12]
    )
    return records, digest


def cowork_records_to_eval(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map cowork node runs onto the eval-trajectory shape the miner knows."""
    out = []
    for rec in records:
        task_id = f"{rec.get('flow_id', '?')}/{rec.get('node_id', '?')}"
        out.append(
            {
                "schema_version": EVAL_EXPORT_SCHEMA_VERSION,
                "task_id": task_id,
                "task_type": rec.get("task_type") or "cowork",
                "arm": rec.get("arm") or "default",
                "model": rec.get("model", ""),
                "passed": bool(rec.get("passed")),
                "prompt": str(rec.get("prompt") or ""),
                "messages": rec["messages"],
                "final_diff": rec.get("final_output", ""),
            }
        )
    logger.info("mapped %d cowork records to eval-trajectory shape", len(out))
    return out


def build_from_cowork_dataset(
    export_path: Path,
    output_path: Path,
    dpo_output_path: Path | None = None,
    dpo_allow_empty: bool = False,
) -> dict[str, Any]:
    """Full ingestion: cowork export -> SFT (+ optional DPO) + manifest + lineage.

    Reuses build_from_eval_dataset end-to-end (validation, mining, atomic
    sidecar); the cowork sidecar adds the cowork export sha256 + lineage so
    the hub lineage registry (T4) can trace cowork -> eval-shape -> dataset.
    """
    _records, digest = load_cowork_export(export_path)
    eval_shaped = cowork_records_to_eval(_records)
    eval_path = export_path.with_name(export_path.name + ".evalshaped.jsonl")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as fh:
        for rec in eval_shaped:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("staged eval-shaped records to %s", eval_path)
    try:
        manifest = build_from_eval_dataset(
            eval_path, output_path, dpo_output_path, dpo_allow_empty=dpo_allow_empty
        )
    finally:
        eval_path.unlink(missing_ok=True)
        logger.info("removed staged eval-shaped file %s", eval_path)
    manifest["source"] = "fusion-cowork"
    manifest["cowork_export_path"] = str(export_path)
    manifest["cowork_export_sha256"] = digest
    manifest["lineage"] = {
        "source": "fusion-cowork",
        "source_path": str(export_path),
        "source_sha256": digest,
        "transforms": ["cowork_records_to_eval", "eval_records_to_sft"],
        "output": str(output_path),
    }
    sidecar = Path(str(output_path) + COWORK_MANIFEST_SUFFIX)
    tmp = Path(str(sidecar) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    tmp.replace(sidecar)
    logger.info("wrote cowork manifest %s", sidecar)
    return manifest


def find_cowork_manifest(dataset_path: str | Path) -> dict[str, Any] | None:
    """Return the from_cowork manifest for a dataset file, or None."""
    if not dataset_path:
        return None
    cand = Path(str(dataset_path) + COWORK_MANIFEST_SUFFIX)
    if not cand.is_file():
        return None
    try:
        data = json.loads(cand.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("unreadable cowork manifest %s: %s", cand, exc)
        return None
    if isinstance(data, dict) and data.get("cowork_export_sha256"):
        return data
    return None


# ---------------------------------------------------------------------------
# Steps-shape ingestion (#136, handoff doc §1) — cowork run exports whose
# records are DAG steps, not chat messages: per run {"run_id", "task",
# "success", "classification"?, "steps": [{"node_id", "node_name",
# "output_data", "error", "status", "upstream"[]}]}.
# ---------------------------------------------------------------------------

_STEPS_REQUIRED_KEYS = ("run_id", "task", "success", "steps")
_STEP_FAIL_STATUS = {"failed", "error", "failure"}
_STEP_SKIP_STATUS = {"skipped", "skip", "gated", "gated_out", "cancelled"}
_CLASSIFICATION_PASS = {"public", "internal"}


def detect_cowork_export_shape(path: Path) -> str:
    """Deterministically detect the export shape from the first valid line.

    "steps" = per-run DAG-step records (#136); "node-runs" = legacy per-node
    flow-export records (T4). Mixed/ambiguous files raise — never guess past
    the first record.
    """
    if not path.is_file():
        raise CoworkIngestError(f"cowork export not found: {path}")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            if "steps" in rec and "run_id" in rec:
                return "steps"
            if "messages" in rec and "passed" in rec:
                return "node-runs"
            raise CoworkIngestError(
                f"unrecognized cowork export record shape in {path}: keys "
                f"{sorted(rec.keys())[:8]}"
            )
    raise CoworkIngestError(f"no records in cowork export {path}")


def load_cowork_runs(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load a steps-shape cowork run export, validate, return (runs, sha256)."""
    if not path.is_file():
        raise CoworkIngestError(f"cowork export not found: {path}")
    digest = sha256_file(path)
    runs: list[dict[str, Any]] = []
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
                logger.warning("skip cowork run line %d: not JSON", lineno)
                continue
            if not isinstance(rec, dict):
                bad += 1
                logger.warning("skip cowork run line %d: not an object", lineno)
                continue
            missing = [k for k in _STEPS_REQUIRED_KEYS if k not in rec]
            if missing:
                bad += 1
                logger.warning(
                    "skip cowork run line %d (run %s): missing keys %s",
                    lineno,
                    rec.get("run_id", "?"),
                    missing,
                )
                continue
            steps = rec.get("steps")
            if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
                bad += 1
                logger.warning(
                    "skip cowork run line %d (run %s): steps must be a list of objects",
                    lineno,
                    rec.get("run_id", "?"),
                )
                continue
            runs.append(rec)
    if not runs:
        raise CoworkIngestError(
            f"no valid cowork run records in {path} (skipped {bad} bad lines)"
        )
    if bad:
        logger.warning("loaded %d cowork runs, skipped %d bad lines", len(runs), bad)
    logger.info(
        "loaded %d cowork runs from %s (sha256=%s)", len(runs), path, digest[:12]
    )
    return runs, digest


def _topo_order_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Kahn topological sort preserving the recorded order as tiebreak.

    Returns None on a cycle (run is dropped loudly by the caller). Steps
    without ids keep their recorded position; upstream references unknown
    ids are ignored for ordering (kept in the step itself).
    """
    ids: list[str] = []
    for idx, s in enumerate(steps):
        ids.append(str(s.get("node_id", "") or "") or f"__idx_{idx}")
    known = set(ids)
    indeg = {i: 0 for i in range(len(steps))}
    dependents: dict[int, list[int]] = {i: [] for i in range(len(steps))}
    for idx, s in enumerate(steps):
        for u in s.get("upstream", None) or []:
            u = str(u)
            if u in known:
                up = ids.index(u)
                indeg[idx] += 1
                dependents[up].append(idx)
    queue = [i for i in range(len(steps)) if indeg[i] == 0]
    ordered: list[dict[str, Any]] = []
    seen = 0
    while queue:
        idx = queue.pop(0)
        seen += 1
        ordered.append(steps[idx])
        for dep in dependents[idx]:
            indeg[dep] -= 1
            if indeg[dep] == 0:
                queue.append(dep)
    if seen != len(steps):
        logger.warning("topo sort: cycle detected across %d steps", len(steps))
        return None
    return ordered


def cowork_runs_to_sft(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert successful cowork runs to chat-SFT samples (#136 §1 contract).

    Deterministic (rule 5): one sample per successful run, messages[0] is
    the run task, steps in topological order become assistant tool-call
    turns + tool messages (the #88 structured shapes the reward plugins and
    struggle judge parse). Filters, all loud and counted:
      - unsuccessful runs dropped;
      - `confidential` / missing / unknown classification dropped
        (identity#119 trainer default, comment 2026-11-28);
      - a successful run carrying a failed step is an inconsistent record
        and dropped;
      - skipped/gated steps dropped with their downstream continuations
        kept (only successful runs reach here);
      - cyclic step graphs dropped.
    """
    stats = {
        "runs_total": len(runs),
        "runs_kept": 0,
        "dropped_unsuccessful": 0,
        "dropped_classification": 0,
        "dropped_inconsistent": 0,
        "dropped_cycle": 0,
        "steps_skipped": 0,
    }
    samples: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("run_id", "") or "?")
        if not run.get("success"):
            stats["dropped_unsuccessful"] += 1
            logger.info("run %s dropped: unsuccessful", run_id)
            continue
        classification = str(run.get("classification", "") or "")
        if classification not in _CLASSIFICATION_PASS:
            stats["dropped_classification"] += 1
            logger.warning(
                "run %s dropped: classification %r not in %s (identity#119 "
                "trainer default: unclassifiable never silently included)",
                run_id,
                classification or "<missing>",
                sorted(_CLASSIFICATION_PASS),
            )
            continue
        executed = []
        failed = 0
        for s in run["steps"]:
            status = str(s.get("status", "") or "").lower()
            if status in _STEP_FAIL_STATUS:
                failed += 1
            elif status in _STEP_SKIP_STATUS:
                stats["steps_skipped"] += 1
                logger.info("run %s: skipped/gated step %s dropped", run_id, s.get("node_id", "?"))
            else:
                executed.append(s)
        if failed:
            stats["dropped_inconsistent"] += 1
            logger.warning(
                "run %s dropped: successful run carries %d failed step(s) — "
                "inconsistent record",
                run_id,
                failed,
            )
            continue
        ordered = _topo_order_steps(executed)
        if ordered is None:
            stats["dropped_cycle"] += 1
            logger.warning("run %s dropped: cyclic step graph", run_id)
            continue
        messages: list[dict[str, Any]] = [{"role": "user", "content": str(run.get("task", ""))}]
        for s in ordered:
            name = str(s.get("node_name", "") or s.get("node_id", "") or "")
            args = s.get("input_bindings", None) or {}
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(s.get("node_id", "") or name),
                    "content": json.dumps(
                        s.get("output_data", None), ensure_ascii=False, default=str
                    ),
                }
            )
        samples.append({"messages": messages, "run_id": run_id})
        stats["runs_kept"] += 1
    logger.info(
        "cowork runs -> SFT: %d kept of %d (unsuccessful=%d classification=%d "
        "inconsistent=%d cycle=%d skipped_steps=%d)",
        stats["runs_kept"],
        stats["runs_total"],
        stats["dropped_unsuccessful"],
        stats["dropped_classification"],
        stats["dropped_inconsistent"],
        stats["dropped_cycle"],
        stats["steps_skipped"],
    )
    return samples, stats


def build_from_cowork_steps_dataset(export_path: Path, output_path: Path) -> dict[str, Any]:
    """Steps-shape ingestion: cowork run export -> SFT dataset + manifest (#136).

    Consumes per-run DAG-step records (handoff doc §1), converts via
    cowork_runs_to_sft, writes messages-JSONL + atomic sidecar manifest
    carrying the export sha256 and the full filter accounting.
    """
    runs, digest = load_cowork_runs(export_path)
    samples, stats = cowork_runs_to_sft(runs)
    if not samples:
        raise CoworkIngestError(
            f"no successful cowork runs survived the filters in {export_path} "
            f"({json.dumps(stats, ensure_ascii=False)})"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(output_path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    manifest = {
        "source": "fusion-cowork",
        "shape": "steps",
        "sft_path": str(output_path),
        "sft_samples": len(samples),
        "cowork_export_path": str(export_path),
        "cowork_export_sha256": digest,
        "stats": stats,
        "lineage": {
            "source": "fusion-cowork",
            "source_path": str(export_path),
            "source_sha256": digest,
            "transforms": ["cowork_runs_to_sft"],
            "output": str(output_path),
        },
    }
    sidecar = Path(str(output_path) + COWORK_MANIFEST_SUFFIX)
    tmp = Path(str(sidecar) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    tmp.replace(sidecar)
    logger.info("wrote cowork steps manifest %s", sidecar)
    return manifest
