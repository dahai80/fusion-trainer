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
