"""Flywheel round registry (#76): trained_checkpoint -> eval run -> pass_rate.

A tiny JSON index at `{cache_path}/flywheel.json` (atomic tmp+replace write,
same discipline as RunManager._persist_meta). Each round records:
    label / checkpoint (adapter dir or base model id) / model / eval_run_id /
    pass_rate / per-type rates / source export sha256 / dataset path /
    trainer run id / recorded_at

`ft flywheel record` appends a round from a fusion-eval summary.json;
`ft flywheel status` prints the pass-rate-vs-rounds curve. The curve is the
acceptance evidence for #76: >=3 points, sources traceable to export hashes.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "flywheel.json"


class FlywheelError(Exception):
    pass


def registry_path(cache_path: str) -> Path:
    return Path(cache_path) / REGISTRY_FILENAME


def load_registry(cache_path: str) -> dict[str, Any]:
    path = registry_path(cache_path)
    if not path.is_file():
        return {"rounds": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FlywheelError(f"unreadable flywheel registry {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("rounds"), list):
        raise FlywheelError(
            f"malformed flywheel registry {path}: expected {{'rounds': [...]}}"
        )
    return data


def save_registry(cache_path: str, registry: dict[str, Any]) -> Path:
    path = registry_path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    logger.info(
        "flywheel registry saved: %d rounds -> %s",
        len(registry.get("rounds", [])),
        path,
    )
    return path


def parse_eval_summary(summary_path: Path, arm: str = "") -> dict[str, Any]:
    """Parse a fusion-eval run summary.json into flywheel round fields.

    Layout (fusion-eval results/<run>/summary.json):
        {"run_id", "created_at", "models": {arm: model_id},
         "arms": {arm: {"total", "passed", "pass_rate", "by_type": {...}}}}
    With multiple arms and no explicit --arm, the first arm wins and a loud
    warning names the choice (deterministic: sorted-key first).
    """
    if not summary_path.is_file():
        raise FlywheelError(f"eval summary not found: {summary_path}")
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FlywheelError(f"unreadable eval summary {summary_path}: {exc}") from exc
    arms = data.get("arms") or {}
    if not isinstance(arms, dict) or not arms:
        raise FlywheelError(f"no arms in eval summary {summary_path}")
    if arm:
        if arm not in arms:
            raise FlywheelError(
                f"arm {arm!r} not in eval summary {summary_path} (arms: {sorted(arms)})"
            )
        chosen = arm
    else:
        chosen = sorted(arms.keys())[0]
        if len(arms) > 1:
            logger.warning(
                "eval summary has %d arms (%s); using %r — pass --arm to pick another",
                len(arms),
                sorted(arms),
                chosen,
            )
    arm_data = arms[chosen]
    models = data.get("models") or {}
    return {
        "eval_run_id": str(data.get("run_id", "")),
        "arm": chosen,
        "model": str(models.get(chosen, "")),
        "pass_rate": float(arm_data.get("pass_rate", 0.0)),
        "passed": int(arm_data.get("passed", 0)),
        "total": int(arm_data.get("total", 0)),
        "by_type": {
            ttype: {
                "pass_rate": float(stats.get("pass_rate", 0.0)),
                "passed": int(stats.get("passed", 0)),
                "total": int(stats.get("total", 0)),
            }
            for ttype, stats in (arm_data.get("by_type") or {}).items()
            if isinstance(stats, dict)
        },
    }


def record_round(
    cache_path: str,
    *,
    label: str,
    checkpoint: str,
    summary_path: Path,
    arm: str = "",
    export_sha256: str = "",
    export_path: str = "",
    dataset_path: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    """Append a flywheel round parsed from a fusion-eval summary."""
    parsed = parse_eval_summary(summary_path, arm=arm)
    if not parsed["eval_run_id"]:
        raise FlywheelError(f"eval summary {summary_path} carries no run_id")
    if not export_sha256 and export_path:
        from .dataset.eval_ingest import sha256_file

        export = Path(export_path)
        if not export.is_file():
            raise FlywheelError(f"export not found: {export}")
        export_sha256 = sha256_file(export)
    registry = load_registry(cache_path)
    entry = {
        "label": label,
        "checkpoint": checkpoint,
        "run_id": run_id,
        "eval_run_id": parsed["eval_run_id"],
        "arm": parsed["arm"],
        "model": parsed["model"],
        "pass_rate": parsed["pass_rate"],
        "passed": parsed["passed"],
        "total": parsed["total"],
        "by_type": parsed["by_type"],
        "export_sha256": export_sha256,
        "export_path": str(export_path),
        "dataset_path": dataset_path,
        "summary_path": str(summary_path),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    if entry["export_sha256"]:
        for existing in registry["rounds"]:
            if (
                existing.get("export_sha256") == entry["export_sha256"]
                and existing.get("eval_run_id") == entry["eval_run_id"]
            ):
                logger.warning(
                    "round (export %s, eval run %s) already recorded as %r — duplicate appended",
                    entry["export_sha256"][:12],
                    entry["eval_run_id"],
                    existing.get("label"),
                )
    registry["rounds"].append(entry)
    save_registry(cache_path, registry)
    logger.info(
        "recorded flywheel round %r: eval_run=%s pass_rate=%.4f",
        label,
        entry["eval_run_id"],
        entry["pass_rate"],
    )
    return entry


def curve_rows(cache_path: str) -> list[dict[str, Any]]:
    """Rounds with pass_rate + delta vs previous round (trend evidence)."""
    rounds = load_registry(cache_path)["rounds"]
    rows = []
    prev: float | None = None
    for i, r in enumerate(rounds):
        rate = float(r.get("pass_rate", 0.0))
        delta = None if prev is None else rate - prev
        rows.append(
            {
                "index": i,
                "label": r.get("label", ""),
                "checkpoint": r.get("checkpoint", ""),
                "model": r.get("model", ""),
                "eval_run_id": r.get("eval_run_id", ""),
                "pass_rate": rate,
                "delta": delta,
                "export_sha256": r.get("export_sha256", ""),
                "by_type": r.get("by_type", {}),
                "recorded_at": r.get("recorded_at", ""),
            }
        )
        prev = rate
    return rows


def format_status(cache_path: str) -> str:
    """Render the pass-rate-vs-round curve + per-type table as text."""
    rows = curve_rows(cache_path)
    if not rows:
        return (
            "No flywheel rounds recorded. Start with:\n"
            "  ft flywheel record --label baseline --summary <fusion-eval summary.json> \\\n"
            "      --checkpoint <base model id>"
        )
    lines = []
    lines.append(f"Flywheel curve ({len(rows)} round{'s' if len(rows) != 1 else ''}):")
    lines.append(
        f"  {'#':>2}  {'label':<18}  {'pass_rate':>9}  {'delta':>7}  "
        f"{'export':<12}  eval_run / checkpoint"
    )
    for row in rows:
        delta = "" if row["delta"] is None else f"{row['delta']:+.4f}"
        export = row["export_sha256"][:12] if row["export_sha256"] else "(baseline)"
        lines.append(
            f"  {row['index']:>2}  {row['label']:<18}  {row['pass_rate']:>9.4f}  {delta:>7}  "
            f"{export:<12}  {row['eval_run_id']} / {row['checkpoint'] or row['model']}"
        )
    first_types = rows[0]["by_type"]
    if first_types:
        all_types = sorted({t for row in rows for t in row["by_type"]})
        lines.append("")
        lines.append("Per-type pass rates:")
        header = f"  {'type':<20}" + "".join(
            f"  {row['label'][:14]:>14}" for row in rows
        )
        lines.append(header)
        for ttype in all_types:
            cells = []
            for row in rows:
                stats = row["by_type"].get(ttype)
                cells.append(f"{stats['pass_rate']:>14.4f}" if stats else f"{'-':>14}")
            lines.append(f"  {ttype:<20}" + "".join(f"  {c:>14}" for c in cells))
    if len(rows) >= 2:
        trend = rows[-1]["pass_rate"] - rows[0]["pass_rate"]
        direction = "UP" if trend > 0 else ("DOWN" if trend < 0 else "FLAT")
        lines.append("")
        lines.append(
            f"Trend (first -> last): {trend:+.4f} ({direction}). "
            "If not up, the experiment report must say why not (#76 acceptance)."
        )
    return "\n".join(lines)
