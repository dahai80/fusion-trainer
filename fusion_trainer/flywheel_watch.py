"""Auto-record flywheel rounds from landing eval summaries (issue #95).

Scans a fusion-eval results directory for summary.json files and records a
gated flywheel round for each one not yet consumed. Consumption state lives
in `{cache_path}/flywheel_watch.json` (summary path + sha256), so re-runs
never double-record. Deterministic: sorted scan order, sorted arm pick via
parse_eval_summary, gate verdict from record_round (#82 discipline).

Usage: ft flywheel watch --dir <fusion-eval results dir> [--gate] [--once]
       (default polls every --interval seconds until Ctrl-C)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILENAME = "flywheel_watch.json"


def _state_path(cache_path: str) -> Path:
    return Path(cache_path) / STATE_FILENAME


def load_state(cache_path: str) -> dict[str, Any]:
    p = _state_path(cache_path)
    if not p.is_file():
        return {"consumed": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("consumed"), dict):
            return data
    except (OSError, ValueError) as exc:
        logger.warning("flywheel watch state unreadable (%s) — starting fresh", exc)
    return {"consumed": {}}


def save_state(cache_path: str, state: dict[str, Any]) -> None:
    p = _state_path(cache_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    import os

    os.replace(tmp, p)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_summaries(watch_dir: Path) -> list[Path]:
    if not watch_dir.is_dir():
        return []
    return sorted(watch_dir.glob("*/summary.json"))


def record_new_summaries(
    watch_dir: Path,
    cache_path: str,
    gate: bool = True,
    checkpoint: str = "",
    run_id: str = "",
) -> list[dict[str, Any]]:
    from fusion_trainer.flywheel import FlywheelError, record_round

    state = load_state(cache_path)
    consumed = state["consumed"]
    recorded: list[dict[str, Any]] = []
    for summary in scan_summaries(watch_dir):
        key = str(summary)
        try:
            digest = _sha256(summary)
        except OSError as exc:
            logger.warning("watch: unreadable summary %s: %s", summary, exc)
            continue
        seen = consumed.get(key)
        if seen and seen.get("sha256") == digest:
            continue
        label = summary.parent.name or f"auto-{digest[:8]}"
        try:
            entry = record_round(
                cache_path,
                label=label,
                checkpoint=checkpoint,
                summary_path=summary,
                run_id=run_id,
                gate=gate,
            )
        except FlywheelError as exc:
            # Not ready / malformed: leave unconsumed, retry on next pass.
            logger.warning("watch: skipped %s: %s", summary, exc)
            continue
        consumed[key] = {
            "sha256": digest,
            "label": label,
            "status": entry.get("status", "promoted"),
            "recorded_at": entry.get("recorded_at", ""),
        }
        recorded.append(entry)
        logger.info(
            "watch: auto-recorded round %r from %s (pass_rate %.4f, status %s)",
            label,
            summary,
            entry["pass_rate"],
            entry.get("status", "promoted"),
        )
    if recorded:
        save_state(cache_path, state)
    return recorded


def watch(
    watch_dir: Path,
    cache_path: str,
    gate: bool = True,
    once: bool = False,
    interval: float = 30.0,
    checkpoint: str = "",
) -> int:
    """Poll the watch dir; returns the number of rounds recorded. --once
    does a single pass (CI / cron friendly)."""
    if not watch_dir.is_dir():
        raise FileNotFoundError(f"watch dir not found: {watch_dir}")
    total = 0
    while True:
        total += len(
            record_new_summaries(watch_dir, cache_path, gate=gate, checkpoint=checkpoint)
        )
        if once:
            break
        time.sleep(max(1.0, interval))
    return total
