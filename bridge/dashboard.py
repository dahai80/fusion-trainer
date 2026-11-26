"""Training-loop dashboard backend (issue #93).

Pure payload builders for the /dashboard page; every section returns an
explicit {"available": bool, "reason": str, ...} shape so the page renders
honest empty states instead of mock data. Data sources (live, no defaults):

  flywheel registry  {FUSION_TRAINER_CACHE}/flywheel.json   rounds + gate
  run records        RunManager.list_runs()                  runs panel
  rollout sinks      FUSION_TRAINER_ROLLOUT_SINKS (os.pathsep-separated
                     files or directories of *.jsonl, rollout-tap schema)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ROLLOUT_SINKS_ENV = "FUSION_TRAINER_ROLLOUT_SINKS"
_RECENT_WINDOW_S = 7 * 86400
_MAX_RUNS = 12


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def flywheel_section(cache_path: str) -> dict[str, Any]:
    from fusion_trainer.flywheel import (
        incumbent_pass_rate,
        load_registry,
        registry_path,
    )

    if not registry_path(cache_path).is_file():
        return {
            **_unavailable(
                "no flywheel registry yet — run `ft flywheel record` after an eval round"
            ),
            "rounds": [],
            "incumbent_pass_rate": None,
        }
    try:
        registry = load_registry(cache_path)
    except FileNotFoundError:
        return {
            **_unavailable(
                "no flywheel registry yet — run `ft flywheel record` after an eval round"
            ),
            "rounds": [],
            "incumbent_pass_rate": None,
        }
    except Exception as exc:
        return {
            **_unavailable(f"flywheel registry unreadable: {exc}"),
            "rounds": [],
            "incumbent_pass_rate": None,
        }
    rounds: list[dict[str, Any]] = []
    for r in registry.get("rounds", []):
        if not isinstance(r, dict):
            continue
        rounds.append(
            {
                "label": str(r.get("label", "")),
                "arm": str(r.get("arm", "")),
                "model": str(r.get("model", "")),
                "pass_rate": r.get("pass_rate"),
                "passed": r.get("passed"),
                "total": r.get("total"),
                "by_type": r.get("by_type") or {},
                "baseline": bool(r.get("baseline")),
                "status": str(r.get("status", "promoted")),
                "regression": r.get("regression"),
                "recorded_at": str(r.get("recorded_at", "")),
            }
        )
    incumbent = incumbent_pass_rate(registry)
    out: dict[str, Any] = {
        "available": bool(rounds),
        "reason": "" if rounds else "flywheel registry exists but has no rounds",
        "rounds": rounds,
        "incumbent_pass_rate": incumbent,
    }
    if rounds:
        latest = rounds[-1]
        out["latest"] = latest
        # Latest round may lack per-type breakdown (e.g. DPO rounds) —
        # fall back to the most recent round that has one, and say which.
        by_type_round = next((r for r in reversed(rounds) if r.get("by_type")), None)
        by_type = (by_type_round or {}).get("by_type") or {}

        def _rate(v: Any) -> float | None:
            # registry stores either a bare rate or {pass_rate, passed, total}
            if isinstance(v, dict):
                v = v.get("pass_rate")
            return v if isinstance(v, (int, float)) else None

        out["by_type"] = [
            {"type": str(k), "pass_rate": rate}
            for k, v in sorted(by_type.items())
            if (rate := _rate(v)) is not None
        ]
        out["by_type_round"] = by_type_round.get("label", "") if by_type_round else ""
        progression = [
            {
                "label": r["label"],
                "pass_rate": r["pass_rate"],
                "status": r["status"],
            }
            for r in rounds
            if isinstance(r.get("pass_rate"), (int, float))
        ]
        out["progression"] = progression
        if latest.get("status") == "discarded":
            out["gate_note"] = (
                f"round {latest['label']} was DISCARDED by the gate "
                f"(pass_rate regressed vs incumbent) — checkpoint not promoted"
            )
    return out


def rollout_section() -> dict[str, Any]:
    raw = os.environ.get(ROLLOUT_SINKS_ENV, "").strip()
    if not raw:
        return {
            **_unavailable(
                f"no rollout sinks configured — set {ROLLOUT_SINKS_ENV} "
                "(os.pathsep-separated JSONL files or directories)"
            ),
            "total": 0,
            "recent": 0,
            "signals": {},
            "positive_share": None,
        }
    now = time.time()
    invalid = 0
    recent = 0
    scored = 0
    positive = 0
    signals: dict[str, int] = {}
    files: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        p = Path(part).expanduser()
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            logger.warning("rollout sink %s does not exist — skipped", p)
    tap_records: list[dict] = []
    sidecar_outcomes: dict[str, dict] = {}
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        invalid += 1
                        continue
                    if not (
                        isinstance(rec, dict)
                        and rec.get("schema_version") == 1
                        and rec.get("rollout_id")
                        and isinstance(rec.get("messages"), list)
                    ):
                        invalid += 1
                        continue
                    # Outcome sidecar record (#97): carries an outcome and no
                    # `source` (the gateway tap contract always writes source
                    # and never outcome). Counted via the join, never as a
                    # separate trajectory.
                    if isinstance(rec.get("outcome"), dict) and not rec.get("source"):
                        sidecar_outcomes[str(rec["rollout_id"])] = rec["outcome"]
                        continue
                    tap_records.append(rec)
        except OSError as exc:
            logger.warning("rollout sink %s unreadable: %s", f, exc)
    total = len(tap_records)
    for rec in tap_records:
        if not isinstance(rec.get("outcome"), dict):
            outcome = sidecar_outcomes.get(str(rec.get("rollout_id")))
            if outcome is not None:
                rec = {**rec, "outcome": outcome}
        created = rec.get("created_at")
        if isinstance(created, str):
            try:
                ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                if now - ts <= _RECENT_WINDOW_S:
                    recent += 1
            except ValueError:
                pass
        outcome = rec.get("outcome")
        signal = "unobserved"
        if isinstance(outcome, dict) and outcome.get("signal"):
            signal = str(outcome["signal"])
        signals[signal] = signals.get(signal, 0) + 1
        if isinstance(outcome, dict):
            score = outcome.get("score")
            if not isinstance(score, (int, float)):
                score = {
                    "task_success": 1.0,
                    "user_accepted": 1.0,
                    "user_edited": 0.5,
                    "user_rejected": 0.0,
                    "error": 0.0,
                }.get(signal)
            if isinstance(score, (int, float)):
                scored += 1
                if float(score) >= 1.0:
                    positive += 1
    if sidecar_outcomes:
        logger.info(
            "rollout sinks: %d sidecar outcome record(s) joined by rollout_id",
            len(sidecar_outcomes),
        )
    if invalid:
        logger.warning(
            "rollout sinks: %d invalid line(s) skipped (schema_version==1 + "
            "rollout_id + messages required)",
            invalid,
        )
    if not files:
        return {
            **_unavailable(f"{ROLLOUT_SINKS_ENV} set but no readable sink files"),
            "total": 0,
            "recent": 0,
            "signals": {},
            "positive_share": None,
        }
    return {
        "available": total > 0,
        "reason": "" if total else "sinks readable but contain no valid records",
        "total": total,
        "recent": recent,
        "signals": dict(sorted(signals.items())),
        "invalid_lines": invalid,
        "positive_share": (positive / scored) if scored else None,
        "sinks": [str(f) for f in files],
    }


def next_step(flywheel: dict[str, Any], rollouts: dict[str, Any]) -> str:
    if not flywheel.get("available"):
        return "record an eval round with `ft flywheel record` to seed the loop"
    if flywheel.get("latest", {}).get("status") == "discarded":
        return (
            "last round was discarded by the gate — mine a fresh eval export, "
            "do not re-consume the regressed dataset"
        )
    if not rollouts.get("available") or not rollouts.get("total"):
        return "configure a rollout tap sink to grow the trajectory pool"
    return "chain the promoted checkpoint: --init-adapter-path for the next round"


def build_payload(cache_path: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    flywheel = flywheel_section(cache_path)
    rollouts = rollout_section()
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "flywheel": flywheel,
        "runs": runs[:_MAX_RUNS],
        "rollouts": rollouts,
        "reward_composition": _unavailable(
            "run records do not tag reward source — office validator vs "
            "outcome-signal vs LLM-judge shares need reward provenance in "
            "run records (tracked with #90 live acceptance)"
        ),
        "next_step": next_step(flywheel, rollouts),
    }
    return payload
