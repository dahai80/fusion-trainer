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
    baseline: bool = False,
    gate: bool = False,
    gate_mode: str = "regression",
    floor_band: dict[str, Any] | None = None,
    claims_registry: Path | None = None,
    allow_reconsume: bool = False,
) -> dict[str, Any]:
    """Append a flywheel round parsed from a fusion-eval summary.

    Fresh-data discipline (#82): a training round (baseline=False) records
    the export it consumed; re-consuming an export another training round
    already consumed raises FlywheelError unless allow_reconsume=True (then
    the entry is flagged reconsumed, never silently accepted).

    Eval-gated promotion (#82): with gate=True the candidate is compared
    against the incumbent — the best pass_rate among prior non-discarded
    rounds. Regression -> status="discarded" (checkpoint NOT promoted,
    loud log); otherwise status="promoted". Historical rounds without a
    status field count as promoted.
    """
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
    if not baseline and export_sha256:
        consumed_by = _consumed_export_shas(registry)
        if export_sha256 in consumed_by:
            if not allow_reconsume:
                raise FlywheelError(
                    f"export {export_sha256[:12]} already consumed by round "
                    f"{consumed_by[export_sha256]!r} — fresh-data discipline "
                    "(#82) forbids re-training on consumed data. Mine a new "
                    "eval export or pass --allow-reconsume to override "
                    "(flagged, never silent)."
                )
            logger.warning(
                "round %r re-consumes export %s (already used by %r) — "
                "OVERRIDE via allow_reconsume, entry flagged reconsumed",
                label,
                export_sha256[:12],
                consumed_by[export_sha256],
            )
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
        "baseline": baseline,
        "consumed": not baseline,
        "reconsumed": (
            bool(export_sha256)
            and not baseline
            and export_sha256 in _consumed_export_shas(registry)
        ),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    if gate:
        incumbent = incumbent_pass_rate(registry)
        if incumbent is None:
            entry["status"] = "promoted"
            entry["incumbent_pass_rate"] = None
            entry["regression"] = None
            logger.info("gate: no prior rounds — round %r promoted by default", label)
        elif gate_mode == "wilson":
            # Phase-2 statistical gate (#134): Wilson CI non-overlap vs the
            # incumbent record (+ optional E-2 floor band), counts from the
            # eval summary. Falls back LOUDLY to the #82 point check when
            # either side lacks counts.
            from .promotion import gate_promotion

            decision = gate_promotion(
                entry, incumbent_entry(registry), floor=floor_band
            )
            entry["gate_mode"] = decision["gate_mode"]
            entry["candidate_ci"] = decision.get("candidate_ci")
            entry["incumbent_ci"] = decision.get("incumbent_ci")
            entry["floor_ci"] = decision.get("floor_ci")
            entry["gate_reason"] = decision["reason"]
            entry["incumbent_pass_rate"] = incumbent
            entry["regression"] = entry["pass_rate"] - incumbent
            entry["status"] = "promoted" if decision["promoted"] else "discarded"
            if decision["promoted"]:
                logger.info("gate PROMOTED round %r: %s", label, decision["reason"])
            else:
                logger.warning(
                    "gate DISCARDED round %r: %s — checkpoint NOT promoted",
                    label,
                    decision["reason"],
                )
        else:
            regression = entry["pass_rate"] - incumbent
            entry["incumbent_pass_rate"] = incumbent
            entry["regression"] = regression
            if regression < 0:
                entry["status"] = "discarded"
                logger.warning(
                    "gate DISCARDED round %r: pass_rate %.4f regresses vs "
                    "incumbent %.4f (%+.4f) — checkpoint NOT promoted (#82)",
                    label,
                    entry["pass_rate"],
                    incumbent,
                    regression,
                )
            else:
                entry["status"] = "promoted"
                logger.info(
                    "gate PROMOTED round %r: pass_rate %.4f >= incumbent %.4f",
                    label,
                    entry["pass_rate"],
                    incumbent,
                )
        if claims_registry is not None:
            # Register-before-use (C-1, #134): every gate decision gets a
            # claims-registry row with the run ids in the evidence chain.
            from .promotion import append_claims_row

            row = append_claims_row(
                claims_registry,
                claim=(
                    f"flywheel round {label!r} ({entry['model']}) gate "
                    f"{entry['status']}: pass_rate {entry['pass_rate']:.4f} "
                    f"({entry['passed']}/{entry['total']})"
                ),
                evidence=(
                    f"eval_run={entry['eval_run_id']} trainer_run={entry['run_id']} "
                    f"checkpoint={entry['checkpoint']} gate_mode="
                    f"{entry.get('gate_mode', 'regression')}"
                ),
                status="verified" if entry["status"] == "promoted" else "rejected",
            )
            entry["claims_row"] = row["id"]
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


def _consumed_export_shas(registry: dict[str, Any]) -> dict[str, str]:
    """Map consumed export sha -> consuming round label.

    A round consumed its export when it trained on it: entries with
    consumed=True. Historical entries (pre-#82, no field) count as
    consumed unless labelled baseline — every pre-#82 non-baseline round
    trained on its recorded export.
    """
    consumed: dict[str, str] = {}
    for r in registry.get("rounds", []):
        sha = r.get("export_sha256") or ""
        if not sha:
            continue
        is_consumed = r.get("consumed")
        if is_consumed is None:
            is_consumed = r.get("label") != "baseline" and not r.get("baseline")
        if is_consumed:
            consumed.setdefault(sha, r.get("label", ""))
    return consumed


def incumbent_pass_rate(registry: dict[str, Any]) -> float | None:
    """Best pass_rate among prior non-discarded rounds (None = no rounds).

    Historical entries without a status field count as promoted.
    """
    best: float | None = None
    for r in registry.get("rounds", []):
        if r.get("status") == "discarded":
            continue
        rate = float(r.get("pass_rate", 0.0))
        if best is None or rate > best:
            best = rate
    return best


def incumbent_entry(registry: dict[str, Any]) -> dict[str, Any] | None:
    """Full record of the best prior non-discarded round (#134).

    Same selection as incumbent_pass_rate but returns the entry so the
    Wilson gate can read its counts (passed/total).
    """
    best: dict[str, Any] | None = None
    best_rate: float | None = None
    for r in registry.get("rounds", []):
        if r.get("status") == "discarded":
            continue
        rate = float(r.get("pass_rate", 0.0))
        if best_rate is None or rate > best_rate:
            best = r
            best_rate = rate
    return best


def mark_round_status(cache_path: str, *, label: str, status: str) -> dict[str, Any]:
    """Set a recorded round's promotion status (promoted|discarded).

    Operational backfill for rounds recorded before the #82 gate existed,
    so the curve reflects which checkpoints were actually kept.
    """
    if status not in ("promoted", "discarded"):
        raise FlywheelError(f"status must be 'promoted' or 'discarded', got {status!r}")
    registry = load_registry(cache_path)
    matches = [r for r in registry["rounds"] if r.get("label") == label]
    if not matches:
        raise FlywheelError(f"no flywheel round labelled {label!r}")
    if len(matches) > 1:
        raise FlywheelError(
            f"{len(matches)} rounds labelled {label!r} — label must be unique to mark"
        )
    entry = matches[0]
    entry["status"] = status
    save_registry(cache_path, registry)
    logger.info(
        "marked flywheel round %r status=%s (pass_rate=%.4f)",
        label,
        status,
        float(entry.get("pass_rate", 0.0)),
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
                "status": r.get("status", ""),
                "reconsumed": bool(r.get("reconsumed")),
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
        f"  {'#':>2}  {'label':<18}  {'status':<10}  {'pass_rate':>9}  {'delta':>7}  "
        f"{'export':<12}  eval_run / checkpoint"
    )
    for row in rows:
        delta = "" if row["delta"] is None else f"{row['delta']:+.4f}"
        export = row["export_sha256"][:12] if row["export_sha256"] else "(baseline)"
        if row["reconsumed"]:
            export += "*"
        status = row["status"] or "-"
        lines.append(
            f"  {row['index']:>2}  {row['label']:<18}  {status:<10}  {row['pass_rate']:>9.4f}  {delta:>7}  "
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
        promoted = [r for r in rows if r["status"] != "discarded"]
        if len(promoted) >= 2:
            trend = promoted[-1]["pass_rate"] - promoted[0]["pass_rate"]
            direction = "UP" if trend > 0 else ("DOWN" if trend < 0 else "FLAT")
            lines.append("")
            lines.append(
                f"Promoted trend (first -> last promoted, discarded excluded): "
                f"{trend:+.4f} ({direction}). "
                "If not up, the experiment report must say why not (#76/#82 acceptance)."
            )
    discarded = [r["label"] for r in rows if r["status"] == "discarded"]
    if discarded:
        lines.append("")
        lines.append(
            f"Discarded rounds (gate rejected, checkpoints NOT promoted): "
            f"{', '.join(discarded)}"
        )
    if any(r["reconsumed"] for r in rows):
        lines.append("* = round re-consumed an already-consumed export (override).")
    return "\n".join(lines)
