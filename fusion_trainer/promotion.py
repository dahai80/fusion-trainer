"""Eval-gated promotion with Wilson CI non-overlap (issue #134, phase-2).

The #82 gate compared point pass rates — a 1-task swing (3.3pp at n=30)
could promote or discard a round inside the noise band. Phase-2 makes
promotion statistical: a candidate adapter promotes only when its Wilson
95% CI sits strictly above the incumbent's (non-overlap) and above the
E-2 noise-floor band when one is registered. Deterministic and LLM-free
(rule 5): the decision is computable from counts alone, and every
promotion/rejection appends a claims-registry row (C-1 discipline) so the
claim "this adapter is better" is registered before use.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WILSON_Z = 1.959963984540054
CLAIMS_ROW_RE = re.compile(r"^\|\s*CL-(\d+)\s*\|")


def wilson_ci(passes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (deterministic).

    Returns (lo, hi) in [0, 1]. n <= 0 returns (0.0, 1.0) — maximum
    uncertainty, never a fake tight band.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = passes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def ci_non_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True when interval a sits strictly above interval b (no shared band)."""
    return a[0] > b[1]


def _counts(rec: dict[str, Any]) -> tuple[int, int]:
    return int(rec.get("passed") or 0), int(rec.get("total") or 0)


def gate_promotion(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    floor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide promotion from counts via Wilson CI non-overlap (#134).

    candidate/incumbent: records carrying at least passed/total (a point
    pass_rate alone degrades to the #82 regression check with a loud log —
    never a silent downgrade). floor: optional E-2 noise-floor record
    {passed, total} — the candidate must ALSO clear the floor band's upper
    bound, so a "better than incumbent" adapter still cannot promote below
    the measured capability floor. Returns the decision dict consumed by
    the flywheel registry entry.
    """
    cand_p, cand_n = _counts(candidate)
    cand_ci = wilson_ci(cand_p, cand_n)
    decision: dict[str, Any] = {
        "gate_mode": "wilson",
        "candidate_ci": [round(cand_ci[0], 4), round(cand_ci[1], 4)],
        "promoted": False,
        "reason": "",
    }
    if incumbent is None:
        decision["promoted"] = True
        decision["reason"] = "no incumbent — first round promotes by default"
        logger.info("gate: no incumbent — first round promotes by default")
        return decision
    inc_p, inc_n = _counts(incumbent)
    if inc_n <= 0 or cand_n <= 0:
        cand_rate = float(candidate.get("pass_rate") or 0.0)
        inc_rate = float(incumbent.get("pass_rate") or 0.0)
        decision["gate_mode"] = "regression-fallback"
        decision["promoted"] = cand_rate >= inc_rate
        decision["reason"] = (
            f"counts unavailable (candidate n={cand_n}, incumbent n={inc_n}) "
            f"— degraded to #82 point regression check {cand_rate:.4f} vs "
            f"{inc_rate:.4f}"
        )
        logger.warning(
            "gate: Wilson CI unavailable (n=%d/%d) — DEGRADED to point "
            "regression check %.4f vs %.4f (register repeat data for the "
            "statistical gate, #134)",
            cand_n,
            inc_n,
            cand_rate,
            inc_rate,
        )
        return decision
    inc_ci = wilson_ci(inc_p, inc_n)
    decision["incumbent_ci"] = [round(inc_ci[0], 4), round(inc_ci[1], 4)]
    if floor is not None and _counts(floor)[1] > 0:
        floor_ci = wilson_ci(*_counts(floor))
        decision["floor_ci"] = [round(floor_ci[0], 4), round(floor_ci[1], 4)]
        if not ci_non_overlap(cand_ci, floor_ci):
            decision["reason"] = (
                f"candidate CI {decision['candidate_ci']} does not clear the "
                f"E-2 floor band {decision['floor_ci']}"
            )
            logger.warning(
                "gate DISCARDED: candidate CI %s overlaps the E-2 floor band "
                "%s — adapter at the capability floor (#134)",
                decision["candidate_ci"],
                decision["floor_ci"],
            )
            return decision
    if ci_non_overlap(cand_ci, inc_ci):
        decision["promoted"] = True
        decision["reason"] = (
            f"candidate CI {decision['candidate_ci']} strictly above "
            f"incumbent CI {decision['incumbent_ci']}"
        )
        logger.info(
            "gate PROMOTED: candidate CI %s > incumbent CI %s (#134)",
            decision["candidate_ci"],
            decision["incumbent_ci"],
        )
        return decision
    decision["reason"] = (
        f"candidate CI {decision['candidate_ci']} overlaps incumbent CI "
        f"{decision['incumbent_ci']} — statistically indistinguishable"
    )
    logger.warning(
        "gate DISCARDED: candidate CI %s overlaps incumbent CI %s — "
        "no statistically separable improvement (#134)",
        decision["candidate_ci"],
        decision["incumbent_ci"],
    )
    return decision


def next_claims_id(registry_path: Path) -> str:
    """Next sequential CL-0NN id from the claims registry (C-1 discipline)."""
    max_id = 0
    if registry_path.is_file():
        for line in registry_path.read_text(encoding="utf-8").splitlines():
            m = CLAIMS_ROW_RE.match(line)
            if m:
                max_id = max(max_id, int(m.group(1)))
    return f"CL-{max_id + 1:03d}"


def append_claims_row(
    registry_path: Path,
    *,
    claim: str,
    evidence: str,
    status: str = "verified",
    date: str | None = None,
) -> dict[str, str]:
    """Append one promotion/rejection row to the claims registry (#134).

    Register-before-use (C-1): every promotion or rejection decision gets
    a row with the run ids in the evidence chain. Appends a single table
    line — the registry keeps refuted/rejected rows forever (never
    deleted). Raises FileNotFoundError when the registry does not exist:
    an absent registry is a setup error, not silently skippable.
    """
    if not registry_path.is_file():
        raise FileNotFoundError(f"claims registry not found: {registry_path}")
    row_id = next_claims_id(registry_path)
    today = date or datetime.now(UTC).date().isoformat()
    row = f"| {row_id} | {today} | {claim} | {evidence} | {status} | {today} |"
    text = registry_path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    registry_path.write_text(text + row + "\n", encoding="utf-8")
    logger.info("claims registry: appended %s (%s)", row_id, status)
    return {"id": row_id, "row": row}


def load_floor_band(path: Path) -> dict[str, Any]:
    """Load an E-2 noise-floor band record from JSON ({passed, total, ...}).

    The floor docs live as markdown in fusion-eval; the gate consumes the
    machine-readable JSON export of the majority-vote baseline (repeat-3).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("passed", "total"):
        if key not in data:
            raise ValueError(f"floor band {path} missing {key!r}")
    return data
