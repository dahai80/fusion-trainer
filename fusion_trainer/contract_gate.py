"""Office-pilot contract gate (issue #133, trainer side of bench#68).

Consumes two office-bench summary.json files (pre/baseline and post/
candidate, `build_summary` shape: arms.<ARM>.pass_rate + per-contract
`contracts: {cid: {passed, total, rate}}`) and decides whether the
candidate checkpoint clears the #133 hard gates:

  1. overall office-bench pass rate >= threshold (default 0.80);
  2. must-not-flip contract ids never regress — any per-contract rate
     drop on a listed id rejects the checkpoint regardless of task gain;
  3. token-per-task cost gate: post <= pre * (1 - reduction) (default
     -30% vs baseline);
  4. hacking_monitor clean run.

Deterministic and LLM-free (rule 5). Gates 3/4 need inputs the bench
summary does not carry (row-level token totals, hacking monitor verdict);
when absent they are reported as "unmeasured" and the decision FAILS
loudly — an unmeasured hard gate never silently passes (#133 "hard,
pre/post").
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MUST_NOT_FLIP = (
    "OB-NOOP-NO-MUTATION",
    "OB-ROBUST-SURFACE-PERTURBATION",
    "OB-ABSTAIN-NO-EVIDENCE",
    "OB-FIDELITY-ANCHOR-PRESERVE",
    "OB-INFER-EXEMPLAR-ATTRIBUTE",
)


class ContractGateError(Exception):
    pass


def load_gate_summary(path: Path) -> dict[str, Any]:
    """Load an office-bench summary and extract the single arm record."""
    data = json.loads(path.read_text(encoding="utf-8"))
    arms = data.get("arms")
    if not isinstance(arms, dict) or not arms:
        raise ContractGateError(f"summary {path} carries no arms record")
    if len(arms) > 1:
        raise ContractGateError(
            f"summary {path} carries {len(arms)} arms — the contract gate "
            "consumes single-arm office-bench summaries"
        )
    arm = next(iter(arms.values()))
    return arm


def gate_contract_probes(
    pre: dict[str, Any],
    post: dict[str, Any],
    *,
    overall_threshold: float = 0.80,
    must_not_flip: tuple[str, ...] = DEFAULT_MUST_NOT_FLIP,
    token_per_task_pre: float | None = None,
    token_per_task_post: float | None = None,
    token_reduction: float = 0.30,
    hacking_clean: bool | None = None,
) -> dict[str, Any]:
    """Decide the #133 pre/post hard gates from bench summaries.

    pre/post: single-arm records (load_gate_summary). Returns a decision
    dict {promoted, checks[], unmeasured[]} — promoted True only when every
    hard gate is both evaluated and passed; unmeasured gates fail loudly.
    """
    checks: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    post_rate = float(post.get("pass_rate") or 0.0)
    ok = post_rate >= overall_threshold
    checks.append(
        {
            "gate": "overall",
            "passed": ok,
            "detail": f"post pass_rate {post_rate:.4f} vs threshold {overall_threshold:.2f}",
        }
    )
    pre_contracts = pre.get("contracts") or {}
    post_contracts = post.get("contracts") or {}
    regressions = []
    for cid in must_not_flip:
        pre_rec = pre_contracts.get(cid) or {}
        post_rec = post_contracts.get(cid) or {}
        pre_rate = float(pre_rec.get("rate") or 0.0)
        post_rate_c = float(post_rec.get("rate") or 0.0)
        if post_rate_c < pre_rate:
            regressions.append(f"{cid} {pre_rate:.4f}->{post_rate_c:.4f}")
    ok = not regressions
    checks.append(
        {
            "gate": "must-not-flip",
            "passed": ok,
            "detail": "no regressions on "
            + f"{len(must_not_flip)} contract ids"
            if ok
            else "regressed: " + "; ".join(regressions),
        }
    )
    if token_per_task_pre is None or token_per_task_post is None:
        unmeasured.append("token-per-task")
        checks.append({"gate": "token-cost", "passed": False, "detail": "unmeasured"})
        logger.warning(
            "contract gate: token-per-task inputs missing — cost gate "
            "UNMEASURED, decision fails loud (#133 hard gate)"
        )
    else:
        budget = token_per_task_pre * (1.0 - token_reduction)
        ok = token_per_task_post <= budget
        checks.append(
            {
                "gate": "token-cost",
                "passed": ok,
                "detail": f"post {token_per_task_post:.1f} vs budget {budget:.1f} "
                f"(pre {token_per_task_pre:.1f}, -{token_reduction:.0%})",
            }
        )
    if hacking_clean is None:
        unmeasured.append("hacking-monitor")
        checks.append({"gate": "hacking-monitor", "passed": False, "detail": "unmeasured"})
        logger.warning(
            "contract gate: hacking monitor verdict missing — gate "
            "UNMEASURED, decision fails loud (#133 hard gate)"
        )
    else:
        checks.append(
            {
                "gate": "hacking-monitor",
                "passed": bool(hacking_clean),
                "detail": "clean run" if hacking_clean else "detections present",
            }
        )
    promoted = all(c["passed"] for c in checks)
    decision = {
        "promoted": promoted,
        "checks": checks,
        "unmeasured": unmeasured,
        "reason": (
            "all hard gates passed"
            if promoted
            else (
                "unmeasured hard gates: " + ", ".join(unmeasured)
                if unmeasured
                else "failed gates: "
                + ", ".join(c["gate"] for c in checks if not c["passed"])
            )
        ),
    }
    if promoted:
        logger.info("contract gate PROMOTED: %s", decision["reason"])
    else:
        logger.warning("contract gate REJECTED: %s", decision["reason"])
    return decision
