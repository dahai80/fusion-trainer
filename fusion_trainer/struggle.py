"""Step-wise struggle judge + expert-guidance injection (issue #116, JudgeMix).

During co-rollout of a reference and a student model, expert guidance is
injected ONLY at steps where the student struggles — never uniformly. The
judge here is deterministic and LLM-free (rule 5): a step struggles when
it repeats the immediately preceding tool call (same normalized name +
arguments) or when the tool results it follows carry error content. Both
signals are the rc38 per-rollout struggle features narrowed to step
granularity, so the injection decision stays explainable from features
alone (an LLM judge may layer on top later, never replace the gate).
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

EXPERT_BLOCK_BUDGET = 1200
MAX_STEPS_IN_BLOCK = 3


def judge_struggle_steps(messages: list[Any]) -> list[dict[str, Any]]:
    """Judge every assistant step in one message sequence (#116).

    A step = one assistant turn (its tool_calls + content). A step is
    struggling when it deterministically shows waste or failure:
    - it repeats a call from the immediately preceding assistant step
      (same normalized name + sorted-arguments JSON), or
    - the tool results directly before it carry error content
      ("error" in the result text, case-insensitive).

    Returns one record per assistant step (in order):
    {step_index, n_calls, repeated_prev, error_results, struggle_score}
    where struggle_score = repeated_prev + 2 * error_results (same
    weighting as the rc38 per-rollout judge: an error counts double a
    redundant call). Non-struggling steps carry score 0 — callers filter.
    """
    steps: list[dict[str, Any]] = []
    prev_keys: list[str] = []
    for index, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "user":
            prev_keys = []
            continue
        if role == "tool":
            content = str(m.get("content", ""))
            # Error results attribute to the step that ISSUED the call
            # (the last assistant step) — that is the step that struggled
            # and the one the injection block tells to change course.
            if "error" in content.lower() and steps:
                steps[-1]["error_results"] += 1
                steps[-1]["struggle_score"] = (
                    steps[-1]["repeated_prev"] + 2 * steps[-1]["error_results"]
                )
            continue
        if role != "assistant":
            continue
        calls = [c for c in (m.get("tool_calls") or []) if isinstance(c, dict)]
        keys = []
        for c in calls:
            fn = c.get("function")
            if isinstance(fn, dict) and fn.get("name"):
                args = fn.get("arguments")
                if isinstance(args, str):
                    with contextlib.suppress(ValueError):
                        args = json.loads(args)
                if isinstance(args, dict):
                    args = json.dumps(args, sort_keys=True, ensure_ascii=False)
                keys.append(f"{fn['name']}|{args}")
        repeated = sum(1 for k in keys if k in prev_keys)
        step = {
            "step_index": index,
            "n_calls": len(calls),
            "repeated_prev": repeated,
            "error_results": 0,
            "struggle_score": repeated,
        }
        steps.append(step)
        prev_keys = keys
    logger.info(
        "struggle judge: %d assistant step(s), %d struggling",
        len(steps),
        sum(1 for s in steps if s["struggle_score"] > 0),
    )
    return steps


def struggling_recently(messages: list[Any], window: int = 3) -> list[dict[str, Any]]:
    """Struggling steps within the last `window` assistant steps (#116).

    The co-rollout driver calls this before each next-turn generation:
    non-empty return = inject expert guidance for the next turn. The
    window keeps a single old failure from poisoning every later turn.
    """
    steps = judge_struggle_steps(messages)
    recent = steps[-window:] if window > 0 else []
    return [s for s in recent if s["struggle_score"] > 0]


def render_expert_block(steps: list[dict[str, Any]]) -> str:
    """Deterministic expert-guidance injection block for struggling steps.

    The driver injects this before the student's next turn when
    `struggling_recently` is non-empty. Each line cites the step index and
    the concrete signal (repeated call / tool error) plus the corrective
    instruction — no LLM, stable across runs, budget-capped like the skill
    injection block (#123): never silently exceeded.
    """
    struggling = [s for s in steps if s["struggle_score"] > 0]
    if not struggling:
        return ""
    lines = [
        "Expert guidance — your recent steps struggled; adjust course before continuing:",
    ]
    for rendered, s in enumerate(struggling):
        if rendered >= MAX_STEPS_IN_BLOCK:
            lines.append(
                f"(+{len(struggling) - MAX_STEPS_IN_BLOCK} more struggling steps omitted)"
            )
            break
        reasons = []
        if s["repeated_prev"]:
            reasons.append(
                f"repeated a tool call from the previous step ({s['repeated_prev']}x) — "
                "change the arguments or use a different tool"
            )
        if s["error_results"]:
            reasons.append(
                f"{s['error_results']} tool result(s) returned an error — "
                "fix the cause before retrying the same call"
            )
        line = f"- Step {s['step_index']}: " + "; ".join(reasons)
        if len("\n".join(lines)) + len(line) > EXPERT_BLOCK_BUDGET:
            lines.append("(further guidance omitted: character budget reached)")
            break
        lines.append(line)
    return "\n".join(lines)
