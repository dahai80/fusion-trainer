"""BNUP DPO negative sample factory (issue #57).

Generates syntactically distinct negative samples for DPO preference
training. Each generator produces a response that violates the
<thinking> + <dsl> format contract in a different way:

  make_broken_json       — corrupt JSON structure (truncated / invalid tokens)
  make_direct_answer     — bypass <thinking>, dump formula + answer directly
  make_wrong_visual_type — swap meta.visual_type to an invalid primitive
  make_missing_tag       — strip <thinking> or <dsl> tag from a valid response
  make_format_drift      — rename required top-level DSL fields

Every generator output is designed to FAIL fusion-core validate_dsl_str /
validate_dsl — proving they are true negatives, not false ones.
"""

from __future__ import annotations

import copy
import json
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

VALID_VISUAL_TYPES = (
    "array_grid",
    "tape_diagram",
    "geometry_2d",
    "isometric_3d",
    "track_timeline",
    "bucket_divider",
    "data_chart",
    "flow_card",
)

INVALID_VISUAL_TYPES = (
    "hologram_4d",
    "vr_scene",
    "neural_mesh",
    "quantum_plot",
)

_THINKING_OPEN = "<thinking>"
_THINKING_CLOSE = "</thinking>"
_DSL_OPEN = "<dsl>"
_DSL_CLOSE = "</dsl>"


def make_broken_json(dsl_str: str) -> str:
    """Corrupt a DSL JSON string by truncating closing brackets.

    Removes trailing closing brackets/braces so json.loads fails.
    Returns the broken string (not wrapped in <dsl> tags — the breakage
    is structural, not just missing wrapper).
    """
    stripped = dsl_str.rstrip()
    if not stripped:
        return '{"meta":'
    remove_count = min(len(stripped), random.randint(3, 8))
    broken = stripped[:-remove_count]
    logger.debug("make_broken_json: removed %d trailing chars", remove_count)
    return broken


def make_direct_answer(question: str, formula: str, answer: str) -> str:
    """Build a non-Socratic direct-answer response.

    Dumps the formula and final answer with NO <thinking> chain and NO
    <dsl> block — the anti-pattern DPO should punish (answer-dumping
    instead of guided Socratic derivation).
    """
    return f"直接计算得：{answer}。\n公式：{formula}"


def make_wrong_visual_type(dsl_dict: dict[str, Any], wrong_type: str = "") -> str:
    """Swap meta.visual_type to an inappropriate/invalid primitive.

    Produces valid JSON but with a visual_type outside the allowed enum,
    so validate_dsl fails on the schema constraint (not JSON syntax).
    """
    mutated = copy.deepcopy(dsl_dict)
    meta = mutated.get("meta", {})
    wrong = wrong_type or random.choice(INVALID_VISUAL_TYPES)
    original = meta.get("visual_type", "")
    meta["visual_type"] = wrong
    mutated["meta"] = meta
    logger.debug(
        "make_wrong_visual_type: %s -> %s", original or "(none)", wrong
    )
    return json.dumps(mutated, ensure_ascii=False)


def make_missing_tag(response: str) -> str:
    """Strip <thinking> or <dsl> tag from a valid response.

    Randomly removes one of the two required tag blocks, leaving the
    other intact — tests partial-format drift (e.g., thinking present
    but DSL missing, or vice versa).
    """
    if _DSL_OPEN in response and _DSL_CLOSE in response:
        start = response.index(_DSL_OPEN)
        end = response.index(_DSL_CLOSE) + len(_DSL_CLOSE)
        result = response[:start].rstrip() + response[end:]
        logger.debug("make_missing_tag: stripped <dsl> block")
        return result
    if _THINKING_OPEN in response and _THINKING_CLOSE in response:
        start = response.index(_THINKING_OPEN)
        end = response.index(_THINKING_CLOSE) + len(_THINKING_CLOSE)
        result = response[:start].rstrip() + response[end:]
        logger.debug("make_missing_tag: stripped <thinking> block")
        return result
    logger.warning("make_missing_tag: no tag pair found, returning as-is")
    return response


_FIELD_RENAMES = {
    "meta": "metadata",
    "parameters": "params",
    "entities": "objects",
    "pipeline": "steps",
}


def make_format_drift(dsl_dict: dict[str, Any]) -> str:
    """Rename required top-level DSL fields to wrong names.

    Produces valid JSON but with renamed keys, so schema validation
    fails on missing required fields (meta/parameters/entities/pipeline).
    """
    mutated = copy.deepcopy(dsl_dict)
    renamed = {}
    for key, val in mutated.items():
        new_key = _FIELD_RENAMES.get(key, key)
        renamed[new_key] = val
    logger.debug(
        "make_format_drift: renamed %d keys", sum(1 for k in mutated if k in _FIELD_RENAMES)
    )
    return json.dumps(renamed, ensure_ascii=False)


def generate_negatives(
    chosen_response: str,
    dsl_dict: dict[str, Any] | None = None,
    question: str = "",
    formula: str = "",
    answer: str = "",
    neg_ratio: int = 3,
) -> list[str]:
    """Generate up to neg_ratio distinct negative responses for one chosen.

    Picks from the 5 generators without repetition (caps at 5). If
    dsl_dict is None, tries to extract it from the chosen_response's
    <dsl> block; if extraction fails, skips DSL-dependent generators.
    """
    if dsl_dict is None:
        dsl_dict = _extract_dsl_dict(chosen_response)

    candidates: list[str] = []
    dsl_str = json.dumps(dsl_dict, ensure_ascii=False) if dsl_dict else ""

    if dsl_dict:
        candidates.append(make_broken_json(dsl_str))
        candidates.append(make_wrong_visual_type(dsl_dict))
        candidates.append(make_format_drift(dsl_dict))
    candidates.append(make_direct_answer(question, formula, answer))
    candidates.append(make_missing_tag(chosen_response))

    random.shuffle(candidates)
    count = min(neg_ratio, len(candidates))
    selected = candidates[:count]
    logger.info(
        "generate_negatives: %d negatives from %d candidates (ratio=%d)",
        len(selected),
        len(candidates),
        neg_ratio,
    )
    return selected


def _extract_dsl_dict(response: str) -> dict[str, Any] | None:
    """Extract and parse the <dsl> JSON block from a response string."""
    if _DSL_OPEN not in response or _DSL_CLOSE not in response:
        return None
    start = response.index(_DSL_OPEN) + len(_DSL_OPEN)
    end = response.index(_DSL_CLOSE)
    raw = response[start:end].strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        logger.debug("_extract_dsl_dict: <dsl> block not valid JSON")
    return None
