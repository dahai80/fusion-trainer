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
import re
from typing import Any

logger = logging.getLogger(__name__)

# params.<name> reference extractor (mirrors fusion-k12 eval_harness._PARAM_SUB)
PARAM_NAME_RE = re.compile(r"params\.(\w+)")

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
    logger.debug("make_wrong_visual_type: %s -> %s", original or "(none)", wrong)
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
        "make_format_drift: renamed %d keys",
        sum(1 for k in mutated if k in _FIELD_RENAMES),
    )
    return json.dumps(renamed, ensure_ascii=False)


# Real E1 failure vocabulary (e2e_28final.json per_sample: "eval exec: '每排'"
# "eval exec: 'cols'" "eval exec: 'row_count'" "eval exec: 'T_bridge'"
# "eval exec: 'L_wire'" "eval exec: 'v_cone'" "eval exec: '31'" — the model
# defined parameters but referenced drifted names in pipeline.eval). Each
# entry maps a live eval reference to the drift form observed in production.
_DRIFT_SUFFIXES = ("s", "_count", "_total", "_per_row", "_each", "_val")
_DRIFT_TRANSLATE = {
    "rows": "每排",
    "cols": "列数",
    "row_count": "排数",
    "col_count": "列数",
    "count": "数量",
    "total": "总数",
    "length": "长度",
    "speed": "速度",
    "time": "时间",
    "area": "面积",
    "width": "宽度",
    "height": "高",
}
_DRIFT_CASE = str.maketrans(
    "abcdefghijklmnopqrstuvwxyz_", "ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
)


def _drift_variants(name: str) -> list[str]:
    """Return plausible drift spellings for a parameter name (real E1 forms)."""
    out: list[str] = []
    low = name.lower()
    # 1. 中文同义名 (最高频 E1 形态: '每排' 代替 rows)
    zh = _DRIFT_TRANSLATE.get(low)
    if zh:
        out.append(zh)
    # 2. 缩写 (cols 代替 columns / cnt 代替 count)
    if low.endswith("count"):
        out.append(low[:-5] + "s" if not low.endswith("s") else low[:-1])
        out.append("cnt_" + low.replace("_count", ""))
    elif low.endswith("s") and len(low) > 3:
        out.append(low[:-1])
    # 3. 大小写漂移 (T_bridge 代替 t_bridge — 真实样本原样)
    cased = name.translate(_DRIFT_CASE)
    if cased != name:
        out.append(cased)
    # 4. 后缀漂移 (_total/_val/_each 附加)
    for suf in _DRIFT_SUFFIXES:
        cand = name + suf
        if cand != name:
            out.append(cand)
    # 去重保序
    seen: set[str] = set()
    uniq = []
    for cand in out:
        if cand not in seen and cand != name:
            seen.add(cand)
            uniq.append(cand)
    return uniq


def make_param_drift(dsl_dict: dict[str, Any], rng: random.Random | None = None) -> str:
    """Rewrite one pipeline.eval reference to an UNDEFINED parameter name.

    The #1 real-model failure mode (10/18 fixable errors in e2e_28final):
    parameters defines L_bridge, eval references 每排/cols/T_bridge. The
    drift name is chosen from the observed-failure vocabulary; JSON stays
    valid and schema passes (parameters schema is unconstrained) — the
    break only surfaces at eval execution (NameError), exactly mirroring
    production. Also may inline a bare number literal (real '31' failure).
    """
    rng = rng or random
    mutated = copy.deepcopy(dsl_dict)
    params = mutated.get("parameters", {})
    pipe = mutated.get("pipeline", [])
    # Collect (step_index, ref) pairs where ref is a defined parameter
    spots: list[tuple[int, str]] = []
    for i, step in enumerate(pipe):
        for ref in PARAM_NAME_RE.findall(str(step.get("eval", ""))):
            if ref in params:
                spots.append((i, ref))
    if not spots:
        return json.dumps(mutated, ensure_ascii=False)
    idx, ref = rng.choice(spots)
    variants = _drift_variants(ref)
    drift = rng.choice(variants) if variants else ref + "x"
    step = pipe[idx]
    expr = str(step.get("eval", ""))
    # Replace exactly one occurrence of params.<ref> with the drift name
    step["eval"] = expr.replace(f"params.{ref}", f"params.{drift}", 1)
    logger.debug("make_param_drift: %s -> %s in step %d", ref, drift, idx + 1)
    return json.dumps(mutated, ensure_ascii=False)


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
        drift_negs = [
            make_param_drift(dsl_dict, random),  # E1 头号失败模式 (#28)
            make_param_drift(dsl_dict, random),
        ]  # 双份提高密度 (去重后仍常为 2)
        others = [
            make_broken_json(dsl_str),
            make_wrong_visual_type(dsl_dict),
            make_format_drift(dsl_dict),
        ]
    else:
        drift_negs, others = [], []
    others += [
        make_direct_answer(question, formula, answer),
        make_missing_tag(chosen_response),
    ]

    # 选择: 负例总数 ≤5 (文档契约); param_drift 优先占 2 席 —
    # neg_ratio=3 时 drift 密度最高, 直击 E1; 其余席位从静态负例池随机填充。
    # 去重保序: 两次 make_param_drift 可能撞出同一负例 (同一 spot + 同一
    # 变体), 撞了就用静态负例补位 — "distinct" 是文档契约, 不靠运气。
    count = max(0, min(neg_ratio, 5))
    random.shuffle(others)
    seen: set[str] = set()
    selected: list[str] = []
    for neg in drift_negs[: min(len(drift_negs), 2, count)] + others:
        if len(selected) >= count:
            break
        if neg not in seen:
            seen.add(neg)
            selected.append(neg)
    random.shuffle(selected)
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
