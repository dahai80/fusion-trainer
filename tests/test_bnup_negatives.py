"""Tests for BNUP DPO negative sample factory (issue #57).

Each generator must produce output that FAILS DSL validation — proving
they are true negatives, not false ones. Every generator test class
includes a ``test_fails_dsl_validation`` that calls the real
``validate_dsl_str`` from fusion-core to verify the negative actually
fails schema validation (not just structural heuristics).
"""

from __future__ import annotations

import json
import re

import pytest
from fusion_core.dsl_schema import validate_dsl_str

from fusion_trainer.dataset.bnup_negatives import (
    generate_negatives,
    make_broken_json,
    make_direct_answer,
    make_format_drift,
    make_missing_tag,
    make_wrong_visual_type,
)


def _extract_dsl_content(response: str) -> str:
    if "<dsl>" in response and "</dsl>" in response:
        start = response.index("<dsl>") + len("<dsl>")
        end = response.rindex("</dsl>")
        return response[start:end].strip()
    return response.strip()


def _assert_negative_fails(response: str) -> None:
    dsl_content = _extract_dsl_content(response)
    ok, msg = validate_dsl_str(dsl_content)
    assert not ok, (
        f"negative should FAIL validate_dsl_str but passed.\n"
        f"response: {response[:200]}\n"
        f"extracted: {dsl_content[:200]}"
    )
    assert msg, (
        f"validate_dsl_str returned empty error for failing negative:\n{response[:200]}"
    )


_VALID_DSL = {
    "meta": {
        "id": "demo_01",
        "grade": 6,
        "topic": "圆柱与圆锥",
        "visual_type": "geometry_2d",
        "fallback_type": "tape_diagram",
    },
    "parameters": {
        "base_area": {"label": "底面积", "value": 12, "unit": "cm²"},
        "height": {"label": "高", "value": 5, "unit": "cm"},
    },
    "entities": [
        {"id": "cylinder", "type": "cylinder_3d", "label": "原圆柱体", "properties": {}}
    ],
    "pipeline": [
        {
            "step": 1,
            "title": "计算总体积",
            "formula": "V=S*h",
            "eval": "params.base_area * params.height",
            "result_unit": "cm³",
            "visual_state": {
                "progress_range": [0, 1],
                "active_entities": ["cylinder"],
                "action": "highlight",
            },
        }
    ],
}

_VALID_RESPONSE = (
    "<thinking>\n1. 识别知识点\n2. 导出公式\n</thinking>\n"
    f"<dsl>\n{json.dumps(_VALID_DSL, ensure_ascii=False)}\n</dsl>"
)


class TestMakeBrokenJson:
    def test_produces_invalid_json(self):
        dsl_str = json.dumps(_VALID_DSL, ensure_ascii=False)
        broken = make_broken_json(dsl_str)
        with pytest.raises(json.JSONDecodeError):
            json.loads(broken)

    def test_empty_input(self):
        broken = make_broken_json("")
        assert broken == '{"meta":'

    def test_fails_dsl_validation(self):
        dsl_str = json.dumps(_VALID_DSL, ensure_ascii=False)
        broken = make_broken_json(dsl_str)
        _assert_negative_fails(broken)


class TestMakeDirectAnswer:
    def test_no_thinking_tag(self):
        result = make_direct_answer("题目", "V=S*h*2/3", "40")
        assert "<thinking>" not in result
        assert "<dsl>" not in result
        assert "40" in result

    def test_no_dsl_block(self):
        result = make_direct_answer("题目", "V=S*h", "60")
        assert "<dsl>" not in result

    def test_fails_dsl_validation(self):
        result = make_direct_answer("题目", "V=S*h", "60")
        _assert_negative_fails(result)


class TestMakeWrongVisualType:
    def test_invalid_visual_type(self):
        result = make_wrong_visual_type(_VALID_DSL, wrong_type="hologram_4d")
        parsed = json.loads(result)
        assert parsed["meta"]["visual_type"] == "hologram_4d"
        assert parsed["meta"]["visual_type"] not in (
            "array_grid",
            "tape_diagram",
            "geometry_2d",
            "isometric_3d",
            "track_timeline",
            "bucket_divider",
            "data_chart",
            "flow_card",
        )

    def test_valid_json_but_schema_fail(self):
        result = make_wrong_visual_type(_VALID_DSL, wrong_type="vr_scene")
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert parsed["meta"]["visual_type"] == "vr_scene"

    def test_fails_dsl_validation(self):
        result = make_wrong_visual_type(_VALID_DSL, wrong_type="hologram_4d")
        _assert_negative_fails(result)


class TestMakeMissingTag:
    def test_strips_dsl_block(self):
        result = make_missing_tag(_VALID_RESPONSE)
        assert "<dsl>" not in result
        assert "<thinking>" in result

    def test_strips_thinking_when_no_dsl(self):
        response = "<thinking>\n思考\n</thinking>\n答案"
        result = make_missing_tag(response)
        assert "<thinking>" not in result

    def test_no_tags_returns_as_is(self):
        response = "plain text response"
        result = make_missing_tag(response)
        assert result == response

    def test_fails_dsl_validation(self):
        result = make_missing_tag(_VALID_RESPONSE)
        _assert_negative_fails(result)


class TestMakeFormatDrift:
    def test_renames_required_fields(self):
        result = make_format_drift(_VALID_DSL)
        parsed = json.loads(result)
        assert "meta" not in parsed
        assert "metadata" in parsed
        assert "pipeline" not in parsed
        assert "steps" in parsed

    def test_loses_required_keys(self):
        result = make_format_drift(_VALID_DSL)
        parsed = json.loads(result)
        required = {"meta", "parameters", "entities", "pipeline"}
        assert not required.issubset(set(parsed.keys()))

    def test_fails_dsl_validation(self):
        result = make_format_drift(_VALID_DSL)
        _assert_negative_fails(result)


class TestGenerateNegatives:
    def test_default_ratio_3(self):
        negatives = generate_negatives(
            chosen_response=_VALID_RESPONSE,
            dsl_dict=_VALID_DSL,
            question="题目",
            formula="V=S*h",
            answer="60",
        )
        assert len(negatives) == 3
        assert len(set(negatives)) == 3

    def test_max_5_distinct(self):
        negatives = generate_negatives(
            chosen_response=_VALID_RESPONSE,
            dsl_dict=_VALID_DSL,
            question="题目",
            formula="V=S*h",
            answer="60",
            neg_ratio=10,
        )
        assert len(negatives) <= 5

    def test_works_without_dsl_dict(self):
        negatives = generate_negatives(
            chosen_response=_VALID_RESPONSE,
            dsl_dict=None,
            question="题目",
            answer="60",
        )
        assert len(negatives) >= 1

    def test_all_negatives_differ_from_chosen(self):
        negatives = generate_negatives(
            chosen_response=_VALID_RESPONSE,
            dsl_dict=_VALID_DSL,
            question="题目",
            answer="60",
            neg_ratio=5,
        )
        for neg in negatives:
            assert neg != _VALID_RESPONSE

    def test_all_negatives_fail_validation_or_eval(self):
        """静态负例必须过不了 validate_dsl_str; param_drift 是语义负例 —
        JSON/schema 合法, 破绽只在 eval 求值时暴露, 用求值器断言。
        (sft-dpo-1013 §2.4: E1 漂移的破绽正是静态校验查不到的那类)

        求值器复刻生产 BNUPEvaluator.evaluate_eval_expression 的核心语义
        (fusion-k12-teacher eval_harness.py: params.xxx → params_val['xxx'],
        eval restricted globals) — 不直接 import 跨项目包 (CI 环境不装
        fusion-k12-teacher, 且 trainer 不依赖它), 但断言语义一致:
        漂移键在 params_val 下标中缺失 (KeyError) 即求值失败。"""

        def _eval_pipeline(pipeline, parameters):
            params_val = {
                k: v.get("value", 0)
                for k, v in parameters.items()
                if isinstance(v, dict)
            }
            for step_item in pipeline:
                expr = str(step_item.get("eval", ""))
                if not expr:
                    continue
                py_expr = re.sub(r"params\.(\w+)", r"params_val['\1']", expr)
                eval(py_expr, {"__builtins__": None}, {"params_val": params_val})

        negatives = generate_negatives(
            chosen_response=_VALID_RESPONSE,
            dsl_dict=_VALID_DSL,
            question="题目",
            formula="V=S*h",
            answer="60",
            neg_ratio=5,
        )
        assert len(negatives) >= 1
        for neg in negatives:
            dsl_content = _extract_dsl_content(neg)
            ok, _msg = validate_dsl_str(dsl_content)
            if ok:
                # 语义负例 (param_drift): 必须在求值器下失败
                dsl = json.loads(dsl_content)
                with pytest.raises((KeyError, NameError)):
                    _eval_pipeline(dsl["pipeline"], dsl["parameters"])
            # 静态负例由 _assert_negative_fails 覆盖 (其余测试)

    def test_valid_dsl_passes_validation(self):
        ok, msg = validate_dsl_str(json.dumps(_VALID_DSL, ensure_ascii=False))
        assert ok, f"baseline valid DSL should PASS but failed: {msg}"
