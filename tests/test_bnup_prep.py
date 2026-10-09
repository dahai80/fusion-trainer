"""Tests for BNUP data preparation (issue #56).

Tests SFT data prep (validate_dsl_json, build_sft_sample, process_raw_sft)
and DPO pair generation (create_dpo_pairs, normalize_for_trainer).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fusion_trainer.dataset.bnup_dpo_prep import (
    create_dpo_pairs,
    normalize_for_trainer,
)
from fusion_trainer.dataset.bnup_prep import (
    DEFAULT_SYSTEM_PROMPT,
    build_sft_sample,
    process_raw_sft,
    validate_dsl_json,
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

_INVALID_DSL = {"meta": {"id": "bad"}, "parameters": {}}


class TestValidateDslJson:
    def test_valid_dsl(self):
        dsl_str = json.dumps(_VALID_DSL, ensure_ascii=False)
        assert validate_dsl_json(dsl_str) is True

    def test_invalid_dsl_missing_fields(self):
        dsl_str = json.dumps(_INVALID_DSL, ensure_ascii=False)
        assert validate_dsl_json(dsl_str) is False

    def test_invalid_json(self):
        assert validate_dsl_json("{broken") is False

    def test_empty_string(self):
        assert validate_dsl_json("") is False


class TestBuildSftSample:
    def test_has_thinking_and_dsl_tags(self):
        sample = build_sft_sample(
            system_prompt="system",
            question="question",
            thinking_content="思考过程",
            dsl_dict=_VALID_DSL,
        )
        assert "<thinking>" in sample["output"]
        assert "</thinking>" in sample["output"]
        assert "<dsl>" in sample["output"]
        assert "</dsl>" in sample["output"]

    def test_instruction_and_system(self):
        sample = build_sft_sample(
            system_prompt="sys",
            question="q1",
            thinking_content="think",
            dsl_dict=_VALID_DSL,
        )
        assert sample["instruction"] == "q1"
        assert sample["system"] == "sys"
        assert sample["input"] == ""

    def test_dsl_is_valid_json_in_output(self):
        sample = build_sft_sample(
            system_prompt="sys",
            question="q",
            thinking_content="t",
            dsl_dict=_VALID_DSL,
        )
        dsl_start = sample["output"].index("<dsl>") + len("<dsl>")
        dsl_end = sample["output"].index("</dsl>")
        dsl_str = sample["output"][dsl_start:dsl_end].strip()
        parsed = json.loads(dsl_str)
        assert parsed == _VALID_DSL


class TestProcessRawSft:
    def test_valid_input(self, tmp_path):
        raw = [
            {"question": "q1", "thinking": "t1", "dsl": _VALID_DSL},
            {"question": "q2", "thinking": "t2", "dsl": _VALID_DSL},
        ]
        input_file = tmp_path / "raw.json"
        input_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        output_file = str(tmp_path / "sft.jsonl")

        result = process_raw_sft(str(input_file), output_file)
        assert result["valid"] == 2
        assert result["skipped"] == 0
        assert result["total"] == 2

        lines = Path(output_file).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        sample = json.loads(lines[0])
        assert "<thinking>" in sample["output"]

    def test_skips_invalid(self, tmp_path):
        raw = [
            {"question": "q1", "thinking": "t1", "dsl": _VALID_DSL},
            {"question": "", "thinking": "t2", "dsl": _VALID_DSL},
            {"question": "q3", "thinking": "t3", "dsl": _INVALID_DSL},
            {"question": "q4", "thinking": "", "dsl": _VALID_DSL},
        ]
        input_file = tmp_path / "raw.json"
        input_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        output_file = str(tmp_path / "sft.jsonl")

        result = process_raw_sft(str(input_file), output_file)
        assert result["valid"] == 1
        assert result["skipped"] == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            process_raw_sft("/nonexistent/file.json", "/tmp/out.jsonl")

    def test_uses_default_system_prompt(self, tmp_path):
        raw = [{"question": "q", "thinking": "t", "dsl": _VALID_DSL}]
        input_file = tmp_path / "raw.json"
        input_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        output_file = str(tmp_path / "sft.jsonl")

        process_raw_sft(str(input_file), output_file)
        lines = Path(output_file).read_text(encoding="utf-8").strip().split("\n")
        sample = json.loads(lines[0])
        assert sample["system"] == DEFAULT_SYSTEM_PROMPT


class TestCreateDpoPairs:
    def _make_sft_file(self, tmp_path, n=2):
        samples = []
        for i in range(n):
            sample = build_sft_sample(
                system_prompt="sys",
                question=f"q{i}",
                thinking_content=f"think{i}",
                dsl_dict=_VALID_DSL,
            )
            samples.append(sample)
        sft_file = tmp_path / "sft.jsonl"
        for s in samples:
            sft_file.write_text(
                json.dumps(s, ensure_ascii=False) + "\n",
                encoding="utf-8",
            ) if sft_file.exists() else None
        # Rewrite properly
        with open(sft_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        return str(sft_file)

    def test_generates_pairs(self, tmp_path):
        sft_file = self._make_sft_file(tmp_path, n=2)
        dpo_file = str(tmp_path / "dpo.jsonl")
        result = create_dpo_pairs(sft_file, dpo_file, neg_ratio=3)
        assert result["pairs"] == 6
        assert result["samples"] == 2

        lines = Path(dpo_file).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 6
        pair = json.loads(lines[0])
        assert "conversations" in pair
        assert "chosen" in pair
        assert "rejected" in pair

    def test_neg_ratio_affects_count(self, tmp_path):
        sft_file = self._make_sft_file(tmp_path, n=1)
        dpo_file = str(tmp_path / "dpo.jsonl")
        result = create_dpo_pairs(sft_file, dpo_file, neg_ratio=2)
        assert result["pairs"] == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            create_dpo_pairs("/nonexistent.jsonl", "/tmp/out.jsonl")


class TestNormalizeForTrainer:
    def test_flat_format(self):
        pairs = [
            {
                "conversations": [
                    {"from": "system", "value": "sys"},
                    {"from": "human", "value": "prompt"},
                ],
                "chosen": {"from": "gpt", "value": "chosen_resp"},
                "rejected": {"from": "gpt", "value": "rejected_resp"},
            }
        ]
        flat = normalize_for_trainer(pairs)
        assert len(flat) == 1
        assert flat[0]["prompt"] == "sys\n\nprompt"
        assert flat[0]["chosen"] == "chosen_resp"
        assert flat[0]["rejected"] == "rejected_resp"

    def test_skips_empty(self):
        pairs = [{"conversations": [], "chosen": {}, "rejected": {}}]
        flat = normalize_for_trainer(pairs)
        assert len(flat) == 0
