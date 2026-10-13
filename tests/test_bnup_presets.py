"""Tests for the BNUP presets module (issue #59).

Verifies fusion_trainer/presets/bnup.py exposes BNUP_SFT_32B,
BNUP_DPO_32B, and BNUP_PIPELINE as required by the acceptance criteria.
"""

from __future__ import annotations

from fusion_trainer.presets.bnup import (
    BNUP_DPO_32B,
    BNUP_PIPELINE,
    BNUP_PRESETS,
    BNUP_SFT_32B,
)


class TestBnupPresetsModule:
    def test_sft_preset_values(self):
        # #63: PRD baseline = 8-bit QDoRA (fine_tune_type=dora +
        # quantize_base=True + quant_bits=8), no longer QLoRA fallback.
        assert BNUP_SFT_32B["method"] == "qdora"
        assert BNUP_SFT_32B["quant_bits"] == 8
        assert "description" in BNUP_SFT_32B

    def test_dpo_preset_values(self):
        assert BNUP_DPO_32B["method"] == "dpo"
        assert BNUP_DPO_32B["beta"] == 0.1
        assert "description" in BNUP_DPO_32B

    def test_pipeline_preset(self):
        assert BNUP_PIPELINE["sft_preset"] == "bnup_sft_32b"
        assert BNUP_PIPELINE["dpo_preset"] == "bnup_dpo_32b"
        assert BNUP_PIPELINE["chain_adapter"] is True
        assert BNUP_PIPELINE["merge"] is True

    def test_all_three_in_registry(self):
        assert set(BNUP_PRESETS) == {"BNUP_SFT_32B", "BNUP_DPO_32B", "BNUP_PIPELINE"}

    def test_sft_preset_consistent_with_sft_presets(self):
        from fusion_trainer.sft.presets import SFT_PRESETS

        assert dict(SFT_PRESETS["bnup_sft_32b"]) == BNUP_SFT_32B

    def test_dpo_preset_consistent_with_rlsl_presets(self):
        from fusion_trainer.rlsl.presets import RLSL_PRESETS

        assert dict(RLSL_PRESETS["bnup_dpo_32b"]) == BNUP_DPO_32B

    def test_presets_are_copies_not_references(self):
        from fusion_trainer.sft.presets import SFT_PRESETS

        BNUP_SFT_32B["__test_marker"] = True
        assert "__test_marker" not in SFT_PRESETS["bnup_sft_32b"]
        del BNUP_SFT_32B["__test_marker"]
