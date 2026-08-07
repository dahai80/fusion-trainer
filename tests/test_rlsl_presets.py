"""Tests for RLSL presets.

Targets: rlsl/presets.py
"""

from __future__ import annotations

from fusion_trainer.rlsl.presets import RLSL_PRESETS


class TestRLSLPresets:
    def test_all_presets_exist(self):
        assert len(RLSL_PRESETS) >= 3

    def test_required_fields(self):
        required = {
            "description",
            "method",
            "grpo_backend",
            "group_size",
            "learning_rate",
            "memory_estimate_gb",
        }
        for name, preset in RLSL_PRESETS.items():
            missing = required - preset.keys()
            assert not missing, f"{name} missing: {missing}"

    def test_all_presets_target_grpo(self):
        # DPO/ORPO/reward are upstream-blocked (fusion-mlx#399); presets target GRPO.
        for name, preset in RLSL_PRESETS.items():
            assert preset["method"] == "grpo", f"{name} is not grpo"

    def test_positive_group_size(self):
        for name, preset in RLSL_PRESETS.items():
            assert preset["group_size"] > 0, f"{name} zero group_size"

    def test_positive_learning_rate(self):
        for name, preset in RLSL_PRESETS.items():
            assert preset["learning_rate"] > 0, f"{name} zero LR"

    def test_grpo_7b_preset(self):
        p = RLSL_PRESETS["rlsl_grpo_7b"]
        assert p["grpo_backend"] == "rft"
        assert p["memory_estimate_gb"] == 22

    def test_grpo_14b_preset(self):
        p = RLSL_PRESETS["rlsl_grpo_14b"]
        assert p["grpo_backend"] == "rft"
        assert p["memory_estimate_gb"] == 62

    def test_grpo_14b_true_backend_preset(self):
        p = RLSL_PRESETS["rlsl_grpo_14b_true"]
        assert p["grpo_backend"] == "grpo"
        assert p["memory_estimate_gb"] == 62

    def test_importable_from_package(self):
        from fusion_trainer.rlsl import RLSL_PRESETS as VIA_PKG

        assert VIA_PKG is RLSL_PRESETS
