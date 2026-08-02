"""Tests for SFT presets and LoRA config.

Targets: presets.py, lora.py
"""

from __future__ import annotations

from fusion_trainer.sft.lora import LoRAConfig
from fusion_trainer.sft.presets import SFT_PRESETS


class TestLoRAConfig:
    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.dropout == 0.05
        assert cfg.quantize_base is False
        assert cfg.quant_bits == 4
        assert cfg.use_dora is False
        assert "q_proj" in cfg.target_modules

    def test_from_method_full(self):
        cfg = LoRAConfig.from_method("full")
        assert cfg.rank == 0
        assert cfg.alpha == 0
        assert cfg.quantize_base is False

    def test_from_method_lora(self):
        cfg = LoRAConfig.from_method("lora")
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.quantize_base is False

    def test_from_method_qlora(self):
        cfg = LoRAConfig.from_method("qlora")
        assert cfg.rank == 16
        assert cfg.alpha == 32.0
        assert cfg.quantize_base is True
        assert cfg.quant_bits == 4

    def test_custom_rank(self):
        cfg = LoRAConfig(rank=64, alpha=128.0)
        assert cfg.rank == 64
        assert cfg.alpha == 128.0

    def test_custom_dropout(self):
        cfg = LoRAConfig(dropout=0.1)
        assert cfg.dropout == 0.1

    def test_custom_target_modules(self):
        cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert cfg.target_modules == ["q_proj", "v_proj"]

    def test_quant_bits_8(self):
        cfg = LoRAConfig(quantize_base=True, quant_bits=8)
        assert cfg.quant_bits == 8

    def test_dora(self):
        cfg = LoRAConfig(use_dora=True)
        assert cfg.use_dora is True


class TestSFTPresets:
    def test_all_presets_exist(self):
        assert len(SFT_PRESETS) >= 8

    def test_required_fields(self):
        required = {"model_type", "method", "learning_rate", "num_epochs"}
        for name, preset in SFT_PRESETS.items():
            missing = required - preset.keys()
            assert not missing, f"{name} missing: {missing}"

    def test_positive_epochs(self):
        for name, preset in SFT_PRESETS.items():
            assert preset["num_epochs"] > 0, f"{name} has zero epochs"

    def test_positive_learning_rate(self):
        for name, preset in SFT_PRESETS.items():
            assert preset["learning_rate"] > 0, f"{name} has zero LR"

    def test_7b_lora_preset(self):
        p = SFT_PRESETS["sft_7b_lora"]
        assert p["method"] == "lora"
        assert p["lora_rank"] == 16
        assert p["batch_size"] == 1
        assert p["memory_estimate_gb"] == 16

    def test_7b_full_preset(self):
        p = SFT_PRESETS["sft_7b_full"]
        assert p["method"] == "full"
        assert p["memory_estimate_gb"] == 32

    def test_14b_qlora_preset(self):
        p = SFT_PRESETS["sft_14b_qlora"]
        assert p["method"] == "qlora"
        assert p["quant_bits"] == 4
        assert p["lora_rank"] == 16
        assert p["memory_estimate_gb"] == 28

    def test_32b_qlora_preset(self):
        p = SFT_PRESETS["sft_32b_qlora"]
        assert p["method"] == "qlora"
        assert p["lora_rank"] == 8  # Reduced rank for memory
        assert p["max_seq_length"] == 2048
        assert p["memory_estimate_gb"] == 48

    def test_vlm_7b_preset(self):
        p = SFT_PRESETS["sft_vlm_7b_qlora"]
        assert p["model_type"] == "vlm"
        assert p["method"] == "qlora"
        assert p["memory_estimate_gb"] == 35

    def test_code_7b_preset(self):
        p = SFT_PRESETS["sft_code_7b_lora"]
        assert p["method"] == "lora"
        assert p["lora_rank"] == 32  # Higher rank for code
        assert p["max_seq_length"] == 8192  # Long context
        assert "gate_proj" in p["lora_target_modules"]
        assert "up_proj" in p["lora_target_modules"]

    def test_finance_7b_preset(self):
        p = SFT_PRESETS["sft_finance_7b_lora"]
        assert p["method"] == "lora"
        assert p["num_epochs"] == 5  # More epochs for finance
        assert p["warmup_steps"] == 200

    def test_science_7b_preset(self):
        p = SFT_PRESETS["sft_science_7b_lora"]
        assert p["method"] == "lora"
        assert p["learning_rate"] == 1e-5
