"""Tests for the SFT module."""

from __future__ import annotations

from fusion_trainer.config import SFTConfig, TrainerConfig, DatasetConfig
from fusion_trainer.sft.lora import LoRAConfig
from fusion_trainer.sft.presets import SFT_PRESETS


def test_lora_config_defaults():
    """Test default LoRA configuration."""
    cfg = LoRAConfig()
    assert cfg.rank == 16
    assert cfg.alpha == 32.0
    assert cfg.dropout == 0.05
    assert "q_proj" in cfg.target_modules


def test_lora_config_from_method():
    """Test LoRA config creation from method name."""
    full = LoRAConfig.from_method("full")
    assert full.rank == 0
    assert full.quantize_base is False

    lora = LoRAConfig.from_method("lora")
    assert lora.rank == 16
    assert lora.quantize_base is False

    qlora = LoRAConfig.from_method("qlora")
    assert qlora.rank == 16
    assert qlora.quant_bits == 4
    assert qlora.quantize_base is True


def test_sft_presets_exist():
    """Test that all SFT presets have the required fields."""
    required = {"model_type", "method", "learning_rate", "num_epochs"}
    for name, preset in SFT_PRESETS.items():
        assert required.issubset(preset.keys()), f"{name} missing: {required - preset.keys()}"
        assert preset["num_epochs"] > 0
        assert preset["learning_rate"] > 0


def test_sft_config_defaults():
    """Test SFTConfig default values."""
    cfg = SFTConfig()
    assert cfg.method == "lora"
    assert cfg.lora_rank == 16
    assert cfg.batch_size == 1
    assert cfg.learning_rate == 2e-5
    assert cfg.num_epochs == 3


def test_trainer_config():
    """Test TrainerConfig with model_hub_path as config string (not a managed directory)."""
    cfg = TrainerConfig(
        dataset=DatasetConfig(path="/tmp/test.jsonl"),
        sft=SFTConfig(model="test-model", method="lora"),
        model_hub_path="/custom/hub/path",
    )
    assert cfg.dataset.path == "/tmp/test.jsonl"
    assert cfg.sft.model == "test-model"
    assert cfg.sft.method == "lora"
    assert cfg.model_hub_path == "/custom/hub/path"
    assert cfg.sft.output_dir != ""  # auto-set from defaults