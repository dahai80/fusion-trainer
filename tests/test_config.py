"""Tests for config module — all dataclass defaults, overrides, edge cases.

Targets: config.py
"""

from __future__ import annotations

import pytest

from fusion_trainer.config import (
    DatasetConfig,
    RLSLConfig,
    SFTConfig,
    TrainerConfig,
    _default_cache_path,
    _default_hub_path,
)

# ===========================================================================
# DatasetConfig
# ===========================================================================


class TestDatasetConfig:
    def test_defaults(self):
        cfg = DatasetConfig()
        assert cfg.path == ""
        assert cfg.hub_dataset == ""
        assert cfg.format == "jsonl"
        assert cfg.val_ratio == 0.05
        assert cfg.max_samples == 0
        assert cfg.max_seq_length == 4096
        assert cfg.min_seq_length == 2
        assert cfg.remove_duplicates is True
        assert cfg.input_field == "text"
        assert cfg.output_field == "output"

    def test_custom_path(self):
        cfg = DatasetConfig(path="/custom/data.jsonl")
        assert cfg.path == "/custom/data.jsonl"

    def test_custom_hub_dataset(self):
        cfg = DatasetConfig(hub_dataset="sft/code")
        assert cfg.hub_dataset == "sft/code"

    def test_custom_format(self):
        cfg = DatasetConfig(format="parquet")
        assert cfg.format == "parquet"

    def test_val_ratio_zero(self):
        cfg = DatasetConfig(val_ratio=0.0)
        assert cfg.val_ratio == 0.0

    def test_val_ratio_one(self):
        cfg = DatasetConfig(val_ratio=1.0)
        assert cfg.val_ratio == 1.0

    def test_max_samples(self):
        cfg = DatasetConfig(max_samples=100)
        assert cfg.max_samples == 100

    def test_max_seq_length(self):
        cfg = DatasetConfig(max_seq_length=2048)
        assert cfg.max_seq_length == 2048

    def test_input_output_fields(self):
        cfg = DatasetConfig(input_field="question", output_field="answer")
        assert cfg.input_field == "question"
        assert cfg.output_field == "answer"


# ===========================================================================
# SFTConfig
# ===========================================================================


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.model == ""
        assert cfg.model_type == "lm"
        assert cfg.method == "lora"
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32.0
        assert cfg.lora_dropout == 0.05
        assert cfg.lora_target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert cfg.quant_bits == 4
        assert cfg.batch_size == 1
        assert cfg.gradient_accumulation_steps == 4
        assert cfg.learning_rate == 2e-5
        assert cfg.lr_scheduler == "cosine"
        assert cfg.warmup_steps == 100
        assert cfg.num_epochs == 3
        assert cfg.max_steps == 0
        assert cfg.weight_decay == 0.01
        assert cfg.max_grad_norm == 1.0
        assert cfg.use_mxfp8 is False  # L2: default off (upstream #402 unresolved)
        assert cfg.save_every == 500
        assert cfg.save_total_limit == 3
        assert cfg.output_dir == ""
        assert cfg.log_every == 10
        assert cfg.run_name == ""

    def test_full_method(self):
        cfg = SFTConfig(method="full")
        assert cfg.method == "full"

    def test_qlora_method(self):
        cfg = SFTConfig(method="qlora", quant_bits=8)
        assert cfg.method == "qlora"
        assert cfg.quant_bits == 8

    def test_custom_lora(self):
        cfg = SFTConfig(
            lora_rank=32, lora_alpha=64.0, lora_target_modules=["q_proj", "v_proj"]
        )
        assert cfg.lora_rank == 32
        assert cfg.lora_alpha == 64.0
        assert cfg.lora_target_modules == ["q_proj", "v_proj"]

    def test_vlm_model(self):
        cfg = SFTConfig(model_type="vlm")
        assert cfg.model_type == "vlm"

    def test_max_steps(self):
        cfg = SFTConfig(max_steps=1000)
        assert cfg.max_steps == 1000

    def test_no_mxfp8(self):
        cfg = SFTConfig(use_mxfp8=False)
        assert cfg.use_mxfp8 is False

    def test_custom_checkpointing(self):
        cfg = SFTConfig(save_every=100, save_total_limit=5, log_every=1)
        assert cfg.save_every == 100
        assert cfg.save_total_limit == 5
        assert cfg.log_every == 1

    def test_custom_output(self):
        cfg = SFTConfig(output_dir="/out", run_name="my_run")
        assert cfg.output_dir == "/out"
        assert cfg.run_name == "my_run"

    def test_lora_target_modules_default_is_copy(self):
        cfg1 = SFTConfig()
        cfg2 = SFTConfig()
        # Ensure they are independent instances
        assert cfg1.lora_target_modules is not cfg2.lora_target_modules


# ===========================================================================
# RLSLConfig
# ===========================================================================


class TestRLSLConfig:
    def test_defaults(self):
        cfg = RLSLConfig()
        assert cfg.reward_model == ""
        assert cfg.reward_model_type == "lm"
        assert cfg.reward_learning_rate == 1e-5
        assert cfg.reward_batch_size == 1
        assert cfg.reward_num_epochs == 2
        assert cfg.method == "dpo"
        assert cfg.beta == 0.1
        assert cfg.clip_epsilon == 0.2
        assert cfg.kl_coef == 0.04
        assert cfg.ppo_epochs == 4
        assert cfg.batch_size == 1
        assert cfg.learning_rate == 5e-6
        assert cfg.num_epochs == 1
        assert cfg.max_steps == 0
        assert cfg.response_length == 512
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9

    def test_orpo_method(self):
        cfg = RLSLConfig(method="orpo")
        assert cfg.method == "orpo"

    def test_rlsl_method(self):
        cfg = RLSLConfig(method="rlsl")
        assert cfg.method == "rlsl"

    def test_custom_beta(self):
        cfg = RLSLConfig(beta=0.2)
        assert cfg.beta == 0.2

    def test_reward_model_config(self):
        cfg = RLSLConfig(reward_model="rm-v1", reward_model_type="classifier")
        assert cfg.reward_model == "rm-v1"
        assert cfg.reward_model_type == "classifier"

    def test_generation_params(self):
        cfg = RLSLConfig(response_length=1024, temperature=0.8, top_p=0.95)
        assert cfg.response_length == 1024
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.95

    def test_ppo_params(self):
        cfg = RLSLConfig(clip_epsilon=0.3, kl_coef=0.1, ppo_epochs=8)
        assert cfg.clip_epsilon == 0.3
        assert cfg.kl_coef == 0.1
        assert cfg.ppo_epochs == 8


# ===========================================================================
# TrainerConfig
# ===========================================================================


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.device == "gpu"
        assert cfg.seed == 42
        assert cfg.model_hub_path != ""
        assert cfg.cache_path != ""
        assert cfg.sft.output_dir != ""  # auto-set in __post_init__
        assert cfg.rlsl.output_dir != ""

    def test_custom_seed_device(self):
        cfg = TrainerConfig(seed=123, device="cpu")
        assert cfg.seed == 123
        assert cfg.device == "cpu"

    def test_custom_model_hub_path(self):
        cfg = TrainerConfig(model_hub_path="/custom/hub")
        assert cfg.model_hub_path == "/custom/hub"

    def test_custom_cache_path(self):
        cfg = TrainerConfig(cache_path="/custom/cache")
        assert cfg.cache_path == "/custom/cache"

    def test_output_dir_from_run_name(self):
        cfg = TrainerConfig(
            sft=SFTConfig(run_name="my_run"),
            cache_path="/cache",
        )
        assert "/cache/checkpoints/my_run" in cfg.sft.output_dir

    def test_output_dir_fallback(self):
        cfg = TrainerConfig(cache_path="/cache")
        assert "sft_default" in cfg.sft.output_dir

    def test_rlsl_output_dir_from_run_name(self):
        cfg = TrainerConfig(
            rlsl=RLSLConfig(run_name="my_rlsl"),
            cache_path="/cache",
        )
        assert "/cache/checkpoints/my_rlsl" in cfg.rlsl.output_dir

    def test_explicit_output_dir(self):
        cfg = TrainerConfig(
            sft=SFTConfig(output_dir="/explicit/path"),
        )
        assert cfg.sft.output_dir == "/explicit/path"

    def test_nested_configs(self):
        cfg = TrainerConfig(
            dataset=DatasetConfig(path="/data.jsonl", max_seq_length=2048),
            sft=SFTConfig(model="qwen-7b", method="qlora"),
            rlsl=RLSLConfig(method="dpo", beta=0.2),
        )
        assert cfg.dataset.path == "/data.jsonl"
        assert cfg.sft.model == "qwen-7b"
        assert cfg.rlsl.beta == 0.2


# ===========================================================================
# Path helpers
# ===========================================================================


class TestPathHelpers:
    def test_default_hub_path_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FUSION_MODEL_HUB", "/env/hub")
            assert _default_hub_path() == "/env/hub"

    def test_default_hub_path_fallback(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("FUSION_MODEL_HUB", raising=False)
            path = _default_hub_path()
            assert path.endswith("fusion-model-hub")

    def test_default_cache_path_env(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("FUSION_TRAINER_CACHE", "/env/cache")
            assert _default_cache_path() == "/env/cache"

    def test_default_cache_path_fallback(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("FUSION_TRAINER_CACHE", raising=False)
            path = _default_cache_path()
            assert path.endswith("runtime_cache")
