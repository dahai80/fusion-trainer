"""Tests for the RLSL module."""

from __future__ import annotations

from fusion_trainer.config import RLSLConfig, TrainerConfig, DatasetConfig


def test_rlsl_config_defaults():
    """Test RLSLConfig default values."""
    cfg = RLSLConfig()
    assert cfg.method == "dpo"
    assert cfg.beta == 0.1
    assert cfg.kl_coef == 0.04
    assert cfg.learning_rate == 5e-6
    assert cfg.temperature == 0.7


def test_rlsl_config_dpo():
    """Test DPO-specific configuration."""
    cfg = RLSLConfig(method="dpo", beta=0.2)
    assert cfg.method == "dpo"
    assert cfg.beta == 0.2


def test_rlsl_config_rlsl():
    """Test RLSL-specific configuration."""
    cfg = RLSLConfig(
        method="rlsl",
        reward_model="test-reward-model",
        clip_epsilon=0.3,
        ppo_epochs=5,
    )
    assert cfg.method == "rlsl"
    assert cfg.reward_model == "test-reward-model"
    assert cfg.clip_epsilon == 0.3
    assert cfg.ppo_epochs == 5


def test_trainer_config_with_rlsl():
    """Test TrainerConfig with RLSL settings."""
    cfg = TrainerConfig(
        dataset=DatasetConfig(path="/tmp/pref.jsonl"),
        rlsl=RLSLConfig(method="orpo", beta=0.15),
    )
    assert cfg.rlsl.method == "orpo"
    assert cfg.rlsl.beta == 0.15