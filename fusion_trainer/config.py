"""Configuration models for Fusion-Trainer.

Fusion-Trainer is a pure computation layer. It does NOT manage the
fusion-model-hub storage — the hub path is simply a configuration
string that tells the Trainer where to look for input data files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_hub_path() -> str:
    """Return the default fusion-model-hub path (read-only reference)."""
    return os.environ.get(
        "FUSION_MODEL_HUB",
        str(Path.home() / "fusion-model-hub"),
    )


def _default_cache_path() -> str:
    return os.environ.get(
        "FUSION_TRAINER_CACHE",
        str(Path.cwd() / "runtime_cache"),
    )


def _default_mlx_port() -> int:
    return int(os.environ.get("FUSION_MLX_PORT", "11432"))


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

DatasetFormat = Literal["jsonl", "parquet", "json", "csv", "arrow"]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    The dataset is always read from an external source. Fusion-Trainer
    never writes to or manages the hub directory.
    """

    # Direct file path to the dataset
    path: str = ""
    """Path to the dataset file, e.g. '/path/to/data.jsonl'."""

    # Hub-based reference
    hub_dataset: str = ""
    """Dataset name within the model hub, e.g. 'sft/code_alpaca'.
    Resolved as: {model_hub_path}/datasets/{hub_dataset}.jsonl"""

    # Format
    format: DatasetFormat = "jsonl"
    """Dataset file format."""

    # Splits
    val_ratio: float = 0.05
    """Ratio of training data to hold out for validation."""

    # Filtering
    max_samples: int = 0
    """0 means use all samples."""

    max_seq_length: int = 4096
    """Truncate samples longer than this."""

    min_seq_length: int = 2
    """Filter samples shorter than this."""

    remove_duplicates: bool = True
    """Remove exact duplicate text entries."""

    # Field mapping
    input_field: str = "text"
    """Field name for input/prompt text."""

    output_field: str = "output"
    """Field name for target/completion text."""

    chat_template: str = ""
    """Chat template to apply (empty = use tokenizer default)."""


# ---------------------------------------------------------------------------
# SFT config
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning configuration."""

    # Model
    model: str = ""
    """Base model name or path (fusion-mlx alias or local path)."""

    model_type: Literal["lm", "vlm"] = "lm"
    """Model type: language model or vision-language model."""

    # Training method
    method: Literal["full", "lora", "qlora"] = "lora"
    """Fine-tuning method."""

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Quantization (for QLoRA)
    quant_bits: int = 4
    """Quantization bits for QLoRA (4 or 8)."""

    # Training hyperparameters
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    lr_scheduler: str = "cosine"
    warmup_steps: int = 100
    num_epochs: int = 3
    max_steps: int = 0
    """0 = compute from num_epochs and dataset size."""

    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Precision
    use_mxfp8: bool = True
    """Use MXFP8 mixed precision training (MLX-native)."""

    # Checkpointing
    save_every: int = 500
    """Save checkpoint every N steps."""

    save_total_limit: int = 3
    """Keep only the most recent N checkpoints."""

    output_dir: str = ""
    """Output directory for checkpoints and final weights (Trainer-local)."""

    # Logging
    log_every: int = 10
    """Log training metrics every N steps."""

    run_name: str = ""
    """Experiment run name for logging."""


# ---------------------------------------------------------------------------
# RLSL config
# ---------------------------------------------------------------------------


@dataclass
class RLSLConfig:
    """Reinforcement Learning from Supervised Learning configuration."""

    # Reward model
    reward_model: str = ""
    """Reward model name or path."""

    reward_model_type: Literal["lm", "classifier"] = "lm"
    """Type of reward model."""

    reward_learning_rate: float = 1e-5
    reward_batch_size: int = 1
    reward_num_epochs: int = 2

    # Alignment method
    method: Literal["dpo", "orpo", "rlsl", "grpo"] = "dpo"
    """Alignment method. 'grpo' = Group Relative Policy Optimization."""

    # DPO / ORPO parameters
    beta: float = 0.1
    """Temperature parameter for DPO/ORPO."""

    # RLSL (PPO-light) parameters
    clip_epsilon: float = 0.2
    kl_coef: float = 0.04
    ppo_epochs: int = 4

    # GRPO parameters
    group_size: int = 8
    """G: samples per prompt for group-relative advantage."""

    grpo_iterations: int = 1
    """Number of GRPO outer iterations (sample → score → SFT-update)."""

    sandbox_mode: Literal["off", "code", "math", "auto"] = "off"
    """Sandbox reward source. 'code' executes tests, 'math' compares answers, 'auto' picks by dataset field."""

    reward_top_k: int = 4
    """Keep top-k of G samples by reward for the SFT-style update (reward-filtered GRPO approximation)."""

    reward_scale: float = 1.0
    """Multiplier applied to raw sandbox reward before ranking."""

    max_tool_calls: int = 3
    """Max tool-call rounds during on-policy generation (Thinking + Tool reinforcement)."""

    # Training
    batch_size: int = 1
    learning_rate: float = 5e-6
    num_epochs: int = 1
    max_steps: int = 0

    # Generation during RL
    response_length: int = 512
    """Max response length for on-policy generation."""

    temperature: float = 0.7
    top_p: float = 0.9

    # Logging
    log_every: int = 10
    save_every: int = 200
    run_name: str = ""
    output_dir: str = ""


# ---------------------------------------------------------------------------
# Top-level trainer config
# ---------------------------------------------------------------------------


@dataclass
class TrainerConfig:
    """Top-level Fusion-Trainer configuration.

    Fusion-Trainer is a pure computation layer. The model_hub_path is
    a read-only reference — the Trainer never creates or modifies
    directories inside the hub.
    """

    # Dataset
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    # SFT
    sft: SFTConfig = field(default_factory=SFTConfig)

    # RLSL
    rlsl: RLSLConfig = field(default_factory=RLSLConfig)

    # Hardware
    device: str = "gpu"
    """Device to use: 'gpu' (default) or 'cpu'."""

    seed: int = 42
    """Random seed for reproducibility."""

    # Paths
    model_hub_path: str = field(default_factory=_default_hub_path)
    """Read-only reference to the fusion-model-hub directory.
    Trainer reads datasets from here but never writes to it."""

    cache_path: str = field(default_factory=_default_cache_path)
    """Trainer-local cache directory for checkpoints and temp data."""

    mlx_port: int = field(default_factory=_default_mlx_port)
    """Port for MLX calls. Default 11432 (fusion-gateway, routes to fusion-mlx).
    Overridable via FUSION_MLX_PORT env var (e.g. 11434 to bypass gateway)."""

    def __post_init__(self):
        if not self.sft.output_dir:
            self.sft.output_dir = os.path.join(
                self.cache_path, "checkpoints", self.sft.run_name or "sft_default"
            )
        if not self.rlsl.output_dir:
            self.rlsl.output_dir = os.path.join(
                self.cache_path, "checkpoints", self.rlsl.run_name or "rlsl_default"
            )
