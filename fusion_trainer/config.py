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
    # R16: default 11434 (fusion-mlx direct), not 11432 (fusion-gateway). The
    # gateway proxies /v1/* but NOT /admin/api/fine-tune/*, so a default of
    # 11432 made every create_job hit a 404 out-of-the-box — a footgun. Users
    # who want the gateway for inference can set FUSION_MLX_PORT=11432.
    # Upstream: fusion-mlx should have the gateway proxy /admin/api/fine-tune/*.
    return int(os.environ.get("FUSION_MLX_PORT", "11434"))


def _default_hub_url() -> str:
    # fusion-gateway (:11432) proxies /api/v1/* to fusion-model-hub.
    return os.environ.get("FUSION_MODEL_HUB_URL", "http://localhost:11432")


def _default_hub_api_key() -> str:
    return os.environ.get("FUSION_MODEL_HUB_API_KEY", "")


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------


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

    # Splits
    val_ratio: float = 0.05
    """Ratio of training data to hold out for validation."""

    # Filtering
    max_samples: int = 0
    """0 means use all samples."""

    max_seq_length: int = 4096
    """Truncate samples longer than this."""

    remove_duplicates: bool = True
    """Remove exact duplicate text entries."""

    # Field mapping
    input_field: str = "text"
    """Field name for input/prompt text."""

    output_field: str = "output"
    """Field name for target/completion text."""


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
    method: Literal["full", "lora", "dora", "qlora"] = "lora"
    """Fine-tuning method: full / lora / dora / qlora (DoRA via fusion-mlx #402)."""

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
    use_mxfp8: bool = False
    """Use MXFP8 mixed precision training (MLX-native). Upstream-blocked
    (fusion-mlx#402); default False. Explicit True fail-stops until upstream
    lands, so the user's intent is never silently downgraded (audit L2)."""

    # Checkpointing
    save_every: int = 500
    """Save checkpoint every N steps."""

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
    ppo_epochs: int = 4

    # GRPO parameters
    group_size: int = 8
    """G: samples per prompt for group-relative advantage."""

    grpo_iterations: int = 1
    """Number of GRPO outer iterations (sample → score → SFT-update)."""

    sandbox_mode: Literal["off", "code", "math", "auto"] = "off"
    """Sandbox reward source. 'code' executes tests, 'math' compares answers, 'auto' picks by dataset field."""

    grpo_backend: Literal["rft", "grpo"] = "rft"
    """GRPO execution backend.
    'rft'  — reward-filtered SFT approximation (on-policy sample -> sandbox
             score -> top-k -> SFT job via /admin/api/fine-tune/jobs). Works
             today; converges toward higher-reward outputs (RFT/RAFT-style).
    'grpo' — real policy-gradient GRPO via fusion-mlx /api/fine-tune/grpo/jobs
             (requires fusion-mlx #363 Phase 2). Trainer hosts a local reward
             HTTP endpoint that fusion-mlx calls back to score completions."""

    reward_top_k: int = 4
    """Keep top-k of G samples by reward for the SFT-style update (reward-filtered GRPO approximation)."""

    reward_scale: float = 1.0
    """Multiplier applied to raw sandbox reward before ranking."""

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
    """Port for MLX calls. Default 11434 (fusion-mlx direct). The fusion-gateway
    (:11432) proxies /v1/* but NOT /admin/api/fine-tune/*, so a default of 11432
    made every create_job 404 out-of-the-box (R16). Overridable via FUSION_MLX_PORT
    env var (set 11432 to use the gateway for inference-only paths)."""

    # Model-hub publish (issue #49). After a successful LoRA run, the produced
    # adapter is registered to fusion-model-hub via POST /api/v1/models
    # (model_type=lora). Off by default — opt-in via publish_adapter=True.
    hub_url: str = field(default_factory=_default_hub_url)
    """fusion-model-hub API base URL. Default gateway :11432 (proxies /api/v1/*).
    Overridable via FUSION_MODEL_HUB_URL env var."""

    hub_api_key: str = field(default_factory=_default_hub_api_key)
    """API key for model-hub (Authorization: Bearer ...). Empty = no auth.
    Overridable via FUSION_MODEL_HUB_API_KEY env var."""

    publish_adapter: bool = False
    """Publish the trained LoRA adapter to fusion-model-hub on completion.
    Off by default — set True (or via YAML) to enable. Failures warn loudly
    but do NOT block the training result (fail-visible, never silent)."""

    def __post_init__(self):
        # E2 (round-2 audit): two runs without --run-name both resolved to
        # checkpoints/sft_default, so the second overwrote the first's
        # adapter_config.json / staged_data / logs. Append a short uuid suffix
        # when no run_name is given so each unnamed run gets a unique dir.
        # (Real adapter weights live in fusion-mlx's adapters/<model>/<name>/
        # which A6 already uniquifies; this is the local pointer/staging dir.)
        import uuid

        if not self.sft.output_dir:
            base = self.sft.run_name or "sft_default"
            suffix = "" if self.sft.run_name else "_" + uuid.uuid4().hex[:8]
            self.sft.output_dir = os.path.join(
                self.cache_path, "checkpoints", f"{base}{suffix}"
            )
        if not self.rlsl.output_dir:
            base = self.rlsl.run_name or "rlsl_default"
            suffix = "" if self.rlsl.run_name else "_" + uuid.uuid4().hex[:8]
            self.rlsl.output_dir = os.path.join(
                self.cache_path, "checkpoints", f"{base}{suffix}"
            )
