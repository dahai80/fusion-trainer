"""RLSL training presets — predefined hyperparameter bundles for GRPO.

GRPO is the only alignment method currently wired end-to-end (DPO/ORPO/
reward-model are upstream-blocked, see fusion-mlx#399). These presets target
GRPO with sensible defaults per model size, including a memory_estimate_gb
for M5 96GB capacity planning — mirroring the SFT_PRESETS convention.

The CLI's rlsl command does not yet consume these (preset support is SFT-only
today); they are exposed for programmatic use and as documented reference
configs. See README "Reinforcement Learning from Supervised Learning (RLSL)".
"""

from __future__ import annotations

RLSL_PRESETS: dict[str, dict] = {
    "rlsl_grpo_7b": {
        "description": "GRPO (reward-filtered SFT) for 7B-class models (e.g. Qwen2.5-7B)",
        "method": "grpo",
        "grpo_backend": "rft",
        "sandbox_mode": "off",
        "group_size": 8,
        "grpo_iterations": 1,
        "reward_top_k": 4,
        "reward_scale": 1.0,
        "batch_size": 1,
        "learning_rate": 5e-6,
        "num_epochs": 1,
        "response_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tool_calls": 3,
        "log_every": 10,
        "save_every": 200,
        "memory_estimate_gb": 22,
    },
    "rlsl_grpo_14b": {
        "description": "GRPO (reward-filtered SFT) for 14B-class models (e.g. Qwen2.5-14B) on M5 96GB",
        "method": "grpo",
        "grpo_backend": "rft",
        "sandbox_mode": "off",
        "group_size": 8,
        "grpo_iterations": 1,
        "reward_top_k": 4,
        "reward_scale": 1.0,
        "batch_size": 1,
        "learning_rate": 5e-6,
        "num_epochs": 1,
        "response_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tool_calls": 3,
        "log_every": 10,
        "save_every": 200,
        "memory_estimate_gb": 62,
    },
    "rlsl_grpo_14b_true": {
        "description": "GRPO real policy-gradient (grpo backend, needs fusion-mlx #363 Phase 2) for 14B on M5 96GB",
        "method": "grpo",
        "grpo_backend": "grpo",
        "sandbox_mode": "off",
        "group_size": 8,
        "grpo_iterations": 1,
        "reward_top_k": 4,
        "reward_scale": 1.0,
        "batch_size": 1,
        "learning_rate": 5e-6,
        "num_epochs": 1,
        "response_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tool_calls": 3,
        "log_every": 10,
        "save_every": 200,
        "memory_estimate_gb": 62,
    },
}
