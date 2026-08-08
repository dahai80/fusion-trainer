"""RLSL Pipeline — Reinforcement Learning from Supervised Learning.

Orchestrates the full alignment pipeline:
  SFT → Preference Dataset → RM Training → DPO/ORPO/RLSL Alignment

This pipeline is optimized for single-machine Apple MLX execution,
with memory-efficient implementations suitable for M5 96GB unified memory.

Fusion-Trainer is a pure computation layer. It reads preference datasets
from external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ..experiment import ExperimentLogger
from ._errors import RLSLUnsupportedError
from .dpo import DPOTrainer
from .grpo import GRPOTrainer
from .orpo import ORPOTrainer
from .reward import RewardTrainer

logger = logging.getLogger(__name__)

# RLSL Phase 2 (PPO-light 策略梯度更新) 上游追踪 issue。
_UPSTREAM_RLSL_PHASE2 = "https://github.com/dahai80/fusion-mlx/issues/399"


class RLSLEngine:
    """RLSL Pipeline Engine.

    Orchestrates the full alignment pipeline:

        SFT base model → Load preference dataset →
        [Optional: Train RM → RLSL loop] or DPO/ORPO →
        Export aligned model adapter
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self.dataset_cfg = config.dataset
        self._loader = DatasetLoader(model_hub_path=config.model_hub_path)
        self._logger = ExperimentLogger(
            run_name=self.rlsl_cfg.run_name or "rlsl_default",
            output_dir=Path(self.rlsl_cfg.output_dir),
        )

    def run(self) -> dict[str, Any]:
        """Execute the alignment pipeline.

        Returns:
            Summary dict with training results.
        """
        print("╔══ Fusion-Trainer RLSL Pipeline ═════════════════════")
        print(f"║  Method:     {self.rlsl_cfg.method}")
        print(f"║  Dataset:    {self.dataset_cfg.path or self.dataset_cfg.hub_dataset}")
        print(f"║  Output:     {self.rlsl_cfg.output_dir}")
        print("╚══════════════════════════════════════════════════════")

        method = self.rlsl_cfg.method

        # Step 1: Load dataset — GRPO uses prompt data, others use preference pairs
        if method == "grpo":
            print("[1/4] Loading GRPO prompt dataset...")
            dataset = self._loader.load_grpo(
                path=self.dataset_cfg.path,
                hub_dataset=self.dataset_cfg.hub_dataset,
                max_samples=self.dataset_cfg.max_samples or 0,
                max_seq_length=self.dataset_cfg.max_seq_length,
            )
            print(f"      {len(dataset)} GRPO prompts loaded")
            if len(dataset) == 0:
                raise ValueError("No valid GRPO prompt data found")
        else:
            print("[1/4] Loading preference dataset...")
            dataset = self._loader.load_preference(
                path=self.dataset_cfg.path,
                hub_dataset=self.dataset_cfg.hub_dataset,
                max_samples=self.dataset_cfg.max_samples or 0,
                max_seq_length=self.dataset_cfg.max_seq_length,
            )
            print(f"      {len(dataset)} preference pairs loaded")
            if len(dataset) == 0:
                raise ValueError("No valid preference data found")

        # Step 2: Run alignment
        if method == "dpo":
            result = self._run_dpo(dataset)
        elif method == "orpo":
            result = self._run_orpo(dataset)
        elif method == "rlsl":
            result = self._run_rlsl(dataset)
        elif method == "grpo":
            result = self._run_grpo(dataset)
        else:
            raise ValueError(f"Unknown alignment method: {method}")

        # Step 3: Save experiment log
        self._logger.save()

        print("\n      ✓ Alignment complete!")
        print(f"      Method: {method}")
        print(f"      Output: {self.rlsl_cfg.output_dir}")

        return result

    def _run_dpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run DPO alignment."""
        trainer = DPOTrainer(self.config)
        return trainer.train(dataset)

    def _run_orpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run ORPO alignment."""
        trainer = ORPOTrainer(self.config)
        return trainer.train(dataset)

    def _run_grpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run GRPO alignment with sandbox feedback reward."""
        trainer = GRPOTrainer(self.config)
        return trainer.train(dataset)

    def _run_rlsl(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full RLSL pipeline (Reward Model + PPO-light).

        Phase 1 trains a reward model via fusion-mlx #424 (now landed).
        Phase 2 runs a PPO-light policy-update loop, which needs an in-model
        policy-gradient endpoint in fusion-mlx that does not exist yet
        (tracked upstream). Phase 1 runs for real; Phase 2 fails visibly
        rather than returning a fake "completed".
        """
        print("\n  [Phase 1/2] Training Reward Model...")
        rm_trainer = RewardTrainer(self.config)
        rm_result = rm_trainer.train(dataset)
        if rm_result.get("status") != "completed":
            return {
                "status": "failed",
                "method": "rlsl",
                "phase": 1,
                "reward_model": rm_result,
                "num_pairs": len(dataset),
            }

        print("\n  [Phase 2/2] Running RLSL alignment (PPO-light)...")
        print(f"       KL coefficient: {self.rlsl_cfg.kl_coef}")
        print(f"       Clip epsilon: {self.rlsl_cfg.clip_epsilon}")
        print(f"       PPO epochs: {self.rlsl_cfg.ppo_epochs}")
        logger.error(
            "RLSL Phase 2 (PPO-light) unsupported: fusion-mlx has no "
            "policy-gradient endpoint yet (upstream %s)",
            _UPSTREAM_RLSL_PHASE2,
        )
        raise RLSLUnsupportedError(
            f"RLSL Phase 2 (PPO-light policy update) is not available: "
            f"fusion-mlx has no policy-gradient training endpoint yet "
            f"(upstream {_UPSTREAM_RLSL_PHASE2}). Phase 1 reward model trained "
            f"successfully; use DPO/ORPO/GRPO for end-to-end alignment."
        )
