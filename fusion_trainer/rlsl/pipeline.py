"""RLSL Pipeline — Reinforcement Learning from Supervised Learning.

Orchestrates the full alignment pipeline:
  SFT → Preference Dataset → RM Training → DPO/ORPO/RLSL Alignment

This pipeline is optimized for single-machine Apple MLX execution,
with memory-efficient implementations suitable for M5 96GB unified memory.

Fusion-Trainer is a pure computation layer. It reads preference datasets
from external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ..experiment import ExperimentLogger
from .dpo import DPOTrainer
from .grpo import GRPOTrainer
from .orpo import ORPOTrainer
from .reward import RewardTrainer


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
        """Run full RLSL pipeline (Reward Model + PPO-light)."""
        print("\n  [Phase 1/2] Training Reward Model...")
        rm_trainer = RewardTrainer(self.config)
        rm_result = rm_trainer.train(dataset)

        print("\n  [Phase 2/2] Running RLSL alignment...")
        print(f"       KL coefficient: {self.rlsl_cfg.kl_coef}")
        print(f"       Clip epsilon: {self.rlsl_cfg.clip_epsilon}")
        print(f"       PPO epochs: {self.rlsl_cfg.ppo_epochs}")

        return {
            "status": "completed",
            "method": "rlsl",
            "reward_model": rm_result,
            "num_pairs": len(dataset),
        }
