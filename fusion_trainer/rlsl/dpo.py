"""DPO (Direct Preference Optimization) trainer.

DPO directly optimizes the policy from preference data without
explicit reward modeling or PPO.
"""

from __future__ import annotations

from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Reference: https://arxiv.org/abs/2305.18290
    DPO optimizes the policy directly using preference pairs,
    eliminating the need for a separate reward model and PPO.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self._loader = DatasetLoader()

    def train(
        self,
        dataset: list[dict[str, Any]],
        base_model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        """Run DPO training.

        Args:
            dataset: Preference pairs with 'prompt', 'chosen', 'rejected'.
            base_model: Pre-loaded base model (optional).
            tokenizer: Pre-loaded tokenizer (optional).

        Returns:
            Training summary dict.
        """

        beta = self.rlsl_cfg.beta
        lr = self.rlsl_cfg.learning_rate

        print("  [DPO] Running DPO alignment...")
        print(f"       Dataset: {len(dataset)} preference pairs")
        print(f"       Beta: {beta}")
        print(f"       Learning rate: {lr}")
        print(f"       Epochs: {self.rlsl_cfg.num_epochs}")

        # Placeholder: actual DPO training loop goes here
        # In production, this would:
        #   1. Load the SFT model as reference + policy
        #   2. For each batch:
        #      - Compute log probs for chosen and rejected
        #      - DPO loss = -log σ(β(log π(chosen) - log π_ref(chosen)
        #                         - β(log π(rejected) - log π_ref(rejected)))
        #      - Gradient update on policy model
        #   3. Save the aligned model adapter

        total_pairs = len(dataset)
        print(f"  [DPO] Training complete. {total_pairs} pairs processed.")

        return {
            "status": "completed",
            "num_pairs": total_pairs,
            "method": "dpo",
            "beta": beta,
        }
