"""DPO (Direct Preference Optimization) trainer.

DPO directly optimizes the policy from preference data without
explicit reward modeling or PPO.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ._errors import RLSLUnsupportedError

logger = logging.getLogger(__name__)

_UPSTREAM_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/399"


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
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("DPO: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "dpo"}

        beta = self.rlsl_cfg.beta
        lr = self.rlsl_cfg.learning_rate
        print("  [DPO] DPO alignment requested")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Beta: {beta}")
        print(f"       Learning rate: {lr}")
        print(f"       Epochs: {self.rlsl_cfg.num_epochs}")

        # DPO needs in-model logprob + gradient updates on chosen/rejected.
        # Per the project rule this must run in fusion-mlx, whose DPO endpoint
        # does not exist yet (tracked in %s). Fail visibly — no fake training.
        logger.error(
            "DPO unsupported: fusion-mlx has no DPO endpoint yet (%s)", _UPSTREAM_ISSUE
        )
        raise RLSLUnsupportedError(
            f"DPO alignment is not available: fusion-mlx has no DPO training "
            f"endpoint yet (upstream {_UPSTREAM_ISSUE}). Use GRPO "
            f"(method=grpo, already supported) or wait for the upstream fix."
        )
