"""Reward model trainer — trains a reward model for RLSL alignment."""

from __future__ import annotations

import logging
from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ._errors import RLSLUnsupportedError

logger = logging.getLogger(__name__)

_UPSTREAM_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/399"


class RewardTrainer:
    """Train a reward model on preference data.

    The reward model learns to score responses: chosen > rejected.
    This is a critical component for the RLSL pipeline.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self._loader = DatasetLoader()

    def train(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train the reward model.

        Args:
            dataset: List of preference pairs with 'prompt', 'chosen', 'rejected'.

        Returns:
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("RewardTrainer: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "reward_model"}

        print("  [RM] Reward model training requested")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Learning rate: {self.rlsl_cfg.reward_learning_rate}")
        print(f"       Epochs: {self.rlsl_cfg.reward_num_epochs}")

        # Reward-model training adds a scalar head + Bradley-Terry pairwise
        # loss, needs in-model gradient updates. Per the project rule this must
        # run in fusion-mlx, which has no RM training endpoint yet (%s).
        logger.error(
            "RewardTrainer unsupported: fusion-mlx has no RM endpoint yet (%s)",
            _UPSTREAM_ISSUE,
        )
        raise RLSLUnsupportedError(
            f"Reward-model training is not available: fusion-mlx has no reward "
            f"model training endpoint yet (upstream {_UPSTREAM_ISSUE}). "
            f"Use GRPO (sandbox reward, already supported) or wait upstream."
        )

    def predict(
        self,
        model_path: str,
        texts: list[str],
    ) -> list[float]:
        """Score texts with a trained reward model.

        No reward model can be trained yet (see train), so prediction is
        unavailable. Raises RLSLUnsupportedError rather than returning fake
        uniform scores.
        """
        logger.error(
            "RewardTrainer.predict unsupported: no RM can be trained yet (%s)",
            _UPSTREAM_ISSUE,
        )
        raise RLSLUnsupportedError(
            f"Reward-model prediction is not available: no reward model can be "
            f"trained yet (upstream {_UPSTREAM_ISSUE})."
        )
