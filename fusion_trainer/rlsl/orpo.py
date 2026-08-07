"""ORPO (Odds Ratio Preference Optimization) trainer.

ORPO combines SFT and preference alignment in a single stage,
using an odds ratio-based loss to penalize rejected responses
while rewarding chosen ones.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import TrainerConfig
from ._errors import RLSLUnsupportedError

logger = logging.getLogger(__name__)

_UPSTREAM_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/399"


class ORPOTrainer:
    """Odds Ratio Preference Optimization trainer.

    Reference: https://arxiv.org/abs/2403.07691
    ORPO extends SFT by adding a preference loss term that
    maximizes the odds ratio between chosen and rejected responses.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl

    def train(
        self,
        dataset: list[dict[str, Any]],
        base_model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        """Run ORPO training.

        Args:
            dataset: Preference pairs with 'prompt', 'chosen', 'rejected'.
            base_model: Pre-loaded base model (optional).
            tokenizer: Pre-loaded tokenizer (optional).

        Returns:
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("ORPO: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "orpo"}

        beta = self.rlsl_cfg.beta
        lr = self.rlsl_cfg.learning_rate
        print("  [ORPO] ORPO alignment requested")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Lambda (β): {beta}")
        print(f"       Learning rate: {lr}")
        print(f"       Epochs: {self.rlsl_cfg.num_epochs}")

        # ORPO combines SFT + odds-ratio preference loss, needs in-model
        # gradient updates. Per the project rule this must run in fusion-mlx,
        # whose ORPO endpoint does not exist yet (tracked in %s). Fail visibly.
        logger.error(
            "ORPO unsupported: fusion-mlx has no ORPO endpoint yet (%s)",
            _UPSTREAM_ISSUE,
        )
        raise RLSLUnsupportedError(
            f"ORPO alignment is not available: fusion-mlx has no ORPO training "
            f"endpoint yet (upstream {_UPSTREAM_ISSUE}). Use GRPO "
            f"(method=grpo, already supported) or wait for the upstream fix."
        )
