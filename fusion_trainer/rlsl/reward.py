"""Reward model trainer — trains a reward model for RLSL alignment.

Delegates the Bradley-Terry value-head gradient loop to fusion-mlx's
/admin/api/fine-tune/reward/jobs endpoint (upstream #424).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..mlx_client import MLXFineTuneClient, MLXFineTuneError
from ._errors import RLSLUnsupportedError

logger = logging.getLogger(__name__)

_UPSTREAM_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/424"
_UPSTREAM_SCORING_ISSUE = "https://github.com/dahai80/fusion-mlx/issues/424"


class RewardTrainer:
    """Train a reward model on preference data.

    The reward model learns a scalar value head so chosen > rejected
    (Bradley-Terry pairwise loss). No reference model is needed.
    Runs in fusion-mlx (#424); trainer never imports mlx-lm directly.
    """

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self._output_dir = Path(self.rlsl_cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._client = MLXFineTuneClient(port=config.mlx_port)
        self._model = self.config.sft.model
        self._adapter_name = ""
        self._pairs: list[dict[str, Any]] = []

    def _build_reward_config(self) -> dict[str, Any]:
        # 映射 RLSLConfig → fusion-mlx RewardConfig 字段(reward.py:RewardConfig)。
        total_pairs = max(1, len(self._pairs))
        iters = self.rlsl_cfg.reward_num_epochs * total_pairs
        return {
            "iters": max(1, iters),
            "batch_size": max(1, int(self.rlsl_cfg.reward_batch_size)),
            "learning_rate": float(self.rlsl_cfg.reward_learning_rate),
            "lora_layers": 16,
            "lora_rank": int(self.config.sft.lora_rank),
            "lora_alpha": float(self.config.sft.lora_alpha),
            "lora_dropout": float(self.config.sft.lora_dropout),
            "max_seq_length": int(self.config.dataset.max_seq_length),
            "seed": int(self.config.seed),
        }

    def train(
        self,
        dataset: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Train the reward model via fusion-mlx /admin/api/fine-tune/reward/jobs.

        Args:
            dataset: List of preference pairs with 'prompt', 'chosen', 'rejected'.

        Returns:
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("RewardTrainer: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "reward_model"}

        if not self._model:
            raise ValueError(
                "RewardTrainer needs sft.model set to a fusion-mlx model id"
            )

        self._pairs = dataset
        pairs = self._normalize_pairs(dataset)
        adapter_name = self.rlsl_cfg.run_name or "reward_run"

        print("  [RM] Reward-model training via fusion-mlx #424")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Learning rate: {self.rlsl_cfg.reward_learning_rate}")
        print(f"       Epochs: {self.rlsl_cfg.reward_num_epochs}")

        cfg = self._build_reward_config()
        job = self._client.create_reward_job(
            model_id=self._model,
            preference_pairs=pairs,
            config=cfg,
            adapter_name=adapter_name,
        )
        job_id = job.get("job_id", "")
        logger.info("Reward job submitted: %s", job_id)
        print(f"       Submitted reward job: {job_id}")

        try:
            final_job = self._client.wait_for_reward_job(
                job_id, poll_interval=2.0
            )
        except MLXFineTuneError as exc:
            logger.error("Reward job failed: %s", exc)
            return {
                "status": "failed",
                "error": str(exc),
                "method": "reward_model",
                "job_id": job_id,
                "output_dir": str(self._output_dir),
            }

        status = final_job.get("status", "")
        adapter_path = final_job.get("adapter_path", "")
        self._adapter_name = final_job.get("adapter_name", adapter_name)
        progress = final_job.get("progress", {}) or {}
        logger.info("Reward job %s ended: %s", job_id, status)
        return {
            "status": status,
            "method": "reward_model",
            "num_pairs": total_pairs,
            "job_id": job_id,
            "adapter_path": adapter_path,
            "adapter_name": self._adapter_name,
            "output_dir": str(self._output_dir),
            "loss": progress.get("loss"),
            "reward_margin": progress.get("reward_margin"),
            "acc_chosen": progress.get("acc_chosen"),
        }

    @staticmethod
    def _normalize_pairs(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # fusion-mlx 要求每对含 prompt/chosen/rejected。兼容 {input,chosen,rejected}。
        pairs = []
        for item in dataset:
            prompt = item.get("prompt") or item.get("input") or ""
            chosen = item.get("chosen") or item.get("output") or ""
            rejected = item.get("rejected") or ""
            pairs.append(
                {
                    "prompt": str(prompt),
                    "chosen": str(chosen),
                    "rejected": str(rejected),
                }
            )
        return pairs

    def predict(
        self,
        model_path: str,
        texts: list[str],
    ) -> list[float]:
        """Score texts with a trained reward model.

        fusion-mlx #424 only ships a training endpoint; there is no
        reward-scoring inference endpoint yet. Raises rather than returning
        fake scores.
        """
        logger.error(
            "RewardTrainer.predict unsupported: no RM scoring endpoint yet (%s)",
            _UPSTREAM_SCORING_ISSUE,
        )
        raise RLSLUnsupportedError(
            f"Reward-model prediction is not available: fusion-mlx has no "
            f"reward-scoring endpoint yet (upstream {_UPSTREAM_SCORING_ISSUE})."
        )
