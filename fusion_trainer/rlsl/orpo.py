"""ORPO (Odds Ratio Preference Optimization) trainer.

ORPO combines SFT and preference alignment in a single stage,
using an odds ratio-based loss to penalize rejected responses
while rewarding chosen ones. Delegates the gradient loop to
fusion-mlx's /admin/api/fine-tune/orpo/jobs endpoint (upstream #399).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

logger = logging.getLogger(__name__)


class ORPOTrainer:
    """Odds Ratio Preference Optimization trainer.

    Reference: https://arxiv.org/abs/2403.07691
    ORPO extends SFT by adding a preference loss term that
    maximizes the odds ratio between chosen and rejected responses.
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

        # Optional live-progress sink (RunManager). None = CLI mode.
        self.on_event = None

    def _build_orpo_config(self) -> dict[str, Any]:
        # 映射 RLSLConfig → DPOConfig(method=orpo)。ORPO 用 lambda_odds,beta 被忽略。
        total_pairs = max(1, len(self._pairs))
        batch_size = max(1, int(self.rlsl_cfg.batch_size))
        # A5: iters = epoch * (total // batch_size),梯度步非样本步(见 dpo.py)。
        steps_per_epoch = max(1, total_pairs // batch_size)
        iters = self.rlsl_cfg.num_epochs * steps_per_epoch
        return {
            "iters": max(1, iters),
            "batch_size": batch_size,
            "learning_rate": float(self.rlsl_cfg.learning_rate),
            "lambda_odds": float(self.rlsl_cfg.beta),
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
        base_model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        """Run ORPO training via fusion-mlx /admin/api/fine-tune/orpo/jobs.

        Args:
            dataset: Preference pairs with 'prompt', 'chosen', 'rejected'.
            base_model: Pre-loaded base model (unused — fusion-mlx loads).
            tokenizer: Pre-loaded tokenizer (unused — fusion-mlx loads).

        Returns:
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("ORPO: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "orpo"}

        if not self._model:
            raise ValueError("ORPO needs sft.model set to a fusion-mlx model id")

        self._pairs = dataset
        pairs = _normalize_pairs(dataset)
        adapter_name = f"{self.rlsl_cfg.run_name or 'orpo_run'}_{uuid.uuid4().hex[:8]}"

        print("  [ORPO] ORPO alignment via fusion-mlx #399")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Lambda (odds): {self.rlsl_cfg.beta}")
        print(f"       Learning rate: {self.rlsl_cfg.learning_rate}")

        cfg = self._build_orpo_config()
        job = self._client.create_orpo_job(
            model_id=self._model,
            preference_pairs=pairs,
            config=cfg,
            adapter_name=adapter_name,
        )
        job_id = job.get("job_id", "")
        logger.info("ORPO job submitted: %s", job_id)
        print(f"       Submitted ORPO job: {job_id}")

        if self.on_event is not None:
            self.on_event({"type": "job", "job_id": job_id, "kind": "orpo"})

        try:
            final_job = self._client.wait_for_dpo_job(
                job_id,
                on_progress=self._on_progress if self.on_event else None,
                poll_interval=2.0,
            )
        except MLXFineTuneError as exc:
            logger.error("ORPO job failed: %s", exc)
            return {
                "status": "failed",
                "error": str(exc),
                "method": "orpo",
                "job_id": job_id,
                "output_dir": str(self._output_dir),
            }

        status = final_job.get("status", "")
        adapter_path = final_job.get("adapter_path", "")
        self._adapter_name = final_job.get("adapter_name", adapter_name)
        logger.info("ORPO job %s ended: %s", job_id, status)
        return {
            "status": status,
            "method": "orpo",
            "num_pairs": total_pairs,
            "job_id": job_id,
            "adapter_path": adapter_path,
            "adapter_name": self._adapter_name,
            "output_dir": str(self._output_dir),
        }

    def _on_progress(
        self, job: dict[str, Any], new_events: list[dict[str, Any]]
    ) -> None:
        """Forward raw ORPO job events to the live-progress sink."""
        if self.on_event is None:
            return
        for ev in new_events:
            self.on_event(dict(ev))


def _normalize_pairs(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # fusion-mlx 要求每对含 prompt/chosen/rejected。兼容 {input,chosen,rejected}。
    pairs = []
    for item in dataset:
        prompt = item.get("prompt") or item.get("input") or ""
        chosen = item.get("chosen") or item.get("output") or ""
        rejected = item.get("rejected") or ""
        pairs.append(
            {"prompt": str(prompt), "chosen": str(chosen), "rejected": str(rejected)}
        )
    return pairs
