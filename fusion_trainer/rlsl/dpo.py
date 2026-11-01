"""DPO (Direct Preference Optimization) trainer.

DPO directly optimizes the policy from preference data without
explicit reward modeling or PPO. Delegates the gradient loop to
fusion-mlx's /admin/api/fine-tune/dpo/jobs endpoint (upstream #399).
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

logger = logging.getLogger(__name__)


class DPOTrainer:
    """Direct Preference Optimization trainer.

    Reference: https://arxiv.org/abs/2305.18290
    DPO optimizes the policy directly using preference pairs,
    eliminating the need for a separate reward model and PPO.
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

    def close(self) -> None:
        # P1-1 (round-3 audit): trainers built a pooled client but never
        # closed it — long-running daemons leaked a connection pool per run.
        # Mirrors SFTTrainer.close (round-2 E4).
        with contextlib.suppress(Exception):
            self._client.close()

    def _build_dpo_config(self) -> dict[str, Any]:
        # 映射 RLSLConfig → fusion-mlx DPOConfig 字段(dpo.py:32)。
        # method 由端点强制,这里不设(避免与端点冲突)。
        total_pairs = max(1, len(self._pairs))
        batch_size = max(1, int(self.rlsl_cfg.batch_size))
        # A5: iters = 梯度更新步数 = epoch * (total // batch_size)。旧公式
        # num_epochs * total_pairs 把每个样本当一步,忽略 batch 聚合,按
        # batch_size 倍数过训练(4x 算力浪费 + 过拟合风险)。与 SFT trainer
        # steps_per_epoch = total // batch_size 对齐。
        steps_per_epoch = max(1, total_pairs // batch_size)
        iters = self.rlsl_cfg.num_epochs * steps_per_epoch
        cfg: dict[str, Any] = {
            "iters": max(1, iters),
            "batch_size": batch_size,
            "learning_rate": float(self.rlsl_cfg.learning_rate),
            "beta": float(self.rlsl_cfg.beta),
            "lora_layers": 16,
            "lora_rank": int(self.config.sft.lora_rank),
            "lora_alpha": float(self.config.sft.lora_alpha),
            "lora_dropout": float(self.config.sft.lora_dropout),
            "max_seq_length": int(self.config.dataset.max_seq_length),
            "seed": int(self.config.seed),
        }
        # #1149: pass SFT adapter path for two-stage SFT→DPO chaining.
        # Only include when non-empty — empty means fresh LoRA (backward compat).
        if self.rlsl_cfg.init_adapter_path:
            cfg["init_adapter_path"] = self.rlsl_cfg.init_adapter_path
            logger.info(
                "DPO: chaining on SFT adapter %s (fusion-mlx #1149)",
                self.rlsl_cfg.init_adapter_path,
            )
        # #83: DPO loss graph holds chosen+rejected forwards at once (~2x SFT
        # activation memory) — pass recompute through when asked (fusion-mlx
        # DPOConfig.gradient_checkpointing, #1460). Key only when on: older
        # fusion-mlx 400s on unknown keys.
        if self.rlsl_cfg.gradient_checkpointing:
            cfg["gradient_checkpointing"] = True
            logger.info(
                "DPO: gradient_checkpointing=True: activation recompute on "
                "(fusion-mlx #1460)"
            )
        # #83: resident reference model — the per-pair load-and-evict does 2
        # full base loads per pair and the Metal churn fatals the enforcer on
        # large bases (fusion-mlx #1472). Key only when on.
        if self.rlsl_cfg.keep_ref_resident:
            cfg["keep_ref_resident"] = True
            logger.info(
                "DPO: keep_ref_resident=True: one ref model held for the "
                "whole job (fusion-mlx #1472)"
            )
        return cfg

    def train(
        self,
        dataset: list[dict[str, Any]],
        base_model: Any = None,
        tokenizer: Any = None,
    ) -> dict[str, Any]:
        """Run DPO training via fusion-mlx /admin/api/fine-tune/dpo/jobs.

        Args:
            dataset: Preference pairs with 'prompt', 'chosen', 'rejected'.
            base_model: Pre-loaded base model (unused — fusion-mlx loads).
            tokenizer: Pre-loaded tokenizer (unused — fusion-mlx loads).

        Returns:
            Training summary dict (noop for empty dataset).
        """
        total_pairs = len(dataset)
        if total_pairs == 0:
            logger.warning("DPO: empty dataset, no-op")
            return {"status": "completed", "num_pairs": 0, "method": "dpo"}

        if not self._model:
            raise ValueError("DPO needs sft.model set to a fusion-mlx model id")

        self._pairs = dataset
        pairs = self._normalize_pairs(dataset)
        adapter_name = f"{self.rlsl_cfg.run_name or 'dpo_run'}_{uuid.uuid4().hex[:8]}"

        print("  [DPO] DPO alignment via fusion-mlx #399")
        print(f"       Dataset: {total_pairs} preference pairs")
        print(f"       Beta: {self.rlsl_cfg.beta}")
        print(f"       Learning rate: {self.rlsl_cfg.learning_rate}")

        cfg = self._build_dpo_config()
        job = self._client.create_dpo_job(
            model_id=self._model,
            preference_pairs=pairs,
            config=cfg,
            adapter_name=adapter_name,
        )
        job_id = job.get("job_id", "")
        logger.info("DPO job submitted: %s", job_id)
        print(f"       Submitted DPO job: {job_id}")

        if self.on_event is not None:
            self.on_event({"type": "job", "job_id": job_id, "kind": "dpo"})

        try:
            final_job = self._client.wait_for_dpo_job(
                job_id,
                on_progress=self._on_progress if self.on_event else None,
                poll_interval=2.0,
                is_canceled=getattr(self, "is_canceled", None),
            )
        except MLXFineTuneError as exc:
            logger.error("DPO job failed: %s", exc)
            return {
                "status": "failed",
                "error": str(exc),
                "method": "dpo",
                "job_id": job_id,
                "output_dir": str(self._output_dir),
            }

        status = final_job.get("status", "")
        adapter_path = final_job.get("adapter_path", "")
        self._adapter_name = final_job.get("adapter_name", adapter_name)
        logger.info("DPO job %s ended: %s", job_id, status)
        return {
            "status": status,
            "method": "dpo",
            "num_pairs": total_pairs,
            "job_id": job_id,
            "adapter_path": adapter_path,
            "adapter_name": self._adapter_name,
            "output_dir": str(self._output_dir),
        }

    def _on_progress(
        self, job: dict[str, Any], new_events: list[dict[str, Any]]
    ) -> None:
        """Forward raw DPO job events to the live-progress sink."""
        if self.on_event is None:
            return
        for ev in new_events:
            self.on_event(dict(ev))

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
