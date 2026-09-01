"""RLSL Pipeline — Reinforcement Learning from Supervised Learning.

Orchestrates the full alignment pipeline:
  SFT → Preference Dataset → RM Training → DPO/ORPO/RLSL Alignment

This pipeline is optimized for single-machine Apple MLX execution,
with memory-efficient implementations suitable for M5 96GB unified memory.

Fusion-Trainer is a pure computation layer. It reads preference datasets
from external paths and writes outputs to its own cache directory.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from pathlib import Path
from typing import Any

from ..config import TrainerConfig
from ..dataset import DatasetLoader
from ..experiment import ExperimentLogger
from .dpo import DPOTrainer
from .grpo import GRPOTrainer
from .orpo import ORPOTrainer
from .reward import RewardTrainer

logger = logging.getLogger(__name__)

# RLSL Phase 2 真实闭环上游能力(均已落地):
#   #363 PR#366 — 真实 policy-gradient GRPO 端点 /api/fine-tune/grpo/jobs
#   #430 PR#432 — GRPO worker 线程 stream 修复
#   #431 PR#433 — reward/score 推理打分端点(加载 RM adapter 打分)
# Phase 1 reward model 训练用 #424(本仓 PR #41 已对接)。
_UPSTREAM_GRPO = "https://github.com/dahai80/fusion-mlx/issues/363"
_UPSTREAM_REWARD_SCORE = "https://github.com/dahai80/fusion-mlx/issues/431"


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

        # Optional live-progress sink (RunManager). None = CLI mode.
        self.on_event = None

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
        try:
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

            # Step 3: Save experiment log (success path)
            self._logger.save()
        finally:
            # A3: every exit path must release the logger's file handles —
            # the empty-dataset ValueErrors above and any trainer exception
            # used to skip save()/close(), leaking the metrics fd and the TB
            # async writer. close() is idempotent and safe after save().
            self._logger.close()

        print("\n      ✓ Alignment complete!")
        print(f"      Method: {method}")
        print(f"      Output: {self.rlsl_cfg.output_dir}")

        return result

    def _run_dpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run DPO alignment."""
        trainer = DPOTrainer(self.config)
        trainer.on_event = self.on_event
        trainer.is_canceled = getattr(self, "is_canceled", None)
        try:
            return trainer.train(dataset)
        finally:
            # P1-1 (round-3 audit): close the trainer's pooled client.
            trainer.close()

    def _run_orpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run ORPO alignment."""
        trainer = ORPOTrainer(self.config)
        trainer.on_event = self.on_event
        trainer.is_canceled = getattr(self, "is_canceled", None)
        try:
            return trainer.train(dataset)
        finally:
            trainer.close()

    def _run_grpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run GRPO alignment with sandbox feedback reward."""
        trainer = GRPOTrainer(self.config)
        trainer.on_event = self.on_event
        trainer.is_canceled = getattr(self, "is_canceled", None)
        try:
            return trainer.train(dataset)
        finally:
            trainer.close()

    def _forward_grpo_events(
        self, job: dict[str, Any], new_events: list[dict[str, Any]]
    ) -> None:
        """Forward raw Phase-2 GRPO job events to the live-progress sink."""
        if self.on_event is None:
            return
        for ev in new_events:
            self.on_event(dict(ev))

    def _run_rlsl(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run full RLSL pipeline (Reward Model + real GRPO policy update).

        Phase 1 trains a reward model via fusion-mlx #424 (PR #429 landed),
        producing an RM adapter. Phase 2 runs a real policy-gradient GRPO
        loop via fusion-mlx #363, with the reward signal coming from that RM
        through the reward/score endpoint (#431). The closed loop:
            RM train (#424) -> RM reward server -> reward/score (#431)
            -> GRPO (#363, PPO-clipped policy gradient)
        """
        from ..mlx_client import MLXFineTuneClient, MLXFineTuneError
        from .rm_reward_server import RMRewardServer

        print("\n  [Phase 1/2] Training Reward Model...")
        if self.on_event is not None:
            self.on_event({"type": "phase", "phase": 1, "name": "reward_model_train"})
        rm_trainer = RewardTrainer(self.config)
        rm_trainer.on_event = self.on_event
        rm_trainer.is_canceled = getattr(self, "is_canceled", None)
        try:
            rm_result = rm_trainer.train(dataset)
        finally:
            # P1-1 (round-3 audit): close the RM trainer's pooled client even
            # on the early-return failure paths below.
            rm_trainer.close()
        if rm_result.get("status") != "completed":
            logger.error("RLSL Phase 1 reward model failed: %s", rm_result)
            return {
                "status": "failed",
                "method": "rlsl",
                "phase": 1,
                "reward_model": rm_result,
                "num_pairs": len(dataset),
            }

        model_id = self.config.sft.model
        rm_adapter = rm_result.get("adapter_name", "")
        if not model_id or not rm_adapter:
            logger.error(
                "RLSL Phase 1 produced no usable adapter: model=%s adapter=%s",
                model_id,
                rm_adapter,
            )
            return {
                "status": "failed",
                "method": "rlsl",
                "phase": 1,
                "reward_model": rm_result,
                "num_pairs": len(dataset),
                "error": "Phase 1 reward model produced no adapter name",
            }

        print("\n  [Phase 2/2] Running RLSL alignment (real GRPO, RM-rewarded)...")
        print(f"       RM adapter: {rm_adapter}")
        print(f"       Group G:    {self.rlsl_cfg.group_size}")
        print(f"       Clip eps:   {self.rlsl_cfg.clip_epsilon}")
        print(f"       Iters:      {self.rlsl_cfg.grpo_iterations}")
        if self.on_event is not None:
            self.on_event({"type": "phase", "phase": 2, "name": "grpo_policy_update"})

        prompts = [
            str(item.get("prompt", "")) for item in dataset if item.get("prompt")
        ]
        if not prompts:
            logger.error("RLSL Phase 2: no prompts derivable from dataset")
            return {
                "status": "failed",
                "method": "rlsl",
                "phase": 2,
                "reward_model": rm_result,
                "error": "no prompts for GRPO",
                "num_pairs": len(dataset),
            }

        client = MLXFineTuneClient(port=self.config.mlx_port)
        reward_server = RMRewardServer(
            client, model_id=model_id, adapter_name=rm_adapter
        )
        endpoint = reward_server.start()
        # A2 (round-2 audit): log/print the REDACTED endpoint — the raw URL
        # carries the reward-server bearer token and must never reach stdout
        # or log files. The real endpoint stays in cfg["reward_endpoint"] for
        # fusion-mlx's verbatim POST.
        logger.info(
            "RLSL Phase 2: RM reward server on %s (model=%s adapter=%s auth=token)",
            reward_server.redacted_endpoint,
            model_id,
            rm_adapter,
        )
        print(f"       RM reward endpoint: {reward_server.redacted_endpoint}")

        g = max(2, int(self.rlsl_cfg.group_size))
        iters = max(1, int(self.rlsl_cfg.grpo_iterations))
        # A6: uuid8 suffix — concurrent RLSL runs with empty run_name used to
        # collide on "rlsl_grpo", overwriting each other's adapter weights.
        grpo_adapter = f"{self.rlsl_cfg.run_name or 'rlsl'}_grpo_{uuid.uuid4().hex[:8]}"
        job_id = ""
        phase2_status = "completed"
        last_adapter = ""
        last_adapter_name = grpo_adapter
        try:
            cfg = {
                "group_size": g,
                "iters": iters,
                "batch_size": max(1, int(self.rlsl_cfg.batch_size)),
                "learning_rate": float(self.rlsl_cfg.learning_rate),
                "lora_layers": 16,
                "lora_rank": int(self.config.sft.lora_rank),
                "lora_alpha": float(self.config.sft.lora_alpha),
                "lora_dropout": float(self.config.sft.lora_dropout),
                "max_completion_len": int(self.rlsl_cfg.response_length),
                "clip_ratio": float(self.rlsl_cfg.clip_epsilon),
                "reward_endpoint": endpoint,
                "temperature": float(self.rlsl_cfg.temperature),
                "seed": int(self.config.seed),
            }
            logger.info(
                "RLSL Phase 2: submitting GRPO job model=%s prompts=%d",
                model_id,
                len(prompts),
            )
            job = client.create_grpo_job(
                model_id=model_id,
                prompts=prompts,
                config=cfg,
                adapter_name=grpo_adapter,
            )
            job_id = job.get("job_id", "")
            print(f"       Submitted GRPO job: {job_id}")
            if self.on_event is not None:
                self.on_event({"type": "job", "job_id": job_id, "kind": "grpo"})

            try:
                final_job = client.wait_for_grpo_job(
                    job_id,
                    on_progress=self._forward_grpo_events if self.on_event else None,
                    poll_interval=2.0,
                    is_canceled=getattr(self, "is_canceled", None),
                )
            except MLXFineTuneError as exc:
                logger.error("RLSL Phase 2 GRPO job failed: %s", exc)
                return {
                    "status": "failed",
                    "method": "rlsl",
                    "phase": 2,
                    "reward_model": rm_result,
                    "grpo_job_id": job_id,
                    "error": str(exc),
                    "num_pairs": len(dataset),
                }

            phase2_status = final_job.get("status", "")
            last_adapter = final_job.get("adapter_path", "")
            last_adapter_name = final_job.get("adapter_name", grpo_adapter)
            progress = final_job.get("progress", {}) or {}
            logger.info(
                "RLSL Phase 2 GRPO job %s ended: %s mean_reward=%s",
                job_id,
                phase2_status,
                progress.get("mean_reward"),
            )
            print(f"       GRPO status: {phase2_status}")
            print(f"       Final adapter: {last_adapter}")
        finally:
            reward_server.stop()
            # P1-1 (round-3 audit): close the Phase-2 pooled client — the RM
            # reward server held a ref, and the local `client` was never closed
            # on the early-return / exception paths, leaking a pool per RLSL run.
            with contextlib.suppress(Exception):
                client.close()

        overall = "completed" if phase2_status == "completed" else phase2_status
        return {
            "status": overall,
            "method": "rlsl",
            "num_pairs": len(dataset),
            "reward_model": rm_result,
            "phase2": {
                "method": "grpo",
                "backend": "grpo",
                "group_size": g,
                "iterations": iters,
                "job_id": job_id,
                "adapter_path": last_adapter,
                "adapter_name": last_adapter_name,
                "mean_reward": (final_job.get("progress", {}) or {}).get("mean_reward"),
            },
            "note": "RLSL closed loop: RM train (#424) -> reward/score (#431) -> GRPO (#363)",
        }
