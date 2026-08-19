"""GRPO (Group Relative Policy Optimization) trainer.

Reference: https://arxiv.org/abs/2402.03300
GRPO samples G completions per prompt, computes group-relative advantages
A_i = (r_i - mean(r)) / (std(r) + eps), and updates the policy with a
clipped surrogate loss using the policy/ref logprob ratio.

Two backends (config.rlsl.grpo_backend):

  'rft' (default) — reward-filtered SFT approximation. Runs entirely over
    the existing fine-tune HTTP API (no RL endpoint needed):
      1. on-policy sample G completions per prompt via /v1/chat/completions
      2. score each with the sandbox reward (sandbox.py)
      3. compute group-relative advantage A_i = (r_i - mean_g) / (std_g + eps)
      4. keep the top-k highest-advantage completions per prompt
      5. stage those (prompt -> best completions) as chat SFT data
      6. submit one SFT/LoRA fine-tune job to /admin/api/fine-tune/jobs
      7. poll to completion, log real loss + mean reward
    Converges toward higher-reward outputs (RFT/RAFT-style); shares GRPO's
    group-relative ranking signal.

  'grpo' — real policy-gradient GRPO via fusion-mlx /api/fine-tune/grpo/jobs
    (upstream dahai80/fusion-mlx#363 Phase 2, landed 2026-08-05). fusion-mlx
    runs the GRPO loop server-side and calls back a local reward HTTP endpoint
    (reward_server.py -> sandbox.reward_completion) to score completions.
    Per the project rule ("只能对接fusion-mlx...遇到上游问题，先提issue，
    在提pr"), this path talks ONLY to fusion-mlx's HTTP API.

Importers/callers: fusion_trainer.rlsl.pipeline.RLSLEngine (method="grpo")
Affected API: GRPOTrainer(config).train(dataset) -> dict
Data schemas:
  dataset: list of GRPO-prompt samples {"prompt": str, "answer"?, "tests"?, ...}
  generated: chat completion strings from fusion-mlx
  staged SFT (rft): {"messages": [{"role":"user",...},{"role":"assistant",...}]}
  grpo job (grpo): {model_id, prompts:[str], config:GRPOConfig, adapter_name}
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from ..config import TrainerConfig
from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

if TYPE_CHECKING:
    from ..experiment import ExperimentLogger

logger = logging.getLogger(__name__)

_EPS = 1e-8


class GRPOTrainer:
    """Group Relative Policy Optimization trainer (reward-filtered SFT approx)."""

    def __init__(self, config: TrainerConfig):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self._output_dir = Path(self.rlsl_cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._client = MLXFineTuneClient(port=config.mlx_port)
        self._model = self.config.sft.model
        self._adapter_name = ""
        self._mean_reward = 0.0

        # Optional live-progress sink (RunManager). None = CLI mode.
        self.on_event = None

    def train(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Run GRPO outer loop.

        dataset: GRPO-prompt samples (prompt + reference fields).
        Returns a summary dict with per-iteration reward trajectory.
        """
        if not dataset:
            logger.warning("GRPO: empty dataset, no-op")
            return {
                "status": "completed",
                "method": "grpo",
                "iterations": 0,
                "num_prompts": 0,
            }

        if not self._model:
            raise ValueError("GRPO needs sft.model set to a fusion-mlx model id")

        backend = self.rlsl_cfg.grpo_backend
        logger.info("GRPO backend=%s", backend)
        if backend == "grpo":
            return self._run_true_grpo(dataset)

        g = max(2, int(self.rlsl_cfg.group_size))
        top_k = max(1, min(int(self.rlsl_cfg.reward_top_k), g))
        iters = max(1, int(self.rlsl_cfg.grpo_iterations))
        mode = self.rlsl_cfg.sandbox_mode
        scale = float(self.rlsl_cfg.reward_scale)

        print("╔══ Fusion-Trainer GRPO ══════════════════════════════")
        print(f"║  Model:      {self._model}")
        print(f"║  Prompts:    {len(dataset)}")
        print(f"║  Group G:    {g}   top-k: {top_k}")
        print(f"║  Iterations: {iters}")
        print(f"║  Sandbox:    {mode}  (scale={scale})")
        print("║  Engine:     fusion-mlx HTTP (reward-filtered SFT approx)")
        print("╚══════════════════════════════════════════════════════")

        from ..experiment import ExperimentLogger

        exp = ExperimentLogger(
            run_name=self.rlsl_cfg.run_name or "grpo_default",
            output_dir=self._output_dir,
        )

        reward_trajectory: list[float] = []
        final_status = "completed"
        last_adapter = ""

        for it in range(iters):
            print(f"\n  [GRPO iter {it + 1}/{iters}]")
            samples = self._sample_and_score(dataset, g, mode, scale)
            if not samples:
                logger.warning("GRPO iter %d produced no scored samples", it)
                continue

            kept = self._select_top_k(samples, top_k)
            if not kept:
                logger.warning("GRPO iter %d top-k selection empty", it)
                continue

            staged = self._stage_sft(kept, it)

            mean_r = sum(s["reward"] for s in samples) / len(samples)
            self._mean_reward = mean_r
            reward_trajectory.append(mean_r)
            exp.log_metrics(
                step=it,
                metrics={
                    "grpo_iter": it,
                    "mean_reward": mean_r,
                    "kept_samples": len(kept),
                    "total_samples": len(samples),
                },
            )
            print(f"      mean_reward={mean_r:.4f} kept={len(kept)}/{len(samples)}")

            job_result = self._submit_sft_job(staged, it, len(kept))
            if job_result["status"] != "completed":
                final_status = job_result["status"]
                logger.error("GRPO iter %d SFT job %s", it, final_status)
                break
            last_adapter = job_result.get("adapter_path", "")
            self._adapter_name = job_result.get("adapter_name", self._adapter_name)

        exp.save()
        print("\n      ✓ GRPO complete!")
        print(f"      Reward trajectory: {[round(r, 3) for r in reward_trajectory]}")
        print(f"      Final adapter: {last_adapter}")

        return {
            "status": final_status,
            "method": "grpo",
            "iterations": iters,
            "num_prompts": len(dataset),
            "group_size": g,
            "reward_trajectory": reward_trajectory,
            "final_mean_reward": reward_trajectory[-1] if reward_trajectory else 0.0,
            "adapter_path": last_adapter,
            "adapter_name": self._adapter_name,
            "note": "reward-filtered SFT approx (backend=rft); use grpo_backend=grpo for true policy-gradient",
        }

    # ------------------------------------------------------------------
    # True GRPO backend (fusion-mlx #363 Phase 2): /api/fine-tune/grpo/jobs
    # ------------------------------------------------------------------

    def _run_true_grpo(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Submit a real policy-gradient GRPO job to fusion-mlx.

        Hosts a local reward HTTP endpoint (reward_server.py) that fusion-mlx
        calls back to score completions with the sandbox reward. Polls the
        job, streaming grpo_step events (loss, mean_reward, mean_ratio) into
        ExperimentLogger. Returns a summary dict with the final adapter path.
        """
        from ..experiment import ExperimentLogger
        from ..mlx_client import MLXFineTuneError
        from .reward_server import RewardServer

        prompts = [s.get("prompt", "") for s in dataset if s.get("prompt")]
        if not prompts:
            logger.warning("GRPO(grpo): no prompts with non-empty text, no-op")
            return {
                "status": "completed",
                "method": "grpo",
                "backend": "grpo",
                "iterations": 0,
                "num_prompts": 0,
            }

        g = max(2, int(self.rlsl_cfg.group_size))
        iters = max(1, int(self.rlsl_cfg.grpo_iterations))
        mode = self.rlsl_cfg.sandbox_mode
        scale = float(self.rlsl_cfg.reward_scale)
        adapter_name = self.rlsl_cfg.run_name or "grpo_run"

        print("╔══ Fusion-Trainer GRPO (true, policy-gradient) ══════")
        print(f"║  Model:      {self._model}")
        print(f"║  Prompts:    {len(prompts)}")
        print(f"║  Group G:    {g}")
        print(f"║  Iters:      {iters}")
        print(f"║  Sandbox:    {mode}  (scale={scale})")
        print("║  Engine:     fusion-mlx /api/fine-tune/grpo/jobs")
        print("╚══════════════════════════════════════════════════════")

        exp = ExperimentLogger(
            run_name=self.rlsl_cfg.run_name or "grpo_default",
            output_dir=self._output_dir,
        )

        reward_server = RewardServer(dataset, mode=mode, scale=scale)
        endpoint = reward_server.start()
        final_status = "completed"
        last_adapter = ""
        job_id = ""
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
            job = self._client.create_grpo_job(
                model_id=self._model,
                prompts=prompts,
                config=cfg,
                adapter_name=adapter_name,
            )
            job_id = job.get("job_id", "")
            print(f"      Submitted GRPO job: {job_id}  reward_endpoint={endpoint}")
            if self.on_event is not None:
                self.on_event({"type": "job", "job_id": job_id, "kind": "grpo"})

            try:
                final_job = self._client.wait_for_grpo_job(
                    job_id,
                    on_progress=lambda j, ev: self._ingest_grpo_progress(j, ev, exp),
                    poll_interval=2.0,
                )
            except MLXFineTuneError as exc:
                print(f"      ✗ GRPO job failed: {exc}")
                exp.save()
                return {
                    "status": "failed",
                    "error": str(exc),
                    "method": "grpo",
                    "backend": "grpo",
                    "job_id": job_id,
                    "output_dir": str(self._output_dir),
                }

            status = final_job.get("status", "")
            if status != "completed":
                final_status = status
                logger.error("GRPO job ended with status: %s", status)
            last_adapter = final_job.get("adapter_path", "")
            self._adapter_name = final_job.get("adapter_name", adapter_name)
        finally:
            reward_server.stop()

        exp.save()
        print("\n      ✓ GRPO complete!")
        print(f"      Final adapter: {last_adapter}")
        return {
            "status": final_status,
            "method": "grpo",
            "backend": "grpo",
            "iterations": iters,
            "num_prompts": len(prompts),
            "group_size": g,
            "job_id": job_id,
            "adapter_path": last_adapter,
            "adapter_name": self._adapter_name,
            "output_dir": str(self._output_dir),
            "note": "true policy-gradient GRPO via fusion-mlx #363 Phase 2",
        }

    def _ingest_grpo_progress(
        self,
        job: dict[str, Any],
        new_events: list[dict[str, Any]],
        exp_logger: ExperimentLogger,
    ) -> None:
        """Stream fusion-mlx GRPO grpo_step events into ExperimentLogger."""
        for ev in new_events:
            etype = ev.get("type", "")
            if self.on_event is not None:
                self.on_event(dict(ev))
            if etype == "grpo_step":
                step = ev.get("iter", 0)
                exp_logger.log_metrics(
                    step=step,
                    metrics={
                        "grpo_loss": ev.get("loss", 0.0),
                        "mean_reward": ev.get("mean_reward", 0.0),
                        "mean_ratio": ev.get("mean_ratio", 0.0),
                        "n_samples": ev.get("n_samples", 0),
                    },
                )
                print(
                    f"      iter {step:4d}  loss={ev.get('loss', 0.0):.4f}  "
                    f"mean_reward={ev.get('mean_reward', 0.0):.4f}  "
                    f"ratio={ev.get('mean_ratio', 0.0):.4f}"
                )
            elif etype == "done":
                logger.info("GRPO job done: adapter=%s", ev.get("adapter_path", ""))

    def _on_progress(
        self, job: dict[str, Any], new_events: list[dict[str, Any]]
    ) -> None:
        """Forward raw SFT (rft backend) job events to the live-progress sink."""
        if self.on_event is None:
            return
        for ev in new_events:
            self.on_event(dict(ev))

    # ------------------------------------------------------------------
    # Step 1-2: on-policy sampling + sandbox scoring
    # ------------------------------------------------------------------

    def _sample_and_score(
        self,
        dataset: list[dict[str, Any]],
        g: int,
        mode: str,
        scale: float,
    ) -> list[dict[str, Any]]:
        """For each prompt, sample G completions and score them. Returns flat list."""
        from .sandbox import reward_completion

        out: list[dict[str, Any]] = []
        for sample in dataset:
            prompt = sample.get("prompt", "")
            if not prompt:
                continue
            completions = self._generate(prompt, g)
            rewards = [
                reward_completion(sample, c, mode=mode, scale=scale)
                for c in completions
            ]
            if not rewards:
                continue
            mean_r = sum(rewards) / len(rewards)
            std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / len(rewards))
            for c, r in zip(completions, rewards, strict=True):
                adv = (r - mean_r) / (std_r + _EPS)
                out.append(
                    {
                        "prompt": prompt,
                        "completion": c,
                        "reward": r,
                        "advantage": adv,
                        "sample": sample,
                    }
                )
        return out

    def _generate(self, prompt: str, n: int) -> list[str]:
        """Sample n completions from fusion-mlx /v1/chat/completions with temperature."""
        url = f"{self._client._base}/v1/chat/completions"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": float(self.rlsl_cfg.temperature),
            "top_p": float(self.rlsl_cfg.top_p),
            "max_tokens": int(self.rlsl_cfg.response_length),
        }
        try:
            resp = httpx.post(
                url, json=body, headers=self._client._headers, timeout=120.0
            )
            if resp.status_code >= 400:
                logger.warning(
                    "generate failed %d: %s", resp.status_code, resp.text[:200]
                )
                return []
            data = resp.json()
        except Exception as exc:
            logger.warning("generate request error: %s", exc)
            return []

        choices = data.get("choices", [])
        results = []
        for ch in choices:
            msg = ch.get("message", {})
            content = msg.get("content", "")
            if content:
                results.append(content)
        return results

    # ------------------------------------------------------------------
    # Step 3-4: group-relative advantage + top-k selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_top_k(
        samples: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Keep top-k completions by advantage across the whole batch.

        Per-prompt grouping is already reflected in the advantage (group-relative),
        so a global top-k by advantage naturally keeps more from high-spread groups.
        """
        ranked = sorted(samples, key=lambda s: s["advantage"], reverse=True)
        return ranked[:top_k] if top_k > 0 else ranked

    # ------------------------------------------------------------------
    # Step 5: stage reward-filtered completions as chat SFT data
    # ------------------------------------------------------------------

    def _stage_sft(
        self,
        kept: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        """Write kept (prompt, completion) pairs as messages-JSONL for fusion-mlx."""
        dest = self._output_dir / f"grpo_iter_{iteration}"
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / "train.jsonl"
        with open(out_path, "w") as f:
            for s in kept:
                messages = [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s["completion"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        logger.info(
            "GRPO iter %d staged %d samples to %s", iteration, len(kept), out_path
        )
        return str(dest)

    # ------------------------------------------------------------------
    # Step 6-7: submit SFT/LoRA fine-tune job + poll
    # ------------------------------------------------------------------

    def _submit_sft_job(
        self,
        dataset_dir: str,
        iteration: int,
        kept_count: int,
    ) -> dict[str, Any]:
        """Submit a LoRA fine-tune job on the reward-filtered data and poll to completion."""
        adapter_name = f"grpo_iter_{iteration}"
        cfg = {
            "fine_tune_type": "lora",
            "lora_rank": 16,
            "lora_alpha": 32.0,
            "optimizer": "adamw",
            "learning_rate": float(self.rlsl_cfg.learning_rate),
            "batch_size": int(self.rlsl_cfg.batch_size),
            "iters": max(1, int(self.rlsl_cfg.ppo_epochs)),
            "val_batches": 1,
            "steps_per_report": 5,
            "max_seq_length": int(self.config.dataset.max_seq_length),
            "grad_accumulation_steps": 1,
            "mask_prompt": True,
            "seed": int(self.config.seed),
        }
        try:
            job = self._client.create_job(
                model_id=self._model,
                dataset=dataset_dir,
                config=cfg,
                adapter_name=adapter_name,
            )
        except MLXFineTuneError as exc:
            return {"status": "failed", "error": str(exc), "adapter_name": adapter_name}

        job_id = job.get("job_id", "")
        print(f"      Submitted SFT job {job_id} on {kept_count} filtered samples")
        if self.on_event is not None:
            self.on_event({"type": "job", "job_id": job_id, "kind": "sft"})
        try:
            final = self._client.wait_for_job(
                job_id,
                on_progress=self._on_progress if self.on_event else None,
                poll_interval=2.0,
            )
        except MLXFineTuneError as exc:
            return {"status": "failed", "error": str(exc), "adapter_name": adapter_name}

        return {
            "status": final.get("status", "failed"),
            "adapter_path": final.get("adapter_path", ""),
            "adapter_name": adapter_name,
            "job_id": job_id,
        }
