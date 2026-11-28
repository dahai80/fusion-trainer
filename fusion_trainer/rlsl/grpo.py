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
    Per the project rule ("integrate with fusion-mlx only ... for upstream
    issues, file an issue first, then a PR"), this path talks ONLY to
    fusion-mlx's HTTP API.

Importers/callers: fusion_trainer.rlsl.pipeline.RLSLEngine (method="grpo")
Affected API: GRPOTrainer(config).train(dataset) -> dict
Data schemas:
  dataset: list of GRPO-prompt samples {"prompt": str, "answer"?, "tests"?, ...}
  generated: chat completion strings from fusion-mlx
  staged SFT (rft): {"messages": [{"role":"user",...},{"role":"assistant",...}]}
  grpo job (grpo): {model_id, prompts:[str], config:GRPOConfig, adapter_name}
"""

from __future__ import annotations

import contextlib
import difflib
import json
import logging
import math
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import TrainerConfig
from ..mlx_client import MLXFineTuneClient, MLXFineTuneError

if TYPE_CHECKING:
    from ..experiment import ExperimentLogger

logger = logging.getLogger(__name__)

_EPS = 1e-8


class GRPOTrainer:
    """Group Relative Policy Optimization trainer (reward-filtered SFT approx)."""

    def __init__(self, config: TrainerConfig, experiment_logger: Any = None):
        self.config = config
        self.rlsl_cfg = config.rlsl
        self._output_dir = Path(self.rlsl_cfg.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._client = MLXFineTuneClient(port=config.mlx_port)
        self._model = self.config.sft.model
        self._adapter_name = ""
        self._mean_reward = 0.0
        # A6: per-run unique suffix so concurrent GRPO runs on the same base
        # model never collide on adapter_name (fusion-mlx addresses adapters
        # by name under ~/.fusion-mlx/adapters/<model>/<name>/ — same name =
        # silent overwrite). run_name alone collides when empty/default; the
        # uuid8 makes every trainer instance unique.
        self._run_suffix = uuid.uuid4().hex[:8]

        # Optional live-progress sink (RunManager). None = CLI mode.
        self.on_event = None
        # P3-2 (round-3 audit): when an owning engine passes its logger in,
        # stream into it instead of building a throwaway one per train() call
        # (each throwaway opened its own events.jsonl + TB writer). Owned by
        # the engine, so the engine's save()/close() in run() covers it.
        self._injected_logger = experiment_logger

    def close(self) -> None:
        # P1-1 (round-3 audit): trainers built a pooled client but never
        # closed it — long-running daemons leaked a connection pool per run.
        # Mirrors SFTTrainer.close (round-2 E4).
        with contextlib.suppress(Exception):
            self._client.close()

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
        from ..mlx_client import MLXFineTuneError

        # P3-2: reuse the owning engine's logger when provided.
        exp = self._injected_logger or ExperimentLogger(
            run_name=self.rlsl_cfg.run_name or "grpo_default",
            output_dir=self._output_dir,
        )

        reward_trajectory: list[float] = []
        final_status = "completed"
        last_adapter = ""
        # T7 (capability plan): reward-hacking detectors over the scored batches.
        from ..hacking_monitor import HackingMonitor

        monitor = HackingMonitor()

        for it in range(iters):
            print(f"\n  [GRPO iter {it + 1}/{iters}]")
            # P0-4 (round-3 audit): a total sampling outage raises
            # MLXFineTuneError from _sample_and_score — fail the run visibly
            # instead of `continue`-ing every iteration into a zero-training
            # fake "completed".
            try:
                _t_sample = time.monotonic()
                self._rollout_completion_tokens = 0
                samples = self._sample_and_score(dataset, g, mode, scale)
                _sample_sec = time.monotonic() - _t_sample
            except MLXFineTuneError as exc:
                logger.error("GRPO iter %d sampling failed: %s", it, exc)
                final_status = "failed"
                break
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
            monitor.observe(
                it,
                [s["reward"] for s in samples],
                [s.get("completion", "") for s in samples],
            )
            exp.log_metrics(
                step=it,
                metrics={
                    "grpo_iter": it,
                    "mean_reward": mean_r,
                    "kept_samples": len(kept),
                    "total_samples": len(samples),
                    # T9 (capability plan 3.2): rollout throughput — the metric
                    # that quantifies fusion-mlx prefix-cache/continuous-batching
                    # gains on the sampling path.
                    "rollout_samples_per_second": (
                        round(len(samples) / _sample_sec, 4) if _sample_sec > 0 else 0.0
                    ),
                    # P2-5 headline: server-reported completion tokens over
                    # the sampling window (tok/s), not just samples/s.
                    "rollout_tokens_per_second": (
                        round(
                            getattr(self, "_rollout_completion_tokens", 0)
                            / _sample_sec,
                            1,
                        )
                        if _sample_sec > 0
                        else 0.0
                    ),
                    "rollout_completion_tokens": getattr(
                        self, "_rollout_completion_tokens", 0
                    ),
                    "rollout_seconds": round(_sample_sec, 4),
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

        prom = monitor.prom_series()
        if prom:
            exp.log_metrics(step=max(1, len(reward_trajectory)), metrics=prom)
        exp.save()
        if final_status == "completed":
            print("\n      ✓ GRPO complete!")
            print(
                f"      Reward trajectory: {[round(r, 3) for r in reward_trajectory]}"
            )
            print(f"      Final adapter: {last_adapter}")
        else:
            logger.error("GRPO ended with status: %s", final_status)
            print(f"\n      ✗ GRPO failed: status={final_status}")

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
            "hacking_metrics": monitor.snapshot(),
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

        # P3-2: reuse the owning engine's logger when provided.
        exp = self._injected_logger or ExperimentLogger(
            run_name=self.rlsl_cfg.run_name or "grpo_default",
            output_dir=self._output_dir,
        )

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
        adapter_name = f"{self.rlsl_cfg.run_name or 'grpo_run'}_{self._run_suffix}"

        print("╔══ Fusion-Trainer GRPO (true, policy-gradient) ══════")
        print(f"║  Model:      {self._model}")
        print(f"║  Prompts:    {len(prompts)}")
        print(f"║  Group G:    {g}")
        print(f"║  Iters:      {iters}")
        print(f"║  Sandbox:    {mode}  (scale={scale})")
        print("║  Engine:     fusion-mlx /api/fine-tune/grpo/jobs")
        print("╚══════════════════════════════════════════════════════")

        # P2-13: prompt-miss action — default "warn" (legacy uniform-0.5 +
        # loud log). Set FUSION_REWARD_PROMPT_MISS=fail to fail the GRPO job
        # visibly when a prompt isn't in the dataset index, or =zero to
        # degrade to a 0.0 baseline. Fail-visible beats a constant reward.
        import os as _os

        miss_action = _os.environ.get("FUSION_REWARD_PROMPT_MISS", "warn")
        # T7 (capability plan): hacking detectors fed by the reward callback.
        from ..hacking_monitor import HackingMonitor

        monitor = HackingMonitor()
        bench_client = None
        if mode == "bench":
            # 3.1 bench-as-GRPO-verifier: score completions via fusion-bench.
            # BenchClient construction validates the URL — a bad bench_url
            # fails visibly here, before any job submission.
            from ..eval_loop import BenchClient

            bench_client = BenchClient(
                base_url=self.config.sft.bench_url,
                api_key=self.config.sft.bench_api_key,
                tenant_id=self.config.sft.bench_tenant_id,
            )
            logger.info(
                "GRPO(grpo): bench reward backend enabled (bench_url=%s)",
                self.config.sft.bench_url,
            )
        reward_server = RewardServer(
            dataset,
            mode=mode,
            scale=scale,
            prompt_miss_action=miss_action,
            monitor=monitor,
            bench_client=bench_client,
        )
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
            # A2 (round-2 audit): print the REDACTED endpoint only — the raw
            # URL carries the reward-server bearer token and must never reach
            # stdout/log. The real endpoint stays in cfg["reward_endpoint"]
            # for fusion-mlx's verbatim POST (upstream must avoid persisting
            # it — tracked separately).
            print(
                f"      Submitted GRPO job: {job_id}  "
                f"reward_endpoint={reward_server.redacted_endpoint}"
            )
            logger.info(
                "GRPO job %s submitted (reward endpoint redacted=%s)",
                job_id,
                reward_server.redacted_endpoint,
            )
            if self.on_event is not None:
                self.on_event({"type": "job", "job_id": job_id, "kind": "grpo"})

            try:
                final_job = self._client.wait_for_grpo_job(
                    job_id,
                    on_progress=lambda j, ev: self._ingest_grpo_progress(j, ev, exp),
                    poll_interval=2.0,
                    is_canceled=getattr(self, "is_canceled", None),
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

        prom = monitor.prom_series()
        if prom:
            exp.log_metrics(step=iters, metrics=prom)
        exp.save()
        if final_status == "completed":
            print("\n      ✓ GRPO complete!")
            print(f"      Final adapter: {last_adapter}")
        else:
            err = final_job.get("error", "") if isinstance(final_job, dict) else ""
            logger.error("GRPO job ended with status: %s error: %s", final_status, err)
            print(f"\n      ✗ GRPO failed: status={final_status} error={err}")
        return {
            "status": final_status,
            "error": final_job.get("error", "") if final_status != "completed" else "",
            "method": "grpo",
            "backend": "grpo",
            "iterations": iters,
            "num_prompts": len(prompts),
            "group_size": g,
            "job_id": job_id,
            "adapter_path": last_adapter,
            "adapter_name": self._adapter_name,
            "output_dir": str(self._output_dir),
            "hacking_metrics": monitor.snapshot(),
            "bench_degraded": bool(reward_server.bench_degraded),
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
        """For each prompt, sample G completions and score them. Returns flat list.

        P0-4 (round-3 audit): _generate now raises MLXFineTuneError on HTTP
        failure. We catch per-prompt so a single bad prompt is skipped (the
        batch keeps going), but track failures. If EVERY prompt's sampling
        failed, we re-raise — a total sampling outage must surface as a failed
        run, not silently degrade to "no samples → completed → zero training".
        A prompt that sampled but produced only empty completions is a
        separate, benign case (still skipped, not counted as a failure).
        """
        from ..mlx_client import MLXFineTuneError
        from .sandbox import reward_completion

        out: list[dict[str, Any]] = []
        n_gen_failures = 0
        n_attempted = 0
        last_err = ""
        for sample in dataset:
            prompt = sample.get("prompt", "")
            if not prompt:
                continue
            n_attempted += 1
            try:
                completions = self._generate(prompt, g)
            except MLXFineTuneError as exc:
                n_gen_failures += 1
                last_err = str(exc)
                logger.warning("GRPO sample failed for prompt: %s", exc)
                continue
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
        # All prompts attempted and every one failed to sample → fail visibly.
        if n_attempted > 0 and n_gen_failures == n_attempted:
            raise MLXFineTuneError(
                f"GRPO sampling failed for all {n_attempted} prompts "
                f"(last error: {last_err}); cannot produce training data"
            )
        if self.rlsl_cfg.teacher_model:
            self._apply_teacher_anchor(out)
        return out

    # ------------------------------------------------------------------
    # SeTPO-lite teacher anchor (trainer#128, arXiv 2611.18459)
    # ------------------------------------------------------------------

    def _teacher_generate(self, prompt: str) -> str:
        """Fetch ONE teacher reference completion for a gated prompt.

        Separate method (not folded into _generate) so tests can stub teacher
        behavior independently of student sampling. Raises MLXFineTuneError on
        HTTP failure — the caller gates per-prompt and fails soft.
        """
        completions, _ = self._client.chat_completions_with_usage(
            self.rlsl_cfg.teacher_model,
            [{"role": "user", "content": prompt}],
            n=1,
            temperature=0.2,
            top_p=0.9,
            max_tokens=int(self.rlsl_cfg.teacher_max_tokens),
        )
        if not completions:
            raise MLXFineTuneError(
                f"teacher returned no completion for prompt: {prompt[:80]}"
            )
        return completions[0]

    @staticmethod
    def _teacher_agreement(completion: str, reference: str) -> float:
        """Deterministic completion-level agreement with the teacher reference.

        Normalized edit similarity (difflib SequenceMatcher on lowercased,
        whitespace-collapsed text). Stdlib, no model in the loop, stable across
        runs — the lite anchor must not introduce a second source of sampling
        variance into the advantage signal.
        """
        norm = lambda s: " ".join(s.lower().split())
        return difflib.SequenceMatcher(None, norm(completion), norm(reference)).ratio()

    def _apply_teacher_anchor(self, samples: list[dict[str, Any]]) -> None:
        """SeTPO-lite: boost advantages in low-reward groups with teacher signal.

        Selective injection per arXiv 2611.18459: ONLY groups whose mean reward
        is below the teacher_quantile of group means receive teacher feedback —
        high-reward groups keep their pure group-relative advantage, preserving
        student exploration. Completion-level anchor (not token-level KL): the
        advantage bonus is teacher_lambda * agreement(completion, reference).
        Teacher failures fail soft per-prompt (logged, skipped) so a teacher
        outage degrades to plain GRPO, never to a failed run.
        """
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        for s in samples:
            by_prompt.setdefault(s["prompt"], []).append(s)
        if not by_prompt:
            return
        group_means = {
            p: sum(x["reward"] for x in g) / len(g) for p, g in by_prompt.items()
        }
        means_sorted = sorted(group_means.values())
        gate = means_sorted[
            max(
                0,
                min(
                    len(means_sorted) - 1,
                    int(self.rlsl_cfg.teacher_quantile * len(means_sorted)),
                ),
            )
        ]
        lam = float(self.rlsl_cfg.teacher_lambda)
        n_gated = n_bonus = n_fail = 0
        for prompt, group in by_prompt.items():
            if group_means[prompt] >= gate:
                continue
            n_gated += 1
            try:
                reference = self._teacher_generate(prompt)
            except Exception as exc:
                n_fail += 1
                logger.warning("teacher anchor failed for prompt (soft skip): %s", exc)
                continue
            for s in group:
                agreement = self._teacher_agreement(s["completion"], reference)
                s["advantage"] = s["advantage"] + lam * agreement
                s["teacher_gated"] = True
                s["teacher_agreement"] = agreement
                s["teacher_reference"] = reference
                n_bonus += 1
        logger.info(
            "SeTPO-lite teacher anchor: gate=%.4f groups=%d gated=%d boosted=%d teacher_failures=%d",
            gate,
            len(by_prompt),
            n_gated,
            n_bonus,
            n_fail,
        )

    def _generate(self, prompt: str, n: int) -> list[str]:
        """Sample n completions via the pooled MLXFineTuneClient.

        P0-4 (round-3 audit): was a raw httpx.post reaching into private
        _base/_headers, bypassing the unified _request wrapper (no retry, no
        JSON guard) and swallowing EVERY error into `return []`. That made a
        sustained fusion-mlx outage look like "all completions empty" → no
        scored samples → the train loop `continue`d every iteration → the run
        returned status "completed" with an empty reward_trajectory = zero
        training, fake success, wasted GPU.

        Now routes through client.chat_completions, which raises
        MLXFineTuneError on HTTP failure. The caller (_sample_and_score)
        catches per-prompt so one bad prompt doesn't kill the batch, but if
        ALL prompts fail it raises so the run fails visibly.
        """
        completions, usage = self._client.chat_completions_with_usage(
            self._model,
            [{"role": "user", "content": prompt}],
            n=n,
            temperature=float(self.rlsl_cfg.temperature),
            top_p=float(self.rlsl_cfg.top_p),
            max_tokens=int(self.rlsl_cfg.response_length),
        )
        # P2-5: accumulate server-reported completion tokens for the rollout
        # tok/s headline metric. Fails soft — usage may be absent on older
        # servers, in which case the tok/s metric reports 0 while samples/s
        # keeps working.
        with contextlib.suppress(AttributeError, TypeError, ValueError):
            self._rollout_completion_tokens += int(usage.get("completion_tokens") or 0)
        return completions

    # ------------------------------------------------------------------
    # Step 3-4: group-relative advantage + top-k selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_top_k(
        samples: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Keep the top-k completions PER PROMPT by advantage, then merge.

        GRPO computes advantage as a group-relative quantity within each
        prompt's G samples. Selection must therefore be per-prompt: taking a
        global top-k over all prompts biases the batch toward the few prompts
        with the highest reward spread and drops most prompts entirely (audit
        L1). We groupby(prompt), keep min(top_k, G) from each group (highest
        advantage first), then merge preserving prompt diversity.
        """
        if top_k <= 0:
            return list(samples)
        by_prompt: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for s in samples:
            p = s.get("prompt", "")
            if p not in by_prompt:
                by_prompt[p] = []
                order.append(p)
            by_prompt[p].append(s)
        kept: list[dict[str, Any]] = []
        for p in order:
            grp = sorted(by_prompt[p], key=lambda s: s["advantage"], reverse=True)
            kept.extend(grp[:top_k])
        return kept

    # ------------------------------------------------------------------
    # Step 5: stage reward-filtered completions as chat SFT data
    # ------------------------------------------------------------------

    def _stage_sft(
        self,
        kept: list[dict[str, Any]],
        iteration: int,
    ) -> str:
        """Write kept (prompt, completion) pairs as messages-JSONL for fusion-mlx.

        Forward-KL anchor (trainer#128, teacher_demo_mix): when enabled, the
        teacher reference completions carried by gated samples are ALSO staged
        as demonstration rows (one per gated prompt, deduped) — mask_prompt
        makes the engine train on the teacher completion tokens, anchoring the
        policy to teacher behavior (the paper's collapse guard, realized as
        demo mixing since exact token-level KL needs engine logprobs,
        fusion-mlx#1756).
        """
        dest = self._output_dir / f"grpo_iter_{iteration}"
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / "train.jsonl"
        demos: dict[str, str] = {}
        if self.rlsl_cfg.teacher_demo_mix:
            for s in kept:
                ref = s.get("teacher_reference")
                if ref and s["prompt"] not in demos:
                    demos[s["prompt"]] = ref
        with open(out_path, "w") as f:
            for s in kept:
                messages = [
                    {"role": "user", "content": s["prompt"]},
                    {"role": "assistant", "content": s["completion"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            for prompt, ref in demos.items():
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": ref},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
        logger.info(
            "GRPO iter %d staged %d samples (+%d teacher demo rows) to %s",
            iteration,
            len(kept),
            len(demos),
            out_path,
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
        adapter_name = f"grpo_iter_{iteration}_{self._run_suffix}"
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
                is_canceled=getattr(self, "is_canceled", None),
            )
        except MLXFineTuneError as exc:
            return {"status": "failed", "error": str(exc), "adapter_name": adapter_name}

        return {
            "status": final.get("status", "failed"),
            "adapter_path": final.get("adapter_path", ""),
            "adapter_name": adapter_name,
            "job_id": job_id,
        }
