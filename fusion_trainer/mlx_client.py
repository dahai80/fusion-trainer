"""HTTP client for fusion-mlx fine-tune API.

Fusion-Trainer is a pure computation layer. It never imports mlx-lm /
mlx-vlm directly — all training is delegated to fusion-mlx's HTTP API
at /admin/api/fine-tune/* (admin router, loopback-only by default).

Importers/callers:
  - fusion_trainer.sft.trainer uses MLXFineTuneClient to submit jobs
  - fusion_trainer.rlsl.engine adapters (GRPO reward) use it for adapter serving

Affected API:
  POST   /admin/api/fine-tune/jobs                — create training job
  GET    /admin/api/fine-tune/jobs/{id}           — poll job status
  GET    /admin/api/fine-tune/jobs/{id}/stream    — SSE progress stream
  POST   /admin/api/fine-tune/jobs/{id}/cancel    — cancel job
  POST   /admin/api/fine-tune/dpo/jobs            — create DPO job (#399)
  POST   /admin/api/fine-tune/orpo/jobs           — create ORPO job (#399)
  GET    /admin/api/fine-tune/dpo/jobs/{id}       — poll DPO/ORPO job (#399)
  GET    /admin/api/fine-tune/adapters            — list saved adapters
  POST   /admin/api/fine-tune/adapters/{model}/{name}/serve — serve adapter

Data schemas: see FineTuneJob status dict from fusion-mlx
(fusion_mlx.training.service.FineTuneJob.to_dict).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

JOBS_PATH = "/admin/api/fine-tune/jobs"
GRPO_JOBS_PATH = "/api/fine-tune/grpo/jobs"
DPO_JOBS_PATH = "/admin/api/fine-tune/dpo/jobs"
ORPO_JOBS_PATH = "/admin/api/fine-tune/orpo/jobs"
ADAPTERS_PATH = "/admin/api/fine-tune/adapters"


class MLXFineTuneError(RuntimeError):
    pass


class MLXFineTuneClient:
    """Thin HTTP wrapper around fusion-mlx's fine-tune admin API.

    Loopback requests with skip_api_key_verification need no auth. For
    non-loopback or locked-down servers, set FUSION_MLX_ADMIN_TOKEN.
    """

    def __init__(
        self,
        port: int,
        base_url: str = "http://localhost",
        admin_token: str = "",
        timeout: float = 30.0,
    ):
        self._base = f"{base_url}:{port}"
        token = admin_token or os.environ.get("FUSION_MLX_ADMIN_TOKEN", "")
        self._headers = {"X-Fusion-Route": "admin"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        logger.info(
            "MLXFineTuneClient init base=%s auth=%s",
            self._base,
            "token" if self._headers else "none",
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}/v1/models", headers=self._headers, timeout=self._timeout
        )
        resp.raise_for_status()
        return resp.json()

    def list_finetunable_models(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}{JOBS_PATH.replace('/jobs', '/models')}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def create_job(
        self,
        model_id: str,
        dataset: str,
        config: dict[str, Any] | None = None,
        adapter_name: str = "",
    ) -> dict[str, Any]:
        body = {
            "model_id": model_id,
            "dataset": dataset,
            "adapter_name": adapter_name,
            "config": config or {},
        }
        logger.info(
            "Creating fine-tune job model=%s dataset=%s adapter=%s",
            model_id,
            dataset,
            adapter_name or "(auto)",
        )
        resp = httpx.post(
            f"{self._base}{JOBS_PATH}",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise MLXFineTuneError(f"create_job failed {resp.status_code}: {resp.text}")
        return resp.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}{JOBS_PATH}/{job_id}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}{JOBS_PATH}/{job_id}/cancel",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_jobs(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}{JOBS_PATH}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
    ) -> dict[str, Any]:
        """Block until job reaches a terminal state, emitting progress.

        on_progress(job_dict, new_events) is called after each poll.
        Returns the final job dict. Raises MLXFineTuneError on failure
        or timeout.
        """
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"job {job_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # GRPO job lifecycle (fusion-mlx #363 Phase 2: /api/fine-tune/grpo/jobs)
    # ------------------------------------------------------------------

    def create_grpo_job(
        self,
        model_id: str,
        prompts: list[str],
        config: dict[str, Any] | None = None,
        adapter_name: str = "",
    ) -> dict[str, Any]:
        """Create a real GRPO training job (policy-gradient, not RFT).

        fusion-mlx runs the GRPO loop server-side and scores rewards via an
        HTTP callback to config['reward_endpoint'] (POST {prompt, completions}
        -> {rewards}). Returns the GRPOJob dict.
        """
        body = {
            "model_id": model_id,
            "prompts": prompts,
            "adapter_name": adapter_name,
            "config": config or {},
        }
        logger.info(
            "Creating GRPO job model=%s prompts=%d adapter=%s",
            model_id,
            len(prompts),
            adapter_name or "(auto)",
        )
        resp = httpx.post(
            f"{self._base}{GRPO_JOBS_PATH}",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise MLXFineTuneError(
                f"create_grpo_job failed {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def get_grpo_job(self, job_id: str) -> dict[str, Any]:
        resp = httpx.get(
            f"{self._base}{GRPO_JOBS_PATH}/{job_id}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_grpo_job(self, job_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}{GRPO_JOBS_PATH}/{job_id}/cancel",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_grpo_jobs(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}{GRPO_JOBS_PATH}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_grpo_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
    ) -> dict[str, Any]:
        """Block until a GRPO job reaches a terminal state, emitting progress."""
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            job = self.get_grpo_job(job_id)
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"grpo job {job_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # DPO/ORPO job lifecycle (fusion-mlx #399: /admin/api/fine-tune/dpo|orpo/jobs)
    # ------------------------------------------------------------------

    def create_dpo_job(
        self,
        model_id: str,
        preference_pairs: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        adapter_name: str = "",
        method: str = "dpo",
    ) -> dict[str, Any]:
        # method in {"dpo","orpo"}:选端点。fusion-mlx 路由会强制 config.method 匹配端点。
        path = DPO_JOBS_PATH if method == "dpo" else ORPO_JOBS_PATH
        body = {
            "model_id": model_id,
            "preference_pairs": preference_pairs,
            "adapter_name": adapter_name,
            "config": config or {},
        }
        logger.info(
            "Creating %s job model=%s pairs=%d adapter=%s",
            method.upper(),
            model_id,
            len(preference_pairs),
            adapter_name or "(auto)",
        )
        resp = httpx.post(
            f"{self._base}{path}",
            json=body,
            headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise MLXFineTuneError(
                f"create_{method}_job failed {resp.status_code}: {resp.text}"
            )
        return resp.json()

    def create_orpo_job(
        self,
        model_id: str,
        preference_pairs: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        adapter_name: str = "",
    ) -> dict[str, Any]:
        return self.create_dpo_job(
            model_id=model_id,
            preference_pairs=preference_pairs,
            config=config,
            adapter_name=adapter_name,
            method="orpo",
        )

    def get_dpo_job(self, job_id: str) -> dict[str, Any]:
        # DPO 与 ORPO 共用 /dpo/jobs/{id} 查询端点(fusion-mlx 路由:GET /dpo/jobs/{id})。
        resp = httpx.get(
            f"{self._base}{DPO_JOBS_PATH}/{job_id}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def cancel_dpo_job(self, job_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}{DPO_JOBS_PATH}/{job_id}/cancel",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def list_dpo_jobs(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}{DPO_JOBS_PATH}",
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_dpo_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
    ) -> dict[str, Any]:
        # 轮询 DPO/ORPO job 到终态。事件流同 SFT/GRPO 模式。
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            job = self.get_dpo_job(job_id)
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"dpo job {job_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def list_adapters(self, model_id: str = "") -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{self._base}{ADAPTERS_PATH}",
            params={"model_id": model_id} if model_id else None,
            headers=self._headers,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def serve_adapter(self, model_id: str, adapter_name: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self._base}{ADAPTERS_PATH}/{model_id}/{adapter_name}/serve",
            headers=self._headers,
            timeout=self._timeout,
        )
        if resp.status_code >= 400:
            raise MLXFineTuneError(
                f"serve_adapter failed {resp.status_code}: {resp.text}"
            )
        return resp.json()

    # ------------------------------------------------------------------
    # Dataset staging
    # ------------------------------------------------------------------

    @staticmethod
    def stage_dataset(
        samples: list[dict[str, Any]],
        dest_dir: str | Path,
        filename: str = "train.jsonl",
    ) -> str:
        """Write samples to a jsonl file fusion-mlx can consume.

        mlx_lm.tuner accepts {"text": "..."} or {"messages": [...]}.
        Fusion-Trainer's DatasetLoader yields {"input","output"} dicts;
        we convert to chat messages so the tokenizer's chat template
        applies correctly and mask_prompt can work.
        """
        dest = Path(dest_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        out_path = dest / filename
        with open(out_path, "w") as f:
            for item in samples:
                if "messages" in item or "text" in item:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                else:
                    inp = item.get("input", "")
                    outp = item.get("output", "")
                    messages = [
                        {"role": "user", "content": str(inp)},
                        {"role": "assistant", "content": str(outp)},
                    ]
                    f.write(
                        json.dumps({"messages": messages}, ensure_ascii=False) + "\n"
                    )
        logger.info("Staged %d samples to %s", len(samples), out_path)
        # Return the DIRECTORY, not the file: mlx_lm.tuner.load_local_dataset
        # treats data_path as a dir and looks for {train,valid,test}.jsonl in it.
        return str(dest)
