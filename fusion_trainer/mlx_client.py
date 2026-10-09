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
  POST   /admin/api/fine-tune/reward/jobs         — create reward-model job (#424)
  GET    /admin/api/fine-tune/reward/jobs/{id}    — poll reward job (#424)
  POST   /admin/api/fine-tune/reward/score        — score completions with RM adapter (#431)
  POST   /admin/api/fine-tune/vlm/jobs            — create VLM (vision-lang) fine-tune job (#797/#800)
  GET    /admin/api/fine-tune/vlm/jobs/{id}       — poll VLM job (#797/#800)
  GET    /admin/api/fine-tune/adapters            — list saved adapters
  POST   /admin/api/fine-tune/adapters/{model}/{name}/serve — serve adapter

Data schemas: see FineTuneJob status dict from fusion-mlx
(fusion_mlx.training.service.FineTuneJob.to_dict).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .config import validate_service_url

logger = logging.getLogger(__name__)

JOBS_PATH = "/admin/api/fine-tune/jobs"
GRPO_JOBS_PATH = "/admin/api/fine-tune/grpo/jobs"
DPO_JOBS_PATH = "/admin/api/fine-tune/dpo/jobs"
ORPO_JOBS_PATH = "/admin/api/fine-tune/orpo/jobs"
REWARD_JOBS_PATH = "/admin/api/fine-tune/reward/jobs"
REWARD_SCORE_PATH = "/admin/api/fine-tune/reward/score"
VLM_JOBS_PATH = "/admin/api/fine-tune/vlm/jobs"
ADAPTERS_PATH = "/admin/api/fine-tune/adapters"
MERGE_ADAPTER_PATH = "/v1/merge-adapter"


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
        self._base = f"{validate_service_url(base_url, field_name='base_url')}:{port}"
        token = admin_token or os.environ.get("FUSION_MLX_ADMIN_TOKEN", "")
        self._headers = {"X-Fusion-Route": "admin"}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._timeout = timeout
        # A1: a shared, pooled client instead of a fresh connection per call.
        # Connection reuse cuts handshake overhead across the long poll loops;
        # the pool also bounds resource use. Transport-level retry is handled
        # per-call (see _poll_get) so transient blips don't kill hour-long runs.
        self._http = httpx.Client(headers=self._headers, timeout=self._timeout)
        logger.info(
            "MLXFineTuneClient init base=%s auth=%s",
            self._base,
            "token" if self._headers else "none",
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._http.close()

    def __del__(self):
        self.close()

    # A8: unified request wrapper. Every HTTP egress now goes through the
    # pooled self._http client (not module-level httpx.get/post), retries
    # transient network errors + 5xx with exponential backoff, raises a single
    # MLXFineTuneError type (never a raw httpx.HTTPStatusError/TransportError
    # — the old mix let raw exceptions escape wait_for_*_job's
    # `except MLXFineTuneError` and kill multi-hour runs), and guards
    # resp.json() against non-JSON bodies (gateway HTML). tolerate_status lets
    # cancel_*_job treat a 404 (job already gone) as success so L3 zombie
    # cleanup isn't aborted by one missing job.
    def _request(
        self,
        method: str,
        url: str,
        *,
        json_body: Any = None,
        params: Any = None,
        max_retries: int = 5,
        tolerate_status: tuple[int, ...] = (),
    ) -> dict[str, Any] | list[dict[str, Any]]:
        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._http.request(method, url, json=json_body, params=params)
            except (httpx.TransportError, httpx.RequestError) as exc:
                last_exc = exc
                logger.warning(
                    "%s %s transient error (attempt %d/%d): %s",
                    method,
                    url,
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(min(backoff, 30.0))
                    backoff *= 2
                    continue
                raise MLXFineTuneError(
                    f"{method} {url} failed after {max_retries} retries: {exc}"
                ) from exc
            if resp.status_code in tolerate_status:
                logger.info("%s %s tolerated status %d", method, url, resp.status_code)
                return {}
            if resp.status_code >= 500:
                last_exc = MLXFineTuneError(f"{resp.status_code} {resp.text[:200]}")
                logger.warning(
                    "%s %s server error %d (attempt %d/%d)",
                    method,
                    url,
                    resp.status_code,
                    attempt,
                    max_retries,
                )
                if attempt < max_retries:
                    time.sleep(min(backoff, 30.0))
                    backoff *= 2
                    continue
                raise MLXFineTuneError(
                    f"{method} {url} returned {resp.status_code} after {max_retries} retries"
                )
            if resp.status_code >= 400:
                raise MLXFineTuneError(
                    f"{method} {url} failed {resp.status_code}: {resp.text}"
                )
            return self._parse_json(resp, method, url)
        raise MLXFineTuneError(f"{method} {url} exhausted retries: {last_exc}")

    @staticmethod
    def _parse_json(resp: httpx.Response, method: str, url: str) -> Any:
        # A8: guard JSON decode — fusion-gateway / a proxy can return HTML on
        # error, which made the old resp.json() raise a raw JSONDecodeError
        # outside any MLXFineTuneError wrapper.
        try:
            return resp.json()
        except (ValueError, json.JSONDecodeError) as exc:
            body_preview = resp.text[:200] if resp.text else "(empty)"
            raise MLXFineTuneError(
                f"{method} {url} returned non-JSON body "
                f"(status {resp.status_code}): {body_preview}"
            ) from exc

    # A1/A8: poll getter — now a thin wrapper over _request (GET, retried).
    # Kept as a named method because callers (wait_for_*_job) expect it and
    # tests patch it; the retry/JSON-guard logic lives once in _request.
    def _poll_get(self, url: str, max_retries: int = 5) -> dict[str, Any]:
        result = self._request("GET", url, max_retries=max_retries)
        if isinstance(result, dict):
            return result
        # A8: a poll unexpectedly returned a list — surface as a contract
        # error rather than letting callers .get() on a list.
        raise MLXFineTuneError(
            f"poll get {url} returned a list, expected a job dict: {result!r}"
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        # A8 note: health is a liveness probe (info/Settings panel), NOT a
        # training poll — it must fail FAST so an unreachable fusion-mlx flips
        # mlx_alive=False in seconds, not after 5 retried backoffs (~45s). One
        # attempt, no retry; the caller (info/RunManager) catches the error.
        result = self._request("GET", f"{self._base}/v1/models", max_retries=1)
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("health returned non-dict")

    def list_finetunable_models(self) -> list[dict[str, Any]]:
        # A8 note: read probe — fail fast (one attempt), no retried backoff.
        result = self._request(
            "GET",
            f"{self._base}{JOBS_PATH.replace('/jobs', '/models')}",
            max_retries=1,
        )
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_finetunable_models returned non-list")

    def chat_completions(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> list[str]:
        """Sample completions via fusion-mlx /v1/chat/completions.

        P0-4 (round-3 audit): GRPO on-policy sampling now routes through the
        unified _request wrapper (pooled client + retry/backoff + JSON guard +
        single MLXFineTuneError) instead of a raw httpx.post that swallowed
        every error into `return []` — producing a zero-training fake
        "completed". A failed sample now raises MLXFineTuneError so the caller
        can fail visibly rather than silently degrade to no training.

        Returns the list of non-empty completion strings (one per choice).
        """
        body = {
            "model": model_id,
            "messages": messages,
            "n": n,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        # Sampling is a single user-facing call (not a long poll), but a
        # transient 5xx should still retry a couple times before failing the
        # GRPO iteration. max_retries=3 balances responsiveness vs resilience.
        result = self._request(
            "POST",
            f"{self._base}/v1/chat/completions",
            json_body=body,
            max_retries=3,
        )
        if not isinstance(result, dict):
            raise MLXFineTuneError(f"chat/completions returned non-dict: {result!r}")
        choices = result.get("choices", [])
        out: list[str] = []
        for ch in choices:
            if isinstance(ch, dict):
                content = (ch.get("message") or {}).get("content", "")
                if content:
                    out.append(str(content))
        return out

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
        result = self._request("POST", f"{self._base}{JOBS_PATH}", json_body=body)
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("create_job returned non-dict")

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._poll_get(f"{self._base}{JOBS_PATH}/{job_id}")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        # A8: tolerate 404 — job already gone (e.g. L3 zombie cleanup after a
        # restart where fusion-mlx evicted the job). The old raise_for_status
        # let a 404 abort the whole _load_existing_runs sweep.
        result = self._request(
            "POST",
            f"{self._base}{JOBS_PATH}/{job_id}/cancel",
            tolerate_status=(404,),
        )
        if isinstance(result, dict):
            return result
        return {}

    def list_jobs(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self._base}{JOBS_PATH}")
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_jobs returned non-list")

    def wait_for_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
        is_canceled=None,
    ) -> dict[str, Any]:
        """Block until job reaches a terminal state, emitting progress.

        on_progress(job_dict, new_events) is called after each poll.
        is_canceled() -> bool, checked each poll; True raises MLXFineTuneError
        (P1-3: so stop_run interrupts a multi-hour wait instead of blocking
        until the 86400s timeout). Returns the final job dict. Raises
        MLXFineTuneError on failure or timeout.

        P3-3 (round-3 audit): the new-events slice assumes fusion-mlx's job
        ``events`` list is append-only and monotonically growing across polls
        (events[last_seen:] where last_seen = prior len). If fusion-mlx ever
        truncates, re-orders, or resets the events array mid-job, this slice
        would miss or replay events. Upstream contract: events are append-only
        for the lifetime of a job. If that changes, switch to an event-id
        cursor instead of a positional index.
        """
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            # A1: a poll failure after retries must not kill the whole run —
            # the job may still be healthy upstream. Keep polling until the
            # real timeout; only a genuinely terminal job status (or deadline)
            # ends the wait.
            try:
                job = self.get_job(job_id)
            except MLXFineTuneError as exc:
                logger.warning("wait_for_job poll failed, continuing: %s", exc)
                time.sleep(poll_interval)
                continue
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            # P1-3 (round-3 audit): honor a cancel flag between polls so
            # stop_run / Ctrl-C interrupts the wait promptly.
            if is_canceled is not None and is_canceled():
                logger.info("wait_for_job %s interrupted by cancel", job_id)
                raise MLXFineTuneError(f"job {job_id} cancelled by user")
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"job {job_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # VLM job lifecycle (fusion-mlx #797/#800: /admin/api/fine-tune/vlm/jobs —
    # vision-language SFT/LoRA over HTTP, drives mlx_vlm.trainer. Parallel to
    # the text SFT path; same job lifecycle, different config field names and
    # a dataset that is a HF dataset id or local images+jsonl dir (NOT the
    # staged chat-messages jsonl). Blocks fusion-trainer #55.)
    # ------------------------------------------------------------------

    def create_vlm_job(
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
            "Creating VLM fine-tune job model=%s dataset=%s adapter=%s",
            model_id,
            dataset,
            adapter_name or "(auto)",
        )
        result = self._request("POST", f"{self._base}{VLM_JOBS_PATH}", json_body=body)
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("create_vlm_job returned non-dict")

    def get_vlm_job(self, job_id: str) -> dict[str, Any]:
        return self._poll_get(f"{self._base}{VLM_JOBS_PATH}/{job_id}")

    def cancel_vlm_job(self, job_id: str) -> dict[str, Any]:
        # A8: tolerate 404 — job already gone (mirrors cancel_job).
        result = self._request(
            "POST",
            f"{self._base}{VLM_JOBS_PATH}/{job_id}/cancel",
            tolerate_status=(404,),
        )
        if isinstance(result, dict):
            return result
        return {}

    def list_vlm_jobs(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self._base}{VLM_JOBS_PATH}")
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_vlm_jobs returned non-list")

    def wait_for_vlm_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
        is_canceled=None,
    ) -> dict[str, Any]:
        # Poll VLM job to terminal state. Same append-only events contract as
        # text SFT (see wait_for_job). A1 retry-on-poll-failure + P1-3 cancel
        # honored — only the poll path differs.
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            try:
                job = self.get_vlm_job(job_id)
            except MLXFineTuneError as exc:
                logger.warning("wait_for_vlm_job poll failed, continuing: %s", exc)
                time.sleep(poll_interval)
                continue
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            if is_canceled is not None and is_canceled():
                logger.info("wait_for_vlm_job %s interrupted by cancel", job_id)
                raise MLXFineTuneError(f"vlm job {job_id} cancelled by user")
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"vlm job {job_id} timed out after {timeout}s")

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
        result = self._request("POST", f"{self._base}{GRPO_JOBS_PATH}", json_body=body)
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("create_grpo_job returned non-dict")

    def get_grpo_job(self, job_id: str) -> dict[str, Any]:
        return self._poll_get(f"{self._base}{GRPO_JOBS_PATH}/{job_id}")

    def cancel_grpo_job(self, job_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"{self._base}{GRPO_JOBS_PATH}/{job_id}/cancel",
            tolerate_status=(404,),
        )
        if isinstance(result, dict):
            return result
        return {}

    def list_grpo_jobs(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self._base}{GRPO_JOBS_PATH}")
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_grpo_jobs returned non-list")

    def wait_for_grpo_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
        is_canceled=None,
    ) -> dict[str, Any]:
        """Block until a GRPO job reaches a terminal state, emitting progress."""
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            # A1: keep polling past transient poll failures (see wait_for_job).
            try:
                job = self.get_grpo_job(job_id)
            except MLXFineTuneError as exc:
                logger.warning("wait_for_grpo_job poll failed, continuing: %s", exc)
                time.sleep(poll_interval)
                continue
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            # P1-3 (round-3 audit): honor cancel between polls.
            if is_canceled is not None and is_canceled():
                logger.info("wait_for_grpo_job %s interrupted by cancel", job_id)
                raise MLXFineTuneError(f"grpo job {job_id} cancelled by user")
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
        result = self._request("POST", f"{self._base}{path}", json_body=body)
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError(f"create_{method}_job returned non-dict")

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
        return self._poll_get(f"{self._base}{DPO_JOBS_PATH}/{job_id}")

    def cancel_dpo_job(self, job_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"{self._base}{DPO_JOBS_PATH}/{job_id}/cancel",
            tolerate_status=(404,),
        )
        if isinstance(result, dict):
            return result
        return {}

    def list_dpo_jobs(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self._base}{DPO_JOBS_PATH}")
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_dpo_jobs returned non-list")

    def wait_for_dpo_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
        is_canceled=None,
    ) -> dict[str, Any]:
        # 轮询 DPO/ORPO job 到终态。事件流同 SFT/GRPO 模式。
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            # A1: keep polling past transient poll failures (see wait_for_job)。
            try:
                job = self.get_dpo_job(job_id)
            except MLXFineTuneError as exc:
                logger.warning("wait_for_dpo_job poll failed, continuing: %s", exc)
                time.sleep(poll_interval)
                continue
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            # P1-3 (round-3 audit): honor cancel between polls.
            if is_canceled is not None and is_canceled():
                logger.info("wait_for_dpo_job %s interrupted by cancel", job_id)
                raise MLXFineTuneError(f"dpo job {job_id} cancelled by user")
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"dpo job {job_id} timed out after {timeout}s")

    # ------------------------------------------------------------------
    # Reward-model job lifecycle (fusion-mlx #424:
    # /admin/api/fine-tune/reward/jobs — Bradley-Terry value-head 训练)
    # ------------------------------------------------------------------

    def create_reward_job(
        self,
        model_id: str,
        preference_pairs: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
        adapter_name: str = "",
    ) -> dict[str, Any]:
        # 训练一个标量 reward head,使 chosen 分数 > rejected(Bradley-Terry)。
        # preference_pairs 每对含 prompt/chosen/rejected。无 reference model。
        body = {
            "model_id": model_id,
            "preference_pairs": preference_pairs,
            "adapter_name": adapter_name,
            "config": config or {},
        }
        logger.info(
            "Creating REWARD job model=%s pairs=%d adapter=%s",
            model_id,
            len(preference_pairs),
            adapter_name or "(auto)",
        )
        result = self._request(
            "POST", f"{self._base}{REWARD_JOBS_PATH}", json_body=body
        )
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("create_reward_job returned non-dict")

    def get_reward_job(self, job_id: str) -> dict[str, Any]:
        return self._poll_get(f"{self._base}{REWARD_JOBS_PATH}/{job_id}")

    def cancel_reward_job(self, job_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"{self._base}{REWARD_JOBS_PATH}/{job_id}/cancel",
            tolerate_status=(404,),
        )
        if isinstance(result, dict):
            return result
        return {}

    def list_reward_jobs(self) -> list[dict[str, Any]]:
        result = self._request("GET", f"{self._base}{REWARD_JOBS_PATH}")
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_reward_jobs returned non-list")

    def wait_for_reward_job(
        self,
        job_id: str,
        on_progress=None,
        poll_interval: float = 2.0,
        timeout: float = 86400.0,
        is_canceled=None,
    ) -> dict[str, Any]:
        # 轮询 reward job 到终态。progress 含 loss/reward_margin/acc_chosen。
        deadline = time.monotonic() + timeout
        last_seen = 0
        while time.monotonic() < deadline:
            # A1: keep polling past transient poll failures (see wait_for_job)。
            try:
                job = self.get_reward_job(job_id)
            except MLXFineTuneError as exc:
                logger.warning("wait_for_reward_job poll failed, continuing: %s", exc)
                time.sleep(poll_interval)
                continue
            events = job.get("events", [])
            if len(events) > last_seen:
                new_events = events[last_seen:]
                last_seen = len(events)
                if on_progress is not None:
                    on_progress(job, new_events)
            status = job.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                return job
            # P1-3 (round-3 audit): honor cancel between polls.
            if is_canceled is not None and is_canceled():
                logger.info("wait_for_reward_job %s interrupted by cancel", job_id)
                raise MLXFineTuneError(f"reward job {job_id} cancelled by user")
            time.sleep(poll_interval)
        raise MLXFineTuneError(f"reward job {job_id} timed out after {timeout}s")

    def score_reward(
        self,
        model_id: str,
        adapter_name: str,
        prompt: str,
        completions: list[str],
    ) -> dict[str, Any]:
        # fusion-mlx #431: 用已训练的 RM adapter 对 (prompt, completions) 打分。
        # POST /admin/api/fine-tune/reward/score -> {rewards, model_id, adapter_name}。
        # 上游端点要求 model_id/adapter_name(可放 body 或 query),prompt 非空,
        # completions 非空 list;不存在 adapter 返回 404,非 RM 返回 400。
        body = {
            "model_id": model_id,
            "adapter_name": adapter_name,
            "prompt": prompt,
            "completions": completions,
        }
        logger.info(
            "Scoring reward model=%s adapter=%s n_completions=%d",
            model_id,
            adapter_name,
            len(completions),
        )
        result = self._request(
            "POST", f"{self._base}{REWARD_SCORE_PATH}", json_body=body
        )
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("score_reward returned non-dict")

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def merge_adapter(
        self,
        model: str,
        adapter_path: str,
        output_path: str = "",
        dequantize: bool = False,
        upload_repo: str = "",
    ) -> dict[str, Any]:
        """Merge a trained LoRA/DoRA adapter into the base model via fusion-mlx.

        POST /v1/merge-adapter (convert router, admin-gated). fusion-mlx
        loads base+adapter, fuses LoRA/DoRA layers, optionally dequantizes,
        and writes SafeTensors to output_path. Returns {status, model,
        output_path}. This is a synchronous call (blocks until merge done).
        """
        body: dict[str, Any] = {
            "model": model,
            "adapter_path": adapter_path,
            "dequantize": dequantize,
        }
        if output_path:
            body["output_path"] = output_path
        if upload_repo:
            body["upload_repo"] = upload_repo
        logger.info(
            "merge_adapter: model=%s adapter=%s -> %s (dequantize=%s)",
            model,
            adapter_path,
            output_path or "(auto)",
            dequantize,
        )
        result = self._request(
            "POST",
            f"{self._base}{MERGE_ADAPTER_PATH}",
            json_body=body,
            max_retries=1,
        )
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("merge_adapter returned non-dict")

    def list_adapters(self, model_id: str = "") -> list[dict[str, Any]]:
        # A8 note: a read/list probe for GUI refresh + publish precheck — fail
        # fast (one attempt) so a down fusion-mlx doesn't hang a refresh for
        # ~45s of retried backoffs. Training polls keep the retry budget.
        params = {"model_id": model_id} if model_id else None
        result = self._request(
            "GET", f"{self._base}{ADAPTERS_PATH}", params=params, max_retries=1
        )
        if isinstance(result, list):
            return result
        raise MLXFineTuneError("list_adapters returned non-list")

    def serve_adapter(self, model_id: str, adapter_name: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            f"{self._base}{ADAPTERS_PATH}/{model_id}/{adapter_name}/serve",
        )
        if isinstance(result, dict):
            return result
        raise MLXFineTuneError("serve_adapter returned non-dict")

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
