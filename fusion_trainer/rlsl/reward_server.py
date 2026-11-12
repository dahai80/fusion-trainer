"""Local HTTP reward server for real GRPO (fusion-mlx #363 Phase 2).

fusion-mlx's GRPO trainer scores completions by calling back an HTTP
endpoint: POST {prompt, completions} -> {rewards}. This module hosts a
tiny stdlib HTTP server that maps the prompt string back to its dataset
sample (which carries the reference answer/tests) and delegates to
sandbox.reward_completion.

Importers/callers: fusion_trainer.rlsl.grpo (backend="grpo" path)
Affected API: RewardServer.start() -> str (reward_endpoint URL), .stop()
Data schemas:
  request  {"prompt": str, "completions": [str, ...]}
  response {"rewards": [float, ...]}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .sandbox import reward_completion

logger = logging.getLogger(__name__)

_REWARD_PATH = "/reward"

# Capability plan 3.1: sandbox_mode="bench" routes scoring through the
# fusion-bench task API (bench-as-GRPO-verifier). A sample must carry a
# `bench_task` id; the per-completion scoring call goes through the
# injected bench client (eval_loop.BenchClient-compatible: run_benchmark).
# Bench outage degrades fail-visibly to the sandbox reward for that batch
# (same fallback semantics as eval_loop #75) and sets .bench_degraded.


class _BenchScorer:
    """Score one completion against its dataset sample's bench task.

    Deterministic adapter (Rule 5): no model routing here — the bench task
    definition owns the scoring logic. Per-(task, completion) cache keeps
    repeat completions from re-billing a bench run. Upstream gap: fusion-bench
    has no first-class per-item scoring endpoint yet (tracked in the plan's
    U5 issue) — this adapter speaks the generic task API with the completion
    in params, which U5 will make first-class.
    """

    def __init__(self, bench_client: Any, executor_key: str = "accuracy"):
        self._client = bench_client
        self._executor = executor_key
        self._cache: dict[tuple[str, str], float] = {}

    def score(self, sample: dict[str, Any], completion: str, scale: float) -> float:
        bench_task = str(sample.get("bench_task", "") or "")
        if not bench_task:
            logger.warning(
                "bench scorer: sample has no bench_task field — reward 0.0 "
                "(fail-visible: bench mode requires bench_task on every sample)"
            )
            return 0.0
        key = (bench_task, hashlib.sha1(completion.encode("utf-8")).hexdigest())
        if key in self._cache:
            return self._cache[key] * scale
        params = dict(sample.get("bench_params") or {})
        params["bench_task"] = bench_task
        if sample.get("prompt"):
            params["prompt"] = sample["prompt"]
        params["completion"] = completion
        out = self._client.run_benchmark(
            model=str(sample.get("bench_model", "grpo-verifier")),
            executor_key=str(sample.get("bench_executor", self._executor)),
            params=params,
            poll_interval=1.0,
            wait_timeout=float(sample.get("bench_timeout", 300)),
        )
        score = out.get("score")
        if score is None:
            raise ValueError(
                f"bench task {bench_task} returned no score (status={out.get('status')})"
            )
        self._cache[key] = float(score)
        return float(score) * scale


class RewardServer:
    """Thread-safe local reward endpoint for fusion-mlx GRPO callback.

    Builds a prompt->sample index from the dataset so the sandbox reward
    (which needs the reference answer/tests) can score completions. If a
    prompt is unknown, falls back to the auto reward with an empty sample
    (uniform 0.5 * scale) and logs a LOUD warning (Rule 12: never silently
    degrade the reward signal — P2 fix).

    Security (audit C1): the endpoint is a private callback for fusion-mlx
    only. A one-time bearer token is generated at start() and embedded in
    the endpoint URL as ?token=...; fusion-mlx POSTs the URL verbatim, so
    the token reaches do_POST as part of self.path. Requests missing or
    mismatching the token, or hitting any path other than /reward, get 403
    / 404 — no unauthenticated process can drive sandbox.reward_completion.
    """

    def __init__(
        self,
        dataset: list[dict[str, Any]],
        mode: str = "auto",
        scale: float = 1.0,
        host: str = "127.0.0.1",
        port: int = 0,
        prompt_miss_action: str = "warn",
        monitor: Any = None,
        bench_client: Any = None,
    ):
        self._mode = mode
        self._scale = scale
        # T7/3.1: bench-mode scorer. None + mode="bench" raises at start() —
        # fail-visible before fusion-mlx ever submits a job against us.
        self._bench_scorer = (
            _BenchScorer(bench_client) if mode == "bench" and bench_client else None
        )
        self.bench_degraded = False
        # P2: index by a short hash of the prompt, not the full prompt text.
        # Avoids the memory double-up of storing full prompts here AND in the
        # dataset, and makes the lookup robust to tiny whitespace diffs that
        # would otherwise key-miss into a silent uniform 0.5 reward.
        self._sample_index: dict[str, dict[str, Any]] = {}
        for sample in dataset:
            prompt = sample.get("prompt", "")
            if prompt:
                self._sample_index[_prompt_key(prompt)] = sample
        self._host = host
        self._port = port
        # P2-13 (round-3 audit): prompt-miss handling. "warn" (default) keeps
        # the legacy degrade-to-uniform-0.5 behavior with a loud log. "fail"
        # returns a fail-visible 503 so a prompt/dataset mismatch can't hide
        # behind a constant reward signal — GRPO fails the job rather than
        # train on garbage. "zero" degrades to 0.0 (not 0.5) with a warning.
        self._prompt_miss_action = prompt_miss_action
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint = ""
        self._token = ""
        # P2-4 (round-3 audit): latency circuit breaker. If N consecutive
        # scoring calls exceed the latency budget, the sandbox reward path is
        # almost certainly wedged (e.g. a code-mode subprocess hung) and the
        # GRPO loop is just burning wall-clock waiting on callbacks. Trip to
        # fail-visible 503 so fusion-mlx fails the job fast instead of hanging
        # a multi-hour run on a wedged reward. Override via env.
        self._latency_budget = float(
            os.environ.get("FUSION_REWARD_LATENCY_BUDGET_S", "30")
        )
        self._latency_trip_after = int(
            os.environ.get("FUSION_REWARD_LATENCY_TRIP_AFTER", "3")
        )
        self._breaker = _LatencyBreaker(self._latency_budget, self._latency_trip_after)
        # T7 (capability plan): optional HackingMonitor — every scoring batch
        # feeds the reward-drift / length-inflation / format-gaming detectors.
        self.monitor = monitor
        self._batch_count = 0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def redacted_endpoint(self) -> str:
        # A2 (round-2 audit): the endpoint carries the bearer token in
        # ?token=...; printing/logging the raw endpoint leaks the secret.
        # Callers must use this redacted form for any human-visible output
        # (stdout, log files, run meta). Only the fusion-mlx POST needs the
        # raw endpoint, and only in the job-config field it sends verbatim.
        if not self._endpoint:
            return ""
        return _redact_token(self._endpoint)

    def start(self) -> str:
        mode = self._mode
        scale = self._scale
        index = self._sample_index
        breaker = self._breaker
        miss_action = self._prompt_miss_action
        monitor = self.monitor
        batch_no = [0]  # T7: mutable batch counter for the closure
        scorer = self._bench_scorer
        server_ref = self  # 3.1: bench degradation flag lives on the instance
        if mode == "bench" and scorer is None:
            raise ValueError(
                "RewardServer mode='bench' requires a bench_client "
                "(eval_loop.BenchClient or compatible) — refusing to start "
                "with a silent uniform reward"
            )
        # Generate a fresh unguessable token per server instance; embed it in
        # the URL the caller (fusion-mlx) POSTs back to verbatim.
        token = secrets.token_urlsafe(32)
        self._token = token

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr log
                # A2 (round-2 audit): fmt contains the request line including
                # ?token=...; strip the token before it reaches the log.
                safe = _redact_token(fmt % args) if args else _redact_token(fmt)
                logger.debug("reward-server: %s", safe)

            def _forbid(self, code: int, msg: str) -> None:
                body = json.dumps({"error": msg}).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802 - stdlib HTTP API
                parsed = urlparse(self.path)
                # C1: only the reward path is accepted; everything else 404.
                if parsed.path != _REWARD_PATH:
                    logger.warning("reward-server: rejected path %r", parsed.path)
                    self._forbid(404, "not found")
                    return
                # C1: validate the bearer token from the query string.
                qs = parse_qs(parsed.query)
                got = (qs.get("token", [""])[0] or "").strip()
                if not got or not secrets.compare_digest(got, token):
                    logger.warning("reward-server: rejected unauthenticated request")
                    self._forbid(403, "forbidden")
                    return
                # P2-4: breaker tripped — fail fast (503) instead of queuing
                # behind a wedged scoring path. A GRPO loop waiting on this
                # callback will then surface the failure visibly.
                if breaker.is_open:
                    logger.error("reward-server: latency breaker OPEN — 503")
                    self._forbid(503, "reward latency breaker open")
                    return
                try:
                    _t0 = time.monotonic()
                    length = int(self.headers.get("Content-Length", 0))
                    if length <= 0 or length > _MAX_BODY_BYTES:
                        logger.warning("reward-server: rejected body length=%s", length)
                        self._forbid(400, "invalid content length")
                        return
                    raw = self.rfile.read(length)
                    body = json.loads(raw.decode("utf-8"))
                    prompt = body.get("prompt", "")
                    completions = body.get("completions", [])
                    if not isinstance(completions, list):
                        logger.warning("reward-server: completions not a list")
                        self._forbid(400, "completions must be a list")
                        return
                    if len(completions) > _MAX_COMPLETIONS:
                        logger.warning(
                            "reward-server: completions capped %d -> %d",
                            len(completions),
                            _MAX_COMPLETIONS,
                        )
                        completions = completions[:_MAX_COMPLETIONS]
                    key = _prompt_key(str(prompt))
                    sample = index.get(key)
                    call_scale = scale
                    if sample is None:
                        # P2-13 (round-3 audit): prompt miss is fail-visible.
                        # The default "warn" degrades to uniform 0.5 (legacy,
                        # logged loudly) so a constant reward signal never
                        # looks like learning. "zero" degrades to 0.0. "fail"
                        # refuses — a prompt/dataset mismatch means GRPO would
                        # train on a meaningless constant reward, so fail the
                        # job visibly instead.
                        logger.warning(
                            "reward-server: prompt not in index (key=%s) — "
                            "miss_action=%s",
                            key,
                            miss_action,
                        )
                        if miss_action == "fail":
                            self._forbid(503, "prompt not in dataset index")
                            return
                        sample = {}
                        if miss_action == "zero":
                            # Local copy only — do NOT mutate the captured
                            # `scale` (it would persist across calls).
                            call_scale = 0.0
                    if scorer is not None:
                        # 3.1 bench-as-verifier: score via the bench task API.
                        # Any bench failure degrades the WHOLE batch to the
                        # sandbox reward with a loud warning (fail-visible,
                        # same semantics as eval_loop #75) — never a silent
                        # constant.
                        try:
                            rewards = [
                                scorer.score(sample, c, call_scale)
                                for c in completions
                            ]
                        except Exception as exc:
                            server_ref.bench_degraded = True
                            logger.error(
                                "bench scoring failed (%s) — degrading batch to "
                                "sandbox reward (fail-visible)",
                                exc,
                            )
                            rewards = [
                                reward_completion(sample, c, mode="auto", scale=call_scale)
                                for c in completions
                            ]
                    else:
                        rewards = [
                            reward_completion(sample, c, mode=mode, scale=call_scale)
                            for c in completions
                        ]
                    breaker.record(time.monotonic() - _t0)
                    if monitor is not None:
                        batch_no[0] += 1
                        monitor.observe(batch_no[0], rewards, completions)
                    payload = json.dumps({"rewards": rewards}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    logger.exception("reward-server scoring failed")
                    # A2 (round-2 audit): never leak exception detail to the
                    # caller — a 500 body echoing str(exc) can disclose the
                    # internal stack/state. Return a fixed message.
                    err = json.dumps({"error": "scoring failed"}).encode("utf-8")
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)

        self._server = ThreadingHTTPServer((self._host, self._port), _Handler)
        actual_port = self._server.server_address[1]
        # Token travels in the URL so fusion-mlx's verbatim POST carries it.
        self._endpoint = (
            f"http://{self._host}:{actual_port}{_REWARD_PATH}?token={token}"
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="grpo-reward-server"
        )
        self._thread.start()
        logger.info(
            "reward server started on %s (mode=%s scale=%s samples=%d auth=token)",
            f"http://{self._host}:{actual_port}{_REWARD_PATH}",
            mode,
            scale,
            len(index),
        )
        return self._endpoint

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._thread = None
        self._token = ""
        self._endpoint = ""
        logger.info("reward server stopped")


# Short stable hash of a prompt for the sample index (P2).
def _prompt_key(prompt: str) -> str:
    return hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:16]


# A2 (round-2 audit): bounds to prevent unbounded reads / huge payloads.
_MAX_BODY_BYTES = 16 * 1024 * 1024
_MAX_COMPLETIONS = 64


class _LatencyBreaker:
    """Consecutive-slow circuit breaker (P2-4).

    Counts scoring calls that exceeded the latency budget in a row; once that
    reaches trip_after, the breaker opens and stays open until reset (a new
    job). A single fast call resets the streak. Thread-safe — the reward
    server is ThreadingHTTPServer so multiple do_POST run concurrently.
    """

    def __init__(self, budget_s: float, trip_after: int):
        self._budget = budget_s
        self._trip_after = max(1, trip_after)
        self._streak = 0
        self._open = False
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._open

    def record(self, elapsed_s: float) -> None:
        with self._lock:
            if elapsed_s > self._budget:
                self._streak += 1
                if self._streak >= self._trip_after and not self._open:
                    self._open = True
                    logger.error(
                        "reward-server: latency breaker TRIPPED — %d consecutive "
                        "calls > %.1fs budget; failing 503 until reset",
                        self._streak,
                        self._budget,
                    )
            else:
                self._streak = 0


def _redact_token(url_or_str: str) -> str:
    # A2 (round-2 audit): strip ?token=... / &token=... from any string before
    # it reaches stdout, log files, or run meta. Replaces the secret value
    # with the literal "<redacted>".
    import re

    return re.sub(r"([?&]token=)[^&\s]*", r"\1<redacted>", url_or_str)
