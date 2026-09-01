"""RM-backed reward server for RLSL Phase 2 (fusion-mlx #431).

RLSL Phase 2 runs a real policy-gradient GRPO loop (fusion-mlx #363). GRPO
scores completions by calling back an HTTP endpoint:
    POST {prompt, completions} -> {rewards}
This module hosts a tiny stdlib HTTP server that receives that callback and
delegates scoring to fusion-mlx's /admin/api/fine-tune/reward/score endpoint
(#431), which loads the Phase-1 reward-model adapter and returns scalar
rewards. This closes the loop:
    Phase 1 RM (#424) -> reward/score (#431) -> Phase 2 GRPO (#363).

Importers/callers: fusion_trainer.rlsl.pipeline (RLSLEngine._run_rlsl Phase 2)
Affected API: RMRewardServer(client, model_id, adapter_name).start() -> str, .stop()
Data schemas:
  request  {"prompt": str, "completions": [str, ...]}
  response {"rewards": [float, ...]}
"""

from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

_REWARD_PATH = "/reward"
# R13: circuit-breaker thresholds. score_reward runs inside the GRPO reward
# callback; if fusion-mlx is transiently down, every callback used to block
# to the 30s client timeout and fusion-mlx's GRPO loop would fail or stall.
# After _CB_FAIL_THRESHOLD consecutive failures the breaker opens: subsequent
# callbacks short-circuit to a zero-reward baseline for _CB_COOLDOWN seconds
# (fail-visible: logged loudly) instead of blocking, then half-opens to probe.
_CB_FAIL_THRESHOLD = 3
_CB_COOLDOWN = 30.0


class RMRewardServer:
    """Local reward endpoint that scores via a trained RM adapter.

    Receives GRPO's {prompt, completions} callback and forwards it to
    fusion-mlx reward/score (#431), returning the RM's scalar rewards. Unlike
    the sandbox RewardServer, no reference answer/tests are needed — the RM
    itself is the reward signal.

    Security (audit C1): one-time bearer token generated at start(), embedded
    in the endpoint URL as ?token=... (fusion-mlx POSTs the URL verbatim). do_POST
    rejects any path != /reward (404) and any request missing/mismatching the
    token (403) — no unauthenticated process can drive score_reward.
    """

    def __init__(
        self,
        client: Any,
        model_id: str,
        adapter_name: str,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self._client = client
        self._model_id = model_id
        self._adapter_name = adapter_name
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint = ""
        self._token = ""
        # R13: circuit-breaker state for score_reward (shared across handler
        # threads via this lock).
        self._cb_lock = threading.Lock()
        self._cb_failures = 0
        self._cb_open_until = 0.0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def redacted_endpoint(self) -> str:
        # A2 (round-2 audit): endpoint carries the bearer token; callers must
        # use this redacted form for stdout/log/run-meta. See _redact_token.
        if not self._endpoint:
            return ""
        return _redact_token(self._endpoint)

    def start(self) -> str:
        client = self._client
        model_id = self._model_id
        adapter_name = self._adapter_name
        token = secrets.token_urlsafe(32)
        self._token = token
        cb_score = self._score_with_breaker

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr log
                # A2 (round-2 audit): fmt holds the request line with ?token=
                safe = _redact_token(fmt % args) if args else _redact_token(fmt)
                logger.debug("rm-reward-server: %s", safe)

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
                    logger.warning("rm-reward-server: rejected path %r", parsed.path)
                    self._forbid(404, "not found")
                    return
                # C1: validate the bearer token from the query string.
                qs = parse_qs(parsed.query)
                got = (qs.get("token", [""])[0] or "").strip()
                if not got or not secrets.compare_digest(got, token):
                    logger.warning("rm-reward-server: rejected unauthenticated request")
                    self._forbid(403, "forbidden")
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length <= 0 or length > _MAX_BODY_BYTES:
                        logger.warning(
                            "rm-reward-server: rejected body length=%s", length
                        )
                        self._forbid(400, "invalid content length")
                        return
                    raw = self.rfile.read(length)
                    body = json.loads(raw.decode("utf-8"))
                    prompt = body.get("prompt", "")
                    completions = body.get("completions", [])
                    if not isinstance(completions, list):
                        completions = []
                    if len(completions) > _MAX_COMPLETIONS:
                        logger.warning(
                            "rm-reward-server: completions capped %d -> %d",
                            len(completions),
                            _MAX_COMPLETIONS,
                        )
                        completions = completions[:_MAX_COMPLETIONS]
                    logger.info(
                        "rm-reward-server callback: model=%s adapter=%s "
                        "prompt_len=%d n_completions=%d",
                        model_id,
                        adapter_name,
                        len(str(prompt)),
                        len(completions),
                    )
                    result = cb_score(
                        client=client,
                        model_id=model_id,
                        adapter_name=adapter_name,
                        prompt=str(prompt),
                        completions=[str(c) for c in completions],
                    )
                    rewards = result.get("rewards", [])
                    logger.info("rm-reward-server scored: rewards=%s", rewards)
                    payload = json.dumps({"rewards": rewards}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    logger.exception("rm-reward-server scoring failed")
                    # A2 (round-2 audit): never leak exception detail to caller.
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
            target=self._server.serve_forever,
            daemon=True,
            name="rlsl-rm-reward-server",
        )
        self._thread.start()
        logger.info(
            "rm-reward server started on %s (model=%s adapter=%s auth=token)",
            f"http://{self._host}:{actual_port}{_REWARD_PATH}",
            model_id,
            adapter_name,
        )
        return self._endpoint

    def _score_with_breaker(
        self,
        *,
        client: Any,
        model_id: str,
        adapter_name: str,
        prompt: str,
        completions: list[str],
    ) -> dict[str, Any]:
        """R13: wrap score_reward in a circuit breaker so a *sustained*
        fusion-mlx outage degrades (not deadlocks) the GRPO reward callback.

        Closed (failures < threshold): forward to score_reward. On failure,
        increment the counter and RE--RAISE — the handler returns 500 so the
        error is fail-visible (a single blip must not be silently swallowed
        into a zero-reward "success"). Only once _CB_FAIL_THRESHOLD consecutive
        failures trip the breaker do we OPEN it.

        Open: short-circuit to a uniform zero-reward baseline for _CB_COOLDOWN
        seconds (logged loudly — fail-visible), keeping the GRPO loop
        responsive (degraded, not dead) while fusion-mlx recovers, then
        half-open to probe one call. A successful probe closes the breaker.
        """
        now = time.monotonic()
        with self._cb_lock:
            if self._cb_open_until and now < self._cb_open_until:
                logger.warning(
                    "rm-reward-server: breaker OPEN, returning zero-reward "
                    "baseline (fusion-mlx score unavailable); GRPO reward "
                    "signal DEGRADED. n_completions=%d",
                    len(completions),
                )
                return {"rewards": [0.0] * len(completions)}
        try:
            result = client.score_reward(
                model_id=model_id,
                adapter_name=adapter_name,
                prompt=prompt,
                completions=completions,
            )
        except Exception as exc:
            tripped = False
            with self._cb_lock:
                self._cb_failures += 1
                if self._cb_failures >= _CB_FAIL_THRESHOLD:
                    self._cb_open_until = time.monotonic() + _CB_COOLDOWN
                    tripped = True
                    logger.warning(
                        "rm-reward-server: breaker OPENED after %d consecutive "
                        "score failures (cooldown=%.0fs). error=%s",
                        self._cb_failures,
                        _CB_COOLDOWN,
                        exc,
                    )
            if tripped:
                # Sustained outage — degrade to baseline so GRPO keeps moving.
                return {"rewards": [0.0] * len(completions)}
            # Closed + below threshold: re-raise so the handler returns 500
            # (fail-visible). A single transient error must NOT silently read
            # as a zero-reward success — that corrupts the reward signal.
            raise
        with self._cb_lock:
            if self._cb_failures or self._cb_open_until:
                logger.info(
                    "rm-reward-server: breaker CLOSED (score recovered after "
                    "%d failures)",
                    self._cb_failures,
                )
            self._cb_failures = 0
            self._cb_open_until = 0.0
        return result

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
        logger.info("rm-reward server stopped")


# A2 (round-2 audit): bounds + token redaction (shared with reward_server).
_MAX_BODY_BYTES = 16 * 1024 * 1024
_MAX_COMPLETIONS = 64


def _redact_token(url_or_str: str) -> str:
    import re

    return re.sub(r"([?&]token=)[^&\s]*", r"\1<redacted>", url_or_str)
