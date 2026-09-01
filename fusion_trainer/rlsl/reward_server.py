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
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from .sandbox import reward_completion

logger = logging.getLogger(__name__)

_REWARD_PATH = "/reward"


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
    ):
        self._mode = mode
        self._scale = scale
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
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint = ""
        self._token = ""

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> str:
        mode = self._mode
        scale = self._scale
        index = self._sample_index
        # Generate a fresh unguessable token per server instance; embed it in
        # the URL the caller (fusion-mlx) POSTs back to verbatim.
        token = secrets.token_urlsafe(32)
        self._token = token

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr log
                logger.debug("reward-server: " + fmt, *args)

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
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length else b"{}"
                    body = json.loads(raw.decode("utf-8"))
                    prompt = body.get("prompt", "")
                    completions = body.get("completions", [])
                    key = _prompt_key(str(prompt))
                    sample = index.get(key)
                    if sample is None:
                        # P2: loud warning on prompt miss — a silent fallback
                        # to uniform 0.5 makes GRPO look like it's learning
                        # when the reward signal is actually constant.
                        logger.warning(
                            "reward-server: prompt not in index (key=%s) — "
                            "degrading to uniform 0.5 reward. Check that the "
                            "GRPO prompts match the dataset prompts exactly.",
                            key,
                        )
                        sample = {}
                    rewards = [
                        reward_completion(sample, c, mode=mode, scale=scale)
                        for c in completions
                    ]
                    payload = json.dumps({"rewards": rewards}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except Exception as exc:
                    logger.exception("reward-server scoring failed")
                    err = json.dumps({"error": str(exc)}).encode("utf-8")
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
        logger.info("reward server stopped")


# Short stable hash of a prompt for the sample index (P2).
def _prompt_key(prompt: str) -> str:
    return hashlib.sha1(str(prompt).encode("utf-8")).hexdigest()[:16]
