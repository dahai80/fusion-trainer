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

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .sandbox import reward_completion

logger = logging.getLogger(__name__)


class RewardServer:
    """Thread-safe local reward endpoint for fusion-mlx GRPO callback.

    Builds a prompt->sample index from the dataset so the sandbox reward
    (which needs the reference answer/tests) can score completions. If a
    prompt is unknown, falls back to the auto reward with an empty sample
    (uniform 0.5 * scale).
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
        self._sample_index: dict[str, dict[str, Any]] = {}
        for sample in dataset:
            prompt = sample.get("prompt", "")
            if prompt:
                self._sample_index[prompt] = sample
        self._host = host
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._endpoint = ""

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> str:
        mode = self._mode
        scale = self._scale
        index = self._sample_index

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr log
                logger.debug("reward-server: " + fmt, *args)

            def do_POST(self):  # noqa: N802 - stdlib HTTP API
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length else b"{}"
                    body = json.loads(raw.decode("utf-8"))
                    prompt = body.get("prompt", "")
                    completions = body.get("completions", [])
                    sample = index.get(prompt, {})
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
        self._endpoint = f"http://{self._host}:{actual_port}/reward"
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True, name="grpo-reward-server"
        )
        self._thread.start()
        logger.info(
            "reward server started on %s (mode=%s scale=%s samples=%d)",
            self._endpoint,
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
        logger.info("reward server stopped")
