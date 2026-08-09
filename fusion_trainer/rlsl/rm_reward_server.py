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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

logger = logging.getLogger(__name__)


class RMRewardServer:
    """Local reward endpoint that scores via a trained RM adapter.

    Receives GRPO's {prompt, completions} callback and forwards it to
    fusion-mlx reward/score (#431), returning the RM's scalar rewards. Unlike
    the sandbox RewardServer, no reference answer/tests are needed — the RM
    itself is the reward signal.
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

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> str:
        client = self._client
        model_id = self._model_id
        adapter_name = self._adapter_name

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # silence default stderr log
                logger.debug("rm-reward-server: " + fmt, *args)

            def do_POST(self):  # noqa: N802 - stdlib HTTP API
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(length) if length else b"{}"
                    body = json.loads(raw.decode("utf-8"))
                    prompt = body.get("prompt", "")
                    completions = body.get("completions", [])
                    if not isinstance(completions, list):
                        completions = []
                    logger.info(
                        "rm-reward-server callback: model=%s adapter=%s "
                        "prompt_len=%d n_completions=%d",
                        model_id,
                        adapter_name,
                        len(str(prompt)),
                        len(completions),
                    )
                    result = client.score_reward(
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
            target=self._server.serve_forever,
            daemon=True,
            name="rlsl-rm-reward-server",
        )
        self._thread.start()
        logger.info(
            "rm-reward server started on %s (model=%s adapter=%s)",
            self._endpoint,
            model_id,
            adapter_name,
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
        logger.info("rm-reward server stopped")
