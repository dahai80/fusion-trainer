"""HTTP client for fusion-model-hub model registry (issue #49).

Fusion-Trainer publishes a trained LoRA adapter to fusion-model-hub so the
gateway can index it (GET /api/v1/models?model_type=lora). This client is the
ONLY trainer-side dependency on the hub HTTP API — it never imports hub code,
only calls POST /api/v1/models over HTTP.

Importers/callers:
  - fusion_trainer.run_manager._run_thread (GUI path, after completed run)
  - fusion_trainer.cli sft/rlsl handlers (CLI path, after train/run returns)
  - fusion_trainer.hub_client.publish_adapter_if_enabled (shared helper)

Affected API:
  POST  /api/v1/models        — register a model (model_type=lora, base_model_id)

Data schemas (request body, ModelCreate from fusion-model-hub):
  name: str (3-64), model_type: "lora", base_model_id: str,
  description: str, tags: list[{key,value}], ...
Response: 201 created (model dict), 409 name exists, 404 base_model_id missing.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

MODELS_PATH = "/api/v1/models"


class HubPublishError(RuntimeError):
    pass


class ModelHubClient:
    """Thin HTTP wrapper around fusion-model-hub's /api/v1/models registry."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: float = 15.0,
    ):
        self._base = base_url.rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        logger.info("ModelHubClient initialized: base=%s auth=%s", self._base, bool(api_key))

    def publish_adapter(
        self,
        name: str,
        base_model: str,
        description: str = "",
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Register a LoRA adapter as a model_type=lora entry in the hub.

        Raises HubPublishError on non-2xx or transport failure so callers can
        decide whether to warn (default) or fail hard.
        """
        if not name or len(name) < 3:
            raise HubPublishError(f"adapter name too short (min 3): {name!r}")
        if not base_model:
            raise HubPublishError("base_model is required for model_type=lora publish")
        payload: dict[str, Any] = {
            "name": name,
            "model_type": "lora",
            "base_model_id": base_model,
            "description": description,
        }
        if tags:
            payload["tags"] = tags
        url = f"{self._base}{MODELS_PATH}"
        logger.info("publish_adapter: POST %s name=%s base=%s", url, name, base_model)
        try:
            resp = httpx.post(url, json=payload, headers=self._headers, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise HubPublishError(f"hub POST transport error: {exc}") from exc
        if resp.status_code == 201:
            body = resp.json() if resp.content else {}
            logger.info("publish_adapter: created model_id=%s name=%s", body.get("id"), name)
            return {"published": True, "model": body, "status_code": 201}
        detail = ""
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HubPublishError(
            f"hub POST {url} failed: {resp.status_code} {detail}"
        )


def _extract_adapter_name(result: dict[str, Any]) -> str:
    # SFT + RLSL dpo/orpo/grpo return adapter_name at top level.
    name = result.get("adapter_name", "")
    if name:
        return name
    # RLSL rlsl method nests it under phase2.
    phase2 = result.get("phase2")
    if isinstance(phase2, dict):
        return phase2.get("adapter_name", "")
    return ""


def publish_adapter_if_enabled(
    config: Any,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Publish the adapter from a completed run result, if enabled.

    No-op (returns {"published": False, "reason": "disabled"}) when
    config.publish_adapter is False or the run did not complete / has no
    adapter. On failure, logs a loud warning and returns
    {"published": False, "error": ...} — NEVER raises, so the training
    result is never blocked by a hub outage (fail-visible, non-blocking).
    """
    if not getattr(config, "publish_adapter", False):
        return {"published": False, "reason": "disabled"}
    if result.get("status") != "completed":
        return {"published": False, "reason": f"status={result.get('status', '')}"}
    name = _extract_adapter_name(result)
    if not name:
        return {"published": False, "reason": "no adapter_name in result"}
    base_model = getattr(config.sft, "model", "") or ""
    if not base_model:
        logger.warning("publish_adapter_if_enabled: empty base_model, skipping publish")
        return {"published": False, "reason": "empty base_model"}
    hub_url = getattr(config, "hub_url", "") or "http://localhost:11432"
    api_key = getattr(config, "hub_api_key", "") or ""
    desc = f"LoRA adapter from fusion-trainer run ({result.get('method', '')})"
    client = ModelHubClient(base_url=hub_url, api_key=api_key)
    try:
        return client.publish_adapter(name=name, base_model=base_model, description=desc)
    except HubPublishError as exc:
        logger.warning(
            "publish_adapter FAILED (non-blocking): name=%s base=%s err=%s",
            name,
            base_model,
            exc,
        )
        return {"published": False, "error": str(exc), "adapter_name": name}
