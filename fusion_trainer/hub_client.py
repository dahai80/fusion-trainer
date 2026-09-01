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

from .config import URLGuardError, _default_hub_url, validate_service_url

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
        # S6: validate before storing so a tainted config URL can't exfil the
        # Bearer key to an attacker server. Raises URLGuardError on reject.
        self._base = validate_service_url(base_url, field_name="hub_url").rstrip("/")
        self._headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"
        self._timeout = timeout
        logger.info(
            "ModelHubClient initialized: base=%s auth=%s", self._base, bool(api_key)
        )

    def list_models(
        self,
        model_type: str = "",
        keyword: str = "",
    ) -> list[dict[str, Any]]:
        """List hub models, optionally filtered by model_type / keyword.

        The hub supports keyword (fuzzy) + model_type filters but NOT an exact
        name= filter (fusion-models-hub#51). Callers needing an exact match
        must filter client-side over the returned list.
        """
        params: dict[str, str] = {}
        if model_type:
            params["model_type"] = model_type
        if keyword:
            params["keyword"] = keyword
        url = f"{self._base}{MODELS_PATH}"
        try:
            resp = httpx.get(
                url, params=params or None, headers=self._headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise HubPublishError(f"hub GET transport error: {exc}") from exc
        if resp.status_code >= 400:
            raise HubPublishError(
                f"hub GET {url} failed: {resp.status_code} {resp.text}"
            )
        data = resp.json()
        if isinstance(data, list):
            return data
        # R17: hub returns {"data":[...]} (and some versions {"items":[...]}).
        # Treating the envelope as an empty list made the L4 409-idempotency
        # check never find an existing lora → re-publish raised instead of
        # returning idempotent. Unwrap the known envelope keys.
        if isinstance(data, dict):
            for key in ("data", "items", "models", "results"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
        return []

    def _find_existing_lora(self, name: str, base_model: str) -> dict[str, Any] | None:
        """Best-effort exact-name + base_model_id match among hub loras (L4).

        Returns the matching model dict or None. Used to make a 409 (name
        exists) idempotent: if the existing lora with this name points at the
        same base model, the publish already succeeded previously.
        """
        try:
            candidates = self.list_models(model_type="lora", keyword=name)
        except HubPublishError as exc:
            logger.warning("publish idempotency check failed (non-fatal): %s", exc)
            return None
        for m in candidates:
            if (
                str(m.get("name", "")) == name
                and str(m.get("base_model_id", "")) == base_model
            ):
                return m
        return None

    def publish_adapter(
        self,
        name: str,
        base_model: str,
        description: str = "",
        tags: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Register a LoRA adapter as a model_type=lora entry in the hub.

        Idempotent on 409 (audit L4): if a lora with the same name AND
        base_model_id already exists, the publish is treated as already-done
        and returns that model — retries of the same run no longer fail
        forever. A 409 whose existing entry has a DIFFERENT base model is a
        real name collision and still raises HubPublishError. Other non-2xx /
        transport failures raise so callers can warn (non-blocking) or fail.
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
            resp = httpx.post(
                url, json=payload, headers=self._headers, timeout=self._timeout
            )
        except httpx.HTTPError as exc:
            raise HubPublishError(f"hub POST transport error: {exc}") from exc
        if resp.status_code == 201:
            # R18: an empty 201 body yields a dangling hub entry (model:{}) —
            # treat a missing id as a publish that did not land cleanly and
            # surface it loudly rather than reporting success.
            body: dict[str, Any] = {}
            if resp.content:
                try:
                    parsed = resp.json()
                    body = parsed if isinstance(parsed, dict) else {}
                except ValueError:
                    logger.warning(
                        "publish_adapter: 201 with non-JSON body: %s",
                        resp.text[:200],
                    )
            if not body.get("id"):
                logger.warning(
                    "publish_adapter: 201 returned no model id for name=%s — "
                    "hub entry may be dangling; not reporting success",
                    name,
                )
                raise HubPublishError(
                    f"hub POST {url} returned 201 but no model id (empty body) "
                    f"for name={name}"
                )
            logger.info(
                "publish_adapter: created model_id=%s name=%s", body.get("id"), name
            )
            return {"published": True, "model": body, "status_code": 201}
        if resp.status_code == 409:
            # L4 idempotency: is the collision our own prior publish?
            existing = self._find_existing_lora(name, base_model)
            if existing is not None:
                logger.info(
                    "publish_adapter: name=%s already published (idempotent) "
                    "model_id=%s",
                    name,
                    existing.get("id"),
                )
                return {
                    "published": True,
                    "model": existing,
                    "status_code": 409,
                    "idempotent": True,
                }
            logger.warning(
                "publish_adapter: 409 name=%s exists with a different base_model "
                "(got collision, not idempotent)",
                name,
            )
        detail = ""
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise HubPublishError(f"hub POST {url} failed: {resp.status_code} {detail}")


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
    hub_url = getattr(config, "hub_url", "") or _default_hub_url()
    api_key = getattr(config, "hub_api_key", "") or ""
    desc = f"LoRA adapter from fusion-trainer run ({result.get('method', '')})"
    try:
        client = ModelHubClient(base_url=hub_url, api_key=api_key)
    except URLGuardError as exc:
        logger.warning(
            "publish_adapter REJECTED hub_url (non-blocking): name=%s err=%s",
            name,
            exc,
        )
        return {"published": False, "error": str(exc), "adapter_name": name}
    try:
        return client.publish_adapter(
            name=name, base_model=base_model, description=desc
        )
    except HubPublishError as exc:
        logger.warning(
            "publish_adapter FAILED (non-blocking): name=%s base=%s err=%s",
            name,
            base_model,
            exc,
        )
        return {"published": False, "error": str(exc), "adapter_name": name}
