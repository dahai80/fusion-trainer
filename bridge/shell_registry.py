"""Shell module registration for the trainer bridge (issue #72).

Registers bridge/manifest.json with the fusion-gateway shell registry
(POST /api/v1/shell/manifests, fg-key auth) and keeps it alive with
periodic heartbeats, so the Trainer console shows up as a first-class
sidebar entry in the fusion-web-shell host (/shell/fusion-trainer/).

K1/K2 line (fusion-gateway#261/#262): the registration body carries NO
api_key, so the gateway issues an fm- module service key and returns the
plaintext exactly once (field `raw_key`). The key is persisted to
<cache>/web_module_key (0600) and used for heartbeat auth; inbound
requests proxied by the gateway carry the same fm- key as Bearer, which
server._auth accepts alongside the static FUSION_TRAINER_BRIDGE_KEY.

Env-gated — requires BOTH:
  FUSION_SHELL_GATEWAY_URL   gateway base, e.g. http://127.0.0.1:11432
  FUSION_ADMIN_KEY           a gateway key authorized for /api/v1/shell/*
                             (resolved via fusion_core.config.admin_key, #80;
                             FUSION_PROFILE=prod missing key disables
                             registration with a loud warning — never fatal)

Optional:
  FUSION_SHELL_UPSTREAM      proxy target override (default
                             http://127.0.0.1:8790, the bridge itself)
  FUSION_SHELL_MANIFEST      manifest path override (default
                             bridge/manifest.json next to this file)

All failures are logged and retried on the next tick — registration must
never take the bridge down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from fusion_core.config import admin_key as fc_admin_key

logger = logging.getLogger(__name__)

GATEWAY_URL_ENV = "FUSION_SHELL_GATEWAY_URL"
# #80: admin-key resolution lives in fusion_core.config (FUSION_ADMIN_KEY);
# ADMIN_KEY_ENV kept only as the documented env name for error messages.
ADMIN_KEY_ENV = "FUSION_ADMIN_KEY"
MANIFEST_PATH_ENV = "FUSION_SHELL_MANIFEST"
UPSTREAM_ENV = "FUSION_SHELL_UPSTREAM"
MODULE_KEY_ENV = "FUSION_TRAINER_BRIDGE_KEY"
CACHE_ENV = "FUSION_TRAINER_CACHE"
DEFAULT_UPSTREAM = "http://127.0.0.1:8790"
REGISTER_INTERVAL_SECONDS = 15.0
HTTP_TIMEOUT_SECONDS = 5.0


def module_key_file() -> Path:
    cache = Path(os.environ.get(CACHE_ENV, str(Path.home() / ".fusion-trainer-bridge")))
    return cache / "web_module_key"


def load_module_key() -> str:
    try:
        return module_key_file().read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def save_module_key(key: str) -> bool:
    key = (key or "").strip()
    if not key:
        return False
    try:
        path = module_key_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key + "\n", encoding="utf-8")
        path.chmod(0o600)
        logger.info("gateway-issued module key persisted: %s (%s)", path, _mask(key))
        return True
    except OSError as e:
        logger.warning("module key file unwritable: %s", e)
        return False


def _mask(key: str) -> str:
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


def accepted_bridge_keys() -> list[str]:
    """Inbound accept set: static bridge key + gateway-issued fm- key.

    The gateway /shell proxy strips the browser's Authorization and
    injects the fm- key it holds for the module; direct loopback access
    keeps using the static FUSION_TRAINER_BRIDGE_KEY. During rotation
    grace either may arrive, so both are accepted.
    """
    keys: list[str] = []
    env_key = os.environ.get(MODULE_KEY_ENV, "").strip()
    if env_key:
        keys.append(env_key)
    # config-conventions §5.1: the gateway client key is the contract
    # credential business processes hold — the console sends it as Bearer.
    try:
        from fusion_core.config import client_key

        ck = client_key().strip()
        if ck and ck not in keys:
            keys.append(ck)
    except ImportError:  # pragma: no cover
        pass
    issued = load_module_key()
    if issued and issued not in keys:
        keys.append(issued)
    return keys


def _extract_issued_key(payload: Any) -> str:
    """Gateway K1 issuance returns the plaintext exactly once in `raw_key`
    (writeShellKeyResponse, fusion-gateway internal/server/shell.go). Older
    names are accepted defensively; a retained-key 200 carries no key."""
    if not isinstance(payload, dict):
        return ""
    for field in ("raw_key", "api_key", "issued_key", "key", "module_key"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = payload.get("data")
    if isinstance(nested, dict):
        for field in ("raw_key", "api_key", "issued_key", "key", "module_key"):
            value = nested.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _default_manifest_path() -> Path:
    return Path(__file__).resolve().parent / "manifest.json"


def resolve_upstream(manifest: dict[str, Any]) -> str:
    upstream = (
        os.environ.get(UPSTREAM_ENV, "").strip()
        or str(manifest.get("upstream") or "").strip()
        or DEFAULT_UPSTREAM
    )
    parsed = urlparse(upstream)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"shell upstream must be an absolute http(s) URL: {upstream}")
    return upstream.rstrip("/")


def build_registration_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """No api_key in the body — that is the K1 trigger: the gateway issues
    the fm- key instead of accepting a module-supplied credential."""
    payload = dict(manifest)
    payload["upstream"] = resolve_upstream(manifest)
    payload.pop("api_key", None)
    return payload


def _gateway_admin_key() -> str:
    # #80: single-point resolution via fusion_core.config.admin_key
    # (env FUSION_ADMIN_KEY; prod-missing raises, dev-missing warns + "").
    # A prod raise here would kill the bridge, so it degrades to a loud
    # warning + disabled registration — registration is never fatal.
    try:
        return fc_admin_key().strip()
    except RuntimeError as e:
        logger.warning("shell registration disabled: %s", e)
        return ""


def shell_registration_enabled() -> bool:
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip()
    return bool(gateway and _gateway_admin_key())


def load_manifest() -> dict[str, Any]:
    path = Path(
        os.environ.get(MANIFEST_PATH_ENV, "").strip() or _default_manifest_path()
    )
    if not path.exists():
        raise FileNotFoundError(f"shell manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("id"):
        raise ValueError(f"shell manifest missing id: {path}")
    logger.info("shell manifest loaded from %s (id=%s)", path, manifest["id"])
    return manifest


def _safe_json(resp: Any) -> Any:
    try:
        return resp.json()
    except Exception:
        return None


async def _registry_loop_body(
    client: Any,
    manifest: dict[str, Any],
    interval: float,
    max_ticks: int | None = None,
) -> None:
    module_id = str(manifest["id"])
    gateway = os.environ.get(GATEWAY_URL_ENV, "").strip().rstrip("/")
    admin_key = _gateway_admin_key()
    admin_headers = {
        "Authorization": f"Bearer {admin_key}",
        "Content-Type": "application/json",
    }
    try:
        payload = build_registration_payload(manifest)
    except Exception as e:
        logger.error("shell registration disabled (manifest unusable): %s", e)
        return
    logger.info(
        "shell manifest upstream=%s api_key=unset (K1 gateway-issued)",
        payload["upstream"],
    )
    registered = False
    ticks = 0
    while max_ticks is None or ticks < max_ticks:
        ticks += 1
        try:
            if not registered:
                resp = await client.post(
                    f"{gateway}/api/v1/shell/manifests",
                    json=payload,
                    headers=admin_headers,
                )
                if resp.status_code == 200:
                    registered = True
                    logger.info("shell manifest registered: %s", module_id)
                    issued = _extract_issued_key(_safe_json(resp))
                    if issued and issued != load_module_key():
                        save_module_key(issued)
                    elif issued:
                        logger.info(
                            "gateway-issued module key unchanged, not rewriting file"
                        )
                    else:
                        logger.info(
                            "registration retained existing service key — plaintext "
                            "shown only at issuance (rotate via gateway admin if lost)"
                        )
                else:
                    logger.warning(
                        "shell manifest register failed: %d %s",
                        resp.status_code,
                        resp.text[:200],
                    )
            else:
                module_key = load_module_key()
                beat_key = (
                    module_key
                    or os.environ.get(MODULE_KEY_ENV, "").strip()
                    or admin_key
                )
                headers = {
                    "Authorization": f"Bearer {beat_key}",
                    "Content-Type": "application/json",
                }
                resp = await client.post(
                    f"{gateway}/api/v1/shell/manifests/{module_id}/heartbeat",
                    headers=headers,
                )
                if resp.status_code == 404:
                    # Gateway restarted and dropped dynamic manifests.
                    logger.info("shell heartbeat 404, re-registering %s", module_id)
                    registered = False
                elif resp.status_code in (401, 403) and module_key:
                    # Issued key rotated/revoked — re-register via admin key
                    # to pick up the fresh issued key.
                    logger.warning(
                        "shell heartbeat %d (issued key invalid), re-registering %s",
                        resp.status_code,
                        module_id,
                    )
                    registered = False
                elif resp.status_code != 200:
                    logger.warning(
                        "shell heartbeat failed: %d %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except asyncio.CancelledError:
            logger.info("shell registry loop cancelled (module=%s)", module_id)
            raise
        except Exception as e:
            logger.warning("shell registry loop error (will retry): %s", e)
        if max_ticks is None or ticks < max_ticks:
            await asyncio.sleep(interval)


async def shell_registry_loop() -> None:
    """Register once, heartbeat forever; re-register after gateway restarts.

    Runs as a daemon background task from the FastAPI lifespan. All
    failures are logged and retried on the next tick.
    """
    try:
        manifest = load_manifest()
    except Exception as e:
        logger.error("shell registration disabled (manifest unusable): %s", e)
        return
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        await _registry_loop_body(client, manifest, REGISTER_INTERVAL_SECONDS)
