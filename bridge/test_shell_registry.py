"""Shell module registration tests (issue #72) — fully mocked HTTP.

Run: .venv-bridge/bin/python -m pytest bridge/test_shell_registry.py -q
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FUSION_TRAINER_BRIDGE_KEY", "test-bridge-key")
os.environ.setdefault("FUSION_TRAINER_CACHE", "/tmp/f-trainer-shell-test-cache")

import bridge.server as bs  # noqa: E402
from bridge import shell_registry as sr  # noqa: E402

MANIFEST = {
    "id": "fusion-trainer",
    "title": "训练控制台",
    "path": "/trainer",
    "entry": "/shell/fusion-trainer/",
    "devices": ["desktop"],
    "category": "platform",
    "pages": [{"route": "/", "title": "训练控制台"}],
    "upstream": "http://127.0.0.1:8790",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
    for var in (
        "FUSION_SHELL_GATEWAY_URL",
        "FUSION_ADMIN_KEY",
        "FUSION_PROFILE",
        "FUSION_SHELL_UPSTREAM",
        "FUSION_SHELL_MANIFEST",
        # accepted_bridge_keys() folds fusion_core.config.client_key() into
        # the accept set; a developer shell with FUSION_CLIENT_KEY exported
        # must not leak into these tests.
        "FUSION_CLIENT_KEY",
        "FUSION_MLX_API_KEY",
        "MLX_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeClient:
    """Scripted (method, url) -> _Resp responder."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        return self.script.pop(0)


def _issue_resp(key="fm-fusion-trainer-abcd"):
    return _Resp(200, {"code": "ok", "key_id": "k1", "raw_key": key})


def _register_call(client):
    reg = [c for c in client.calls if c[0].endswith("/api/v1/shell/manifests")]
    assert reg, "no registration call"
    return reg[0]


# ── payload / key handling ───────────────────────────────────────────────


def test_registration_payload_has_no_api_key():
    payload = sr.build_registration_payload(MANIFEST)
    assert payload["upstream"] == "http://127.0.0.1:8790"
    assert "api_key" not in payload  # K1 trigger: gateway issues fm- key


def test_registration_payload_env_upstream_override(clean_env, monkeypatch):
    monkeypatch.setenv("FUSION_SHELL_UPSTREAM", "http://10.0.0.5:9000/")
    payload = sr.build_registration_payload(MANIFEST)
    assert payload["upstream"] == "http://10.0.0.5:9000"


def test_registration_payload_rejects_relative_upstream():
    with pytest.raises(ValueError):
        sr.build_registration_payload({**MANIFEST, "upstream": "localhost:8790"})


def test_extract_issued_key_prefers_raw_key():
    assert sr._extract_issued_key({"raw_key": "fm-x-1", "api_key": "other"}) == "fm-x-1"


def test_extract_issued_key_empty_on_retained():
    assert sr._extract_issued_key({"code": "ok", "key_id": "k1"}) == ""
    assert sr._extract_issued_key(None) == ""


def test_module_key_roundtrip_0600(clean_env):
    assert sr.load_module_key() == ""
    assert sr.save_module_key("fm-fusion-trainer-secret")
    assert sr.load_module_key() == "fm-fusion-trainer-secret"
    assert (sr.module_key_file().stat().st_mode & 0o777) == 0o600


def test_accepted_bridge_keys_dedup(clean_env, monkeypatch):
    monkeypatch.setenv("FUSION_TRAINER_BRIDGE_KEY", "static-key")
    assert sr.accepted_bridge_keys() == ["static-key"]
    sr.save_module_key("fm-fusion-trainer-issued")
    assert sr.accepted_bridge_keys() == ["static-key", "fm-fusion-trainer-issued"]
    monkeypatch.setenv("FUSION_TRAINER_BRIDGE_KEY", "fm-fusion-trainer-issued")
    assert sr.accepted_bridge_keys() == ["fm-fusion-trainer-issued"]


def test_registration_enabled_requires_both_env(clean_env, monkeypatch):
    assert not sr.shell_registration_enabled()
    monkeypatch.setenv("FUSION_SHELL_GATEWAY_URL", "http://127.0.0.1:11432")
    assert not sr.shell_registration_enabled()
    monkeypatch.setenv("FUSION_ADMIN_KEY", "fg-key")
    assert sr.shell_registration_enabled()


def test_prod_missing_admin_key_disables_not_fatal(clean_env, monkeypatch):
    # #80: fusion_core.config.admin_key raises in prod when unset — the
    # bridge must degrade to disabled registration with a warning, never die.
    monkeypatch.setenv("FUSION_SHELL_GATEWAY_URL", "http://127.0.0.1:11432")
    monkeypatch.setenv("FUSION_PROFILE", "prod")
    monkeypatch.delenv("FUSION_ADMIN_KEY", raising=False)
    assert not sr.shell_registration_enabled()


def test_default_manifest_is_valid():
    manifest = sr.load_manifest()
    assert manifest["id"] == "fusion-trainer"
    assert manifest["entry"] == "/shell/fusion-trainer/"
    assert manifest["category"] in ("platform", "domain", "infra", "dev")
    assert manifest["devices"]
    # gateway validateManifest requires id/title/path/entry/devices/category
    for field in ("id", "title", "path", "entry", "devices", "category"):
        assert manifest.get(field), f"manifest missing {field}"


# ── registry loop behavior ───────────────────────────────────────────────


def _run_loop(client, ticks):
    monkey_env = {
        "FUSION_SHELL_GATEWAY_URL": "http://127.0.0.1:11432",
        "FUSION_ADMIN_KEY": "fg-key",
    }
    old = {}
    for k, v in monkey_env.items():
        old[k] = os.environ.get(k)
        os.environ[k] = v
    try:
        asyncio.run(
            sr._registry_loop_body(client, MANIFEST, interval=0, max_ticks=ticks)
        )
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_loop_registers_and_persists_issued_key():
    client = _FakeClient([_issue_resp("fm-fusion-trainer-k1")])
    _run_loop(client, ticks=1)
    url, body, headers = _register_call(client)
    assert "Bearer fg-key" in headers["Authorization"]
    assert body["id"] == "fusion-trainer"
    assert "api_key" not in body
    assert sr.load_module_key() == "fm-fusion-trainer-k1"


def test_loop_heartbeats_with_issued_key():
    client = _FakeClient(
        [
            _issue_resp("fm-fusion-trainer-k2"),
            _Resp(200, {"code": "ok"}),
        ]
    )
    _run_loop(client, ticks=2)
    beat = [c for c in client.calls if "/heartbeat" in c[0]]
    assert len(beat) == 1
    assert "Bearer fm-fusion-trainer-k2" in beat[0][2]["Authorization"]


def test_loop_reregisters_on_gateway_restart_404():
    client = _FakeClient(
        [
            _issue_resp("fm-fusion-trainer-k3"),
            _Resp(404, {"code": "manifest_unknown"}),
            _issue_resp("fm-fusion-trainer-k3"),
        ]
    )
    _run_loop(client, ticks=3)
    reg = [c for c in client.calls if c[0].endswith("/api/v1/shell/manifests")]
    assert len(reg) == 2  # 404 heartbeat triggered a re-registration


def test_loop_reregisters_on_issued_key_invalid():
    client = _FakeClient(
        [
            _issue_resp("fm-fusion-trainer-old"),
            _Resp(403, {"code": "heartbeat_forbidden"}),
            _issue_resp("fm-fusion-trainer-new"),
        ]
    )
    _run_loop(client, ticks=3)
    assert sr.load_module_key() == "fm-fusion-trainer-new"


def test_loop_survives_register_failure():
    client = _FakeClient([_Resp(403, {"code": "manifest_rejected"})])
    _run_loop(client, ticks=1)  # must not raise
    assert sr.load_module_key() == ""


def test_loop_missing_manifest_aborts():
    async def run():
        await sr.shell_registry_loop()

    os.environ["FUSION_SHELL_MANIFEST"] = "/nonexistent/manifest.json"
    asyncio.run(run())  # logs and returns — never raises


# ── bridge inbound auth accepts the gateway-issued key ───────────────────


@pytest.fixture()
def api_client():
    from fastapi.testclient import TestClient

    return TestClient(bs.app)


def test_auth_accepts_static_and_issued_key(api_client, clean_env):
    sr.save_module_key("fm-fusion-trainer-bridgekey")
    ok_static = api_client.get(
        "/presets", headers={"Authorization": "Bearer test-bridge-key"}
    )
    ok_issued = api_client.get(
        "/presets", headers={"Authorization": "Bearer fm-fusion-trainer-bridgekey"}
    )
    bad = api_client.get("/presets", headers={"Authorization": "Bearer nope"})
    assert ok_static.status_code == 200
    assert ok_issued.status_code == 200
    assert bad.status_code == 401


def test_health_reports_shell_registration_flag(api_client, clean_env):
    body = api_client.get("/health").json()
    assert body["shell_registered"] is False
