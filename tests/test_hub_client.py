"""Unit tests for ModelHubClient + publish_adapter_if_enabled (issue #49).

Mocks fusion-model-hub POST /api/v1/models via monkeypatch on
hub_client.httpx.post — same pattern as tests/test_mlx_client.py.
"""

from __future__ import annotations

import pytest

import fusion_trainer.hub_client as hub_mod
from fusion_trainer.config import SFTConfig, TrainerConfig
from fusion_trainer.hub_client import (
    HubPublishError,
    ModelHubClient,
    publish_adapter_if_enabled,
)


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    @property
    def content(self):
        return b"x" if self._payload or self.text else b""

    def json(self):
        return self._payload


class TestClientHeaders:
    def test_base_url_stripped(self):
        client = ModelHubClient(base_url="http://localhost:11432/")
        assert client._base == "http://localhost:11432"

    def test_auth_header_when_key(self):
        client = ModelHubClient(base_url="http://h", api_key="k")
        assert client._headers["Authorization"] == "Bearer k"

    def test_no_auth_without_key(self):
        client = ModelHubClient(base_url="http://h")
        assert "Authorization" not in client._headers


class TestPublishAdapterHappy:
    def test_201_returns_published_true(self, monkeypatch):
        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp(201, payload={"id": "m1", "name": "my-adapter"})

        monkeypatch.setattr(hub_mod.httpx, "post", _fake_post)
        client = ModelHubClient(base_url="http://hub:11432", api_key="tok")
        result = client.publish_adapter(
            name="my-adapter", base_model="qwen2.5-7b", description="d"
        )
        assert result["published"] is True
        assert result["model"]["id"] == "m1"
        assert captured["url"] == "http://hub:11432/api/v1/models"
        assert captured["json"]["model_type"] == "lora"
        assert captured["json"]["base_model_id"] == "qwen2.5-7b"
        assert captured["json"]["name"] == "my-adapter"
        assert captured["headers"]["Authorization"] == "Bearer tok"

    def test_tags_included_when_given(self, monkeypatch):
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(201, payload={"id": "m"}),
        )
        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["json"] = json
            return _Resp(201, payload={"id": "m"})

        monkeypatch.setattr(hub_mod.httpx, "post", _fake_post)
        client = ModelHubClient(base_url="http://h")
        client.publish_adapter(
            name="my-adapter",
            base_model="b",
            tags=[{"key": "src", "value": "sft"}],
        )
        assert captured["json"]["tags"] == [{"key": "src", "value": "sft"}]


class TestPublishAdapterErrors:
    def test_short_name_raises(self):
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="too short"):
            client.publish_adapter(name="ab", base_model="b")

    def test_empty_base_model_raises(self):
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="base_model is required"):
            client.publish_adapter(name="adapter", base_model="")

    def test_404_base_model_missing_raises(self, monkeypatch):
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(
                404, payload={"detail": "base not found"}, text="nf"
            ),
        )
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="404"):
            client.publish_adapter(name="adapter", base_model="missing")

    def test_409_name_exists_raises(self, monkeypatch):
        # L4: a 409 whose existing entry has a DIFFERENT base_model (or no
        # matching entry is found) is a real name collision and still raises.
        # GET list_models returns an empty list here -> no own match -> raise.
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(409, payload={"detail": "exists"}),
        )
        monkeypatch.setattr(
            hub_mod.httpx, "get", lambda *a, **kw: _Resp(200, payload=[])
        )
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="409"):
            client.publish_adapter(name="adapter", base_model="b")

    def test_409_idempotent_when_own_model_exists(self, monkeypatch):
        # L4: a 409 is idempotent if a lora with the same name AND base_model_id
        # already exists -> treat as already-published, return that model.
        existing = {
            "id": "hub-1",
            "name": "adapter",
            "base_model_id": "b",
            "model_type": "lora",
        }
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(409, payload={"detail": "exists"}),
        )
        monkeypatch.setattr(
            hub_mod.httpx,
            "get",
            lambda *a, **kw: _Resp(200, payload=[existing]),
        )
        client = ModelHubClient(base_url="http://h")
        result = client.publish_adapter(name="adapter", base_model="b")
        assert result["published"] is True
        assert result["idempotent"] is True
        assert result["model"]["id"] == "hub-1"

    def test_transport_error_raises(self, monkeypatch):
        import httpx as _httpx

        def _boom(*a, **kw):
            raise _httpx.ConnectError("no route")

        monkeypatch.setattr(hub_mod.httpx, "post", _boom)
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="transport error"):
            client.publish_adapter(name="adapter", base_model="b")

    def test_201_empty_body_raises_no_dangling_success(self, monkeypatch):
        # R18: a 201 with empty body must not report published=True (dangling).
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(201, payload=None, text=""),
        )
        client = ModelHubClient(base_url="http://h")
        with pytest.raises(HubPublishError, match="no model id"):
            client.publish_adapter(name="adapter", base_model="b")


class TestListModelsEnvelope:
    def test_data_envelope_unwrapped(self, monkeypatch):
        # R17: hub returning {"data":[...]} must be unwrapped, not treated as
        # empty (which silently broke L4 409 idempotency).
        existing = {
            "id": "hub-2",
            "name": "adapter",
            "base_model_id": "b",
            "model_type": "lora",
        }
        monkeypatch.setattr(
            hub_mod.httpx,
            "get",
            lambda *a, **kw: _Resp(200, payload={"data": [existing]}),
        )
        client = ModelHubClient(base_url="http://h")
        models = client.list_models(model_type="lora", keyword="adapter")
        assert models and models[0]["id"] == "hub-2"


class TestPublishAdapterIfEnabled:
    def _cfg(self, publish=True):
        return TrainerConfig(
            sft=SFTConfig(model="qwen2.5-7b"),
            publish_adapter=publish,
        )

    def test_disabled_noop(self):
        cfg = self._cfg(publish=False)
        result = publish_adapter_if_enabled(
            cfg, {"status": "completed", "adapter_name": "a"}
        )
        assert result["published"] is False
        assert result["reason"] == "disabled"

    def test_non_completed_skips(self):
        cfg = self._cfg()
        result = publish_adapter_if_enabled(
            cfg, {"status": "failed", "adapter_name": "a"}
        )
        assert result["published"] is False
        assert "status=" in result["reason"]

    def test_no_adapter_name_skips(self):
        cfg = self._cfg()
        result = publish_adapter_if_enabled(cfg, {"status": "completed"})
        assert result["published"] is False
        assert result["reason"] == "no adapter_name in result"

    def test_phase2_nested_name_extracted(self, monkeypatch):
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(201, payload={"id": "m"}),
        )
        cfg = self._cfg()
        result = publish_adapter_if_enabled(
            cfg,
            {
                "status": "completed",
                "method": "rlsl",
                "phase2": {"adapter_name": "rlsl-ad"},
            },
        )
        assert result["published"] is True

    def test_failure_warns_not_raises(self, monkeypatch):
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(500, payload={"detail": "boom"}),
        )
        cfg = self._cfg()
        result = publish_adapter_if_enabled(
            cfg, {"status": "completed", "adapter_name": "my-ad", "method": "lora"}
        )
        assert result["published"] is False
        assert "error" in result
        assert result["adapter_name"] == "my-ad"

    def test_env_fallback_publish_when_config_field_false(self, monkeypatch):
        # S3 (soft #3): the GUI path drops publish_adapter from the config it
        # builds (upstream fusion-agent-studio#268). FUSION_TRAINER_PUBLISH=1
        # must force publish on even when config.publish_adapter is False, so
        # GUI-triggered runs can still publish via env until #268 lands.
        monkeypatch.setenv("FUSION_TRAINER_PUBLISH", "1")
        monkeypatch.setattr(
            hub_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(201, payload={"id": "m"}),
        )
        cfg = self._cfg(publish=False)  # GUI dropped the flag -> False
        result = publish_adapter_if_enabled(
            cfg, {"status": "completed", "adapter_name": "gui-ad", "method": "lora"}
        )
        assert result["published"] is True

    def test_env_fallback_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("FUSION_TRAINER_PUBLISH", raising=False)
        cfg = self._cfg(publish=False)
        result = publish_adapter_if_enabled(
            cfg, {"status": "completed", "adapter_name": "a"}
        )
        assert result["published"] is False
        assert result["reason"] == "disabled"
