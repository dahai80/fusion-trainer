"""Unit tests for MLXFineTuneClient header/port wiring.

Covers the two blocking bugs for real training:
  - admin route header X-Fusion-Route: admin (fusion-mlx requires it)
  - default port 11434 (fusion-mlx direct; R16 — gateway 11432 does not
    proxy /admin/api/fine-tune/*). Overridable via FUSION_MLX_PORT.
"""

from __future__ import annotations

from fusion_trainer.config import _default_mlx_port
from fusion_trainer.mlx_client import MLXFineTuneClient


class TestDefaultPort:
    def test_default_port_is_direct(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_PORT", raising=False)
        assert _default_mlx_port() == 11434

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_PORT", "12000")
        assert _default_mlx_port() == 12000


class TestAdminRouteHeader:
    def test_route_header_always_present(self):
        client = MLXFineTuneClient(port=11434)
        assert client._headers.get("X-Fusion-Route") == "admin"

    def test_route_header_with_token(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_ADMIN_TOKEN", "secret-token")
        client = MLXFineTuneClient(port=11434)
        assert client._headers.get("X-Fusion-Route") == "admin"
        assert client._headers.get("Authorization") == "Bearer secret-token"

    def test_no_auth_header_without_token(self, monkeypatch, tmp_path):
        # #80: resolve_api_key falls back to ~/.fusion-mlx/settings.json —
        # point HOME at an empty tmp dir and clear the fusion_core settings
        # cache so the test is hermetic on machines that DO have the file.
        import fusion_core.config as fc_config

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("FUSION_MLX_ADMIN_TOKEN", raising=False)
        fc_config._SETTINGS_CACHE.clear()
        client = MLXFineTuneClient(port=11434)
        assert client._headers.get("X-Fusion-Route") == "admin"
        assert "Authorization" not in client._headers
        fc_config._SETTINGS_CACHE.clear()

    def test_settings_json_fallback_auth(self, monkeypatch, tmp_path):
        # #80: no env token, but ~/.fusion-mlx/settings.json carries
        # auth.api_key — resolve_api_key picks it up as the Bearer token.
        import fusion_core.config as fc_config

        settings_dir = tmp_path / ".fusion-mlx"
        settings_dir.mkdir()
        (settings_dir / "settings.json").write_text(
            '{"auth": {"api_key": "settings-json-token"}}'
        )
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.delenv("FUSION_MLX_ADMIN_TOKEN", raising=False)
        fc_config._SETTINGS_CACHE.clear()
        try:
            client = MLXFineTuneClient(port=11434)
            assert client._headers.get("Authorization") == "Bearer settings-json-token"
        finally:
            fc_config._SETTINGS_CACHE.clear()

    def test_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_ADMIN_TOKEN", "env-token")
        client = MLXFineTuneClient(port=11434, admin_token="explicit-token")
        assert client._headers.get("Authorization") == "Bearer explicit-token"


class TestFusionSourceHeader:
    # Bridge/model-manager hosts must declare X-Fusion-Source: model-hub —
    # merge/serve model-management endpoints are gated on it
    # (fusion-mlx auth.py:531). #100: the declaration is sent by default
    # (env FUSION_MLX_SOURCE overrides; explicit arg wins) — a merge
    # through the gateway 403s without it.
    def test_source_header_when_set(self):
        client = MLXFineTuneClient(port=11434, source="model-hub")
        assert client._headers.get("X-Fusion-Source") == "model-hub"

    def test_model_hub_by_default(self):
        client = MLXFineTuneClient(port=11434)
        assert client._headers.get("X-Fusion-Source") == "model-hub"


class TestBaseURL:
    def test_base_url_includes_port(self):
        client = MLXFineTuneClient(port=11434)
        assert client._base == "http://localhost:11434"


class TestScoreReward:
    # fusion-mlx #431: score_reward POST /admin/api/fine-tune/reward/score
    def test_score_reward_happy_path(self, monkeypatch):
        captured = {}

        class _Resp:
            status_code = 200

            def json(self):
                return {
                    "rewards": [6.7, -7.2],
                    "model_id": "qwen-0.6b",
                    "adapter_name": "rm-adapter",
                }

            @property
            def text(self):
                return ""

        # A8: all egress now goes through the pooled self._http.request, so
        # mock that instead of the module-level httpx.post.
        class _FakeHttp:
            def request(self, method, url, json=None, params=None):
                captured["url"] = url
                captured["json"] = json
                return _Resp()

            def close(self):
                pass

        client = MLXFineTuneClient(port=11434, admin_token="tok")
        client._http = _FakeHttp()
        result = client.score_reward(
            model_id="qwen-0.6b",
            adapter_name="rm-adapter",
            prompt="hello",
            completions=["a", "b"],
        )
        assert result["rewards"] == [6.7, -7.2]
        assert captured["url"].endswith("/admin/api/fine-tune/reward/score")
        assert captured["json"]["model_id"] == "qwen-0.6b"
        assert captured["json"]["adapter_name"] == "rm-adapter"
        assert captured["json"]["prompt"] == "hello"
        assert captured["json"]["completions"] == ["a", "b"]
        assert client._headers["Authorization"] == "Bearer tok"

    def test_score_reward_raises_on_error(self, monkeypatch):
        class _Resp:
            status_code = 404
            text = "adapter not found"

            def json(self):
                return {}

        from fusion_trainer.mlx_client import MLXFineTuneError

        class _FakeHttp:
            def request(self, method, url, json=None, params=None):
                return _Resp()

            def close(self):
                pass

        client = MLXFineTuneClient(port=11434)
        client._http = _FakeHttp()
        import pytest

        with pytest.raises(MLXFineTuneError, match="404"):
            client.score_reward("m", "a", "p", ["c"])


def test_job_event_cursor_synthesizes_from_progress():
    # fusion-mlx to_dict carries no events array — cursor falls back to the
    # `progress` snapshot and emits one train_loss event per advanced step.
    from fusion_trainer.mlx_client import _JobEventCursor

    c = _JobEventCursor()
    assert c.collect({"status": "running", "progress": {"step": 0}}) == []
    ev1 = c.collect(
        {
            "status": "running",
            "progress": {"step": 1, "train_loss": 1.5, "val_loss": 1.2},
        }
    )
    assert len(ev1) == 1 and ev1[0]["type"] == "train_loss" and ev1[0]["step"] == 1
    # same step again: no duplicate
    assert (
        c.collect({"status": "running", "progress": {"step": 1, "train_loss": 1.5}})
        == []
    )
    ev2 = c.collect({"status": "running", "progress": {"step": 2, "train_loss": 1.1}})
    assert ev2[0]["step"] == 2


def test_job_event_cursor_prefers_events_array():
    from fusion_trainer.mlx_client import _JobEventCursor

    c = _JobEventCursor()
    job = {
        "events": [{"type": "train_loss", "step": 1}, {"type": "val_loss", "step": 1}]
    }
    assert len(c.collect(job)) == 2
    assert c.collect(job) == []


def test_client_default_fusion_source_header(monkeypatch):
    # #100: model-management endpoints are gated on X-Fusion-Source; the
    # client declares model-hub by default, FUSION_MLX_SOURCE overrides,
    # an explicit constructor arg wins over both.
    monkeypatch.delenv("FUSION_MLX_SOURCE", raising=False)
    assert MLXFineTuneClient(port=11434)._headers["X-Fusion-Source"] == "model-hub"
    monkeypatch.setenv("FUSION_MLX_SOURCE", "trainer-console")
    assert (
        MLXFineTuneClient(port=11434)._headers["X-Fusion-Source"] == "trainer-console"
    )
    assert (
        MLXFineTuneClient(port=11434, source="explicit")._headers["X-Fusion-Source"]
        == "explicit"
    )


def test_merge_adapter_normalizes_catalog_id():
    # #100: merge-adapter base model must be the slash repo id; the dash
    # catalog form is rejected by hf ("Cannot have -- in repo_id").
    client = MLXFineTuneClient(port=11434)
    captured = {}

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"status": "ok"}

        text = "{}"

    class _FakeHttp:
        def request(self, method, url, json=None, params=None):
            captured["body"] = json
            return _FakeResp()

    client._http = _FakeHttp()
    client.merge_adapter("mlx-community--Qwen3-0.6B-4bit", "/tmp/adapter")
    assert captured["body"]["model"] == "mlx-community/Qwen3-0.6B-4bit"
    client.merge_adapter("sft_1795605347026_00eee724-20261125-191812", "/tmp/a")
    assert captured["body"]["model"] == "sft_1795605347026_00eee724-20261125-191812"
