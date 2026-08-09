"""Unit tests for MLXFineTuneClient header/port wiring.

Covers the two blocking bugs for real training:
  - admin route header X-Fusion-Route: admin (fusion-mlx requires it)
  - default port 11432 (fusion-gateway, routes to fusion-mlx; see issue #16).
    Overridable via FUSION_MLX_PORT (e.g. 11434 to bypass gateway).
"""

from __future__ import annotations

from fusion_trainer.config import _default_mlx_port
from fusion_trainer.mlx_client import MLXFineTuneClient


class TestDefaultPort:
    def test_default_port_is_gateway(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_PORT", raising=False)
        assert _default_mlx_port() == 11432

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

    def test_no_auth_header_without_token(self, monkeypatch):
        monkeypatch.delenv("FUSION_MLX_ADMIN_TOKEN", raising=False)
        client = MLXFineTuneClient(port=11434)
        assert client._headers.get("X-Fusion-Route") == "admin"
        assert "Authorization" not in client._headers

    def test_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("FUSION_MLX_ADMIN_TOKEN", "env-token")
        client = MLXFineTuneClient(port=11434, admin_token="explicit-token")
        assert client._headers.get("Authorization") == "Bearer explicit-token"


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

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

        import fusion_trainer.mlx_client as client_mod

        monkeypatch.setattr(client_mod.httpx, "post", _fake_post)
        client = MLXFineTuneClient(port=11434, admin_token="tok")
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
        assert captured["headers"]["Authorization"] == "Bearer tok"

    def test_score_reward_raises_on_error(self, monkeypatch):
        class _Resp:
            status_code = 404
            text = "adapter not found"

            def json(self):
                return {}

        import fusion_trainer.mlx_client as client_mod
        from fusion_trainer.mlx_client import MLXFineTuneError

        monkeypatch.setattr(
            client_mod.httpx,
            "post",
            lambda *a, **kw: _Resp(),
        )
        client = MLXFineTuneClient(port=11434)
        import pytest

        with pytest.raises(MLXFineTuneError, match="404"):
            client.score_reward("m", "a", "p", ["c"])
