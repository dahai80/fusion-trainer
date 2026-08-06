"""Unit tests for MLXFineTuneClient header/port wiring.

Covers the two blocking bugs for real training:
  - admin route header X-Fusion-Route: admin (fusion-mlx requires it)
  - default port 11434 (fusion-mlx canonical; 11432 is the gateway)
"""

from __future__ import annotations

from fusion_trainer.config import _default_mlx_port
from fusion_trainer.mlx_client import MLXFineTuneClient


class TestDefaultPort:
    def test_default_port_is_fusion_mlx(self, monkeypatch):
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
