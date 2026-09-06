from __future__ import annotations

import pytest

from fusion_trainer.audit import PermissionDeniedError
from fusion_trainer.run_manager import RunManager


@pytest.fixture
def rm(tmp_path, monkeypatch):
    monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
    monkeypatch.delenv("FUSION_TRAINER_API_KEY", raising=False)
    monkeypatch.delenv("FUSION_TRAINER_ACTOR", raising=False)
    return RunManager(cache_path=str(tmp_path))


class TestAuditWiring:
    def test_audit_file_created_on_init(self, rm, tmp_path):
        assert (tmp_path / "audit.jsonl").exists() or rm._audit is not None

    def test_stop_run_unknown_audited(self, rm):
        res = rm.stop_run("nope_run")
        assert res["status"] == "unknown"
        entries = rm._audit.read()
        assert any(e["action"] == "stop_run" and e["ok"] is False for e in entries)

    def test_delete_adapter_audited(self, rm):
        res = rm.delete_adapter("my-adapter")
        assert res["ok"] is False
        entries = rm._audit.read()
        assert any(
            e["action"] == "delete_adapter" and e["ok"] is False for e in entries
        )

    def test_publish_run_unknown_audited(self, rm):
        res = rm.publish_run("nope_run")
        assert res["published"] is False
        entries = rm._audit.read()
        assert any(e["action"] == "publish_run" for e in entries)

    def test_info_reports_api_key_state(self, rm):
        info = rm.info()
        assert info["api_key_required"] is False
        assert "audit.jsonl" in info["audit_log"]


class TestApiKeyGateWiring:
    def test_stop_run_rejected_with_wrong_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
        monkeypatch.setenv("FUSION_TRAINER_API_KEY", "trainer-secret")
        monkeypatch.delenv("FUSION_TRAINER_ACTOR", raising=False)
        rm = RunManager(cache_path=str(tmp_path))
        with pytest.raises(PermissionDeniedError):
            rm.stop_run("nope_run", api_key="wrong")
        entries = rm._audit.read()
        assert any(e["action"] == "stop_run" and e["ok"] is False for e in entries)
        assert rm.info()["api_key_required"] is True

    def test_stop_run_allowed_with_right_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
        monkeypatch.setenv("FUSION_TRAINER_API_KEY", "trainer-secret")
        monkeypatch.delenv("FUSION_TRAINER_ACTOR", raising=False)
        rm = RunManager(cache_path=str(tmp_path))
        res = rm.stop_run("nope_run", api_key="trainer-secret")
        assert res["status"] == "unknown"

    def test_publish_run_trainer_key_distinct_from_hub_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
        monkeypatch.setenv("FUSION_TRAINER_API_KEY", "trainer-secret")
        monkeypatch.delenv("FUSION_TRAINER_ACTOR", raising=False)
        rm = RunManager(cache_path=str(tmp_path))
        with pytest.raises(PermissionDeniedError):
            rm.publish_run("nope_run", api_key="hub-secret", trainer_api_key="wrong")
