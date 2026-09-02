from __future__ import annotations

import json

import pytest

from fusion_trainer.audit import ApiKeyGate, AuditLog, PermissionDeniedError


class TestAuditLog:
    def test_record_appends_jsonl(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("start_sft", run_id="run-1", detail="qwen-7b", ok=True)
        entries = log.read()
        assert len(entries) == 1
        e = entries[0]
        assert e["action"] == "start_sft"
        assert e["run_id"] == "run-1"
        assert e["detail"] == "qwen-7b"
        assert e["ok"] is True
        assert e["actor"] == "local"
        assert "timestamp" in e

    def test_multiple_records_ordered(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("start_sft", run_id="a")
        log.record("stop_run", run_id="a", ok=True)
        log.record("publish_run", run_id="a", ok=False, detail="hub down")
        entries = log.read()
        assert [e["action"] for e in entries] == [
            "start_sft",
            "stop_run",
            "publish_run",
        ]
        assert entries[2]["ok"] is False
        assert entries[2]["detail"] == "hub down"

    def test_actor_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FUSION_TRAINER_ACTOR", "operator-alice")
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("stop_run", run_id="x")
        assert log.read()[0]["actor"] == "operator-alice"

    def test_actor_explicit_override(self, tmp_path):
        log = AuditLog(tmp_path / "audit.jsonl")
        log.record("delete_adapter", run_id="x", actor="cron")
        assert log.read()[0]["actor"] == "cron"

    def test_file_is_append_only_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        log = AuditLog(path)
        log.record("start_sft", run_id="1")
        log.record("stop_run", run_id="1")
        text = path.read_text()
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == 2
        for l in lines:
            json.loads(l)

    def test_read_missing_file_empty(self, tmp_path):
        log = AuditLog(tmp_path / "nope.jsonl")
        assert log.read() == []

    def test_read_skips_unparseable_lines(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        path.write_text('{"action":"ok"}\nBROKEN\n{"action":"ok2"}\n')
        log = AuditLog(path)
        entries = log.read()
        assert [e["action"] for e in entries] == ["ok", "ok2"]


class TestApiKeyGate:
    def test_unset_env_allows_all(self, monkeypatch):
        monkeypatch.delenv("FUSION_TRAINER_API_KEY", raising=False)
        gate = ApiKeyGate()
        assert not gate.enabled
        gate.check(None)
        gate.check("anything")
        gate.check("")

    def test_set_env_rejects_wrong_key(self, monkeypatch):
        monkeypatch.setenv("FUSION_TRAINER_API_KEY", "s3cret")
        gate = ApiKeyGate()
        assert gate.enabled
        gate.check("s3cret")
        with pytest.raises(PermissionDeniedError):
            gate.check("wrong")
        with pytest.raises(PermissionDeniedError):
            gate.check(None)

    def test_explicit_key_at_construction(self, monkeypatch):
        monkeypatch.delenv("FUSION_TRAINER_API_KEY", raising=False)
        gate = ApiKeyGate(key="explicit-key")
        assert gate.enabled
        gate.check("explicit-key")
        with pytest.raises(PermissionDeniedError):
            gate.check("nope")
