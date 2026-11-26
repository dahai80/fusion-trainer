"""Dashboard payload tests (issue #93): live data + honest empty states."""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import bridge.dashboard as bd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_registry(cache_path: Path, rounds: list[dict]) -> None:
    (cache_path / "flywheel.json").write_text(
        json.dumps({"rounds": rounds}), encoding="utf-8"
    )


def _round(label, pass_rate, status="promoted", by_type=None, baseline=False):
    return {
        "label": label,
        "model": "qwen3-27b-8bit",
        "pass_rate": pass_rate,
        "passed": int(pass_rate * 30),
        "total": 30,
        "by_type": by_type or {},
        "baseline": baseline,
        "status": status,
        "recorded_at": "2026-11-25T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# flywheel section
# ---------------------------------------------------------------------------


def test_flywheel_missing_registry(tmp_path):
    out = bd.flywheel_section(str(tmp_path))
    assert out["available"] is False
    assert "no flywheel registry" in out["reason"]
    assert out["rounds"] == []


def test_flywheel_malformed_registry(tmp_path):
    (tmp_path / "flywheel.json").write_text("{bad", encoding="utf-8")
    out = bd.flywheel_section(str(tmp_path))
    assert out["available"] is False
    assert "unreadable" in out["reason"]


def test_flywheel_rounds_and_progression(tmp_path):
    _write_registry(
        tmp_path,
        [
            _round("baseline", 0.4, baseline=True),
            _round("sft-r1", 0.6),
            _round("grpo-r2", 0.5, status="discarded"),
        ],
    )
    out = bd.flywheel_section(str(tmp_path))
    assert out["available"] is True
    assert len(out["rounds"]) == 3
    assert out["incumbent_pass_rate"] == 0.6
    assert out["latest"]["label"] == "grpo-r2"
    assert len(out["progression"]) == 3
    assert "DISCARDED" in out["gate_note"]


def test_flywheel_by_type_latest_round(tmp_path):
    _write_registry(
        tmp_path,
        [
            _round(
                "r1",
                0.5,
                by_type={"email": {"pass_rate": 0.4, "passed": 2, "total": 5}},
            ),
            _round(
                "r2",
                0.7,
                by_type={
                    "email": {"pass_rate": 0.8, "passed": 4, "total": 5},
                    "slide": 0.6,
                },
            ),
        ],
    )
    out = bd.flywheel_section(str(tmp_path))
    types = {t["type"]: t["pass_rate"] for t in out["by_type"]}
    assert types == {"email": 0.8, "slide": 0.6}
    assert out["by_type_round"] == "r2"


def test_flywheel_by_type_falls_back(tmp_path):
    _write_registry(
        tmp_path,
        [
            _round(
                "r1",
                0.5,
                by_type={"email": {"pass_rate": 0.4, "passed": 2, "total": 5}},
            ),
            _round("r2", 0.7),  # no breakdown (e.g. DPO round)
        ],
    )
    out = bd.flywheel_section(str(tmp_path))
    types = {t["type"]: t["pass_rate"] for t in out["by_type"]}
    assert types == {"email": 0.4}
    assert out["by_type_round"] == "r1"


# ---------------------------------------------------------------------------
# rollout section
# ---------------------------------------------------------------------------


def _rollout(rid, model="m", created=None, outcome=None):
    rec = {
        "schema_version": 1,
        "rollout_id": rid,
        "source": "chat_completions",
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if created:
        rec["created_at"] = created
    if outcome is not None:
        rec["outcome"] = outcome
    return rec


def test_rollouts_no_env(monkeypatch):
    monkeypatch.delenv(bd.ROLLOUT_SINKS_ENV, raising=False)
    out = bd.rollout_section()
    assert out["available"] is False
    assert bd.ROLLOUT_SINKS_ENV in out["reason"]


def test_rollouts_counts_and_signals(tmp_path, monkeypatch):
    sink = tmp_path / "tap.jsonl"
    now = datetime.now(UTC)
    lines = [
        _rollout("ro-1", outcome={"signal": "task_success", "score": 1.0}),
        _rollout("ro-2", outcome={"signal": "user_rejected", "score": 0.0}),
        _rollout(
            "ro-3",
            created=now.isoformat(),
            outcome={"signal": "task_success"},
        ),
        _rollout("ro-4"),  # unobserved outcome still counts as GRPO prompt
        '{"broken',  # invalid line, counted loudly
    ]
    sink.write_text(
        "\n".join(json.dumps(r) if isinstance(r, dict) else r for r in lines),
        encoding="utf-8",
    )
    monkeypatch.setenv(bd.ROLLOUT_SINKS_ENV, str(sink))
    out = bd.rollout_section()
    assert out["available"] is True
    assert out["total"] == 4
    assert out["recent"] == 1
    assert out["signals"] == {
        "task_success": 2,
        "unobserved": 1,
        "user_rejected": 1,
    }
    assert out["positive_share"] == 2 / 3  # 2 of 3 scored >= 1.0 (ro-4 unobserved)
    assert out["invalid_lines"] == 1


def test_rollouts_signal_default_scores(tmp_path, monkeypatch):
    sink = tmp_path / "tap.jsonl"
    sink.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                _rollout("a", outcome={"signal": "user_accepted"}),
                _rollout("b", outcome={"signal": "user_rejected"}),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(bd.ROLLOUT_SINKS_ENV, str(sink))
    out = bd.rollout_section()
    assert out["positive_share"] == 0.5


def test_rollouts_missing_path_warns(tmp_path, monkeypatch):
    monkeypatch.setenv(bd.ROLLOUT_SINKS_ENV, str(tmp_path / "nope"))
    out = bd.rollout_section()
    assert out["available"] is False
    assert "no readable sink files" in out["reason"]


# ---------------------------------------------------------------------------
# payload assembly
# ---------------------------------------------------------------------------


def test_next_step_priorities():
    fw_empty = {"available": False}
    assert "flywheel record" in bd.next_step(fw_empty, {"available": False})
    fw_disc = {"available": True, "latest": {"status": "discarded"}}
    assert "discarded" in bd.next_step(fw_disc, {"available": False})
    fw_ok = {"available": True, "latest": {"status": "promoted"}}
    assert "rollout tap" in bd.next_step(fw_ok, {"available": False})
    assert "init-adapter-path" in bd.next_step(fw_ok, {"available": True, "total": 5})


def test_build_payload_shape(tmp_path, monkeypatch):
    monkeypatch.delenv(bd.ROLLOUT_SINKS_ENV, raising=False)
    _write_registry(tmp_path, [_round("baseline", 0.4, baseline=True)])
    payload = bd.build_payload(str(tmp_path), [{"run_id": "r1", "status": "completed"}])
    assert payload["flywheel"]["available"] is True
    assert payload["runs"][0]["run_id"] == "r1"
    assert payload["rollouts"]["available"] is False
    assert payload["reward_composition"]["available"] is False
    assert payload["next_step"]


def test_build_payload_run_limit(tmp_path, monkeypatch):
    monkeypatch.delenv(bd.ROLLOUT_SINKS_ENV, raising=False)
    runs = [{"run_id": f"r{i}"} for i in range(30)]
    payload = bd.build_payload(str(tmp_path), runs)
    assert len(payload["runs"]) == bd._MAX_RUNS


# ---------------------------------------------------------------------------
# endpoint wiring
# ---------------------------------------------------------------------------


def test_api_dashboard_endpoint(tmp_path, monkeypatch):
    import os

    os.environ.setdefault("FUSION_TRAINER_BRIDGE_KEY", "test-bridge-key")
    from fastapi.testclient import TestClient

    import bridge.server as bs

    monkeypatch.setenv("FUSION_TRAINER_CACHE", str(tmp_path))
    monkeypatch.delenv(bd.ROLLOUT_SINKS_ENV, raising=False)
    client = TestClient(bs.app)
    r = client.get(
        "/api/dashboard", headers={"Authorization": "Bearer test-bridge-key"}
    )
    assert r.status_code == 200
    body = r.json()
    assert "flywheel" in body and "runs" in body and "rollouts" in body
    r2 = client.get("/api/dashboard")
    assert r2.status_code == 401


def test_dashboard_page_served():
    import os

    os.environ.setdefault("FUSION_TRAINER_BRIDGE_KEY", "test-bridge-key")
    from fastapi.testclient import TestClient

    import bridge.server as bs

    client = TestClient(bs.app)
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "训练闭环看板" in r.text


# ---------------------------------------------------------------------------
# outcome sidecar join (#97)
# ---------------------------------------------------------------------------


def test_rollouts_sidecar_join_no_double_count(tmp_path, monkeypatch):
    tap = tmp_path / "rollout-tap.jsonl"
    side = tmp_path / "rollout-outcomes.jsonl"
    now = datetime.now(UTC)
    tap.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    **_rollout("ro-1", created=now.isoformat()),
                    "source": "chat_completions",
                },
                {**_rollout("ro-2"), "source": "chat_completions"},
            ]
        ),
        encoding="utf-8",
    )
    side.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {
                    "schema_version": 1,
                    "rollout_id": "ro-1",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "outcome": {"signal": "task_success", "score": 1.0},
                },
                {
                    "schema_version": 1,
                    "rollout_id": "ro-9",
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "outcome": {"signal": "user_rejected", "score": 0.0},
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(bd.ROLLOUT_SINKS_ENV, os.pathsep.join([str(tap), str(side)]))
    out = bd.rollout_section()
    assert out["total"] == 2  # sidecar lines never counted as trajectories
    assert out["signals"] == {"task_success": 1, "unobserved": 1}
    assert out["positive_share"] == 1.0  # ro-1 joined positive; ro-2 unobserved


def test_rollouts_inline_outcome_beats_sidecar(tmp_path, monkeypatch):
    tap = tmp_path / "tap.jsonl"
    side = tmp_path / "side.jsonl"
    tap.write_text(
        json.dumps(
            {
                **_rollout("ro-1"),
                "source": "chat_completions",
                "outcome": {"signal": "user_rejected", "score": 0.0},
            }
        ),
        encoding="utf-8",
    )
    side.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout_id": "ro-1",
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "outcome": {"signal": "task_success", "score": 1.0},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(bd.ROLLOUT_SINKS_ENV, os.pathsep.join([str(tap), str(side)]))
    out = bd.rollout_section()
    assert out["signals"] == {"user_rejected": 1}
    assert out["positive_share"] == 0.0
