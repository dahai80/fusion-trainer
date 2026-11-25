"""Auto-record watch tests (#95): summary landing -> gated round, no double-record."""

from __future__ import annotations

import json

import pytest

from fusion_trainer.flywheel_watch import (
    load_state,
    record_new_summaries,
    save_state,
    scan_summaries,
)


def _summary(dir_, run_id, pass_rate=0.7):
    d = dir_ / run_id
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "models": {"m-a": "model-x"},
                "arms": {
                    "m-a": {
                        "total": 10,
                        "passed": int(pass_rate * 10),
                        "pass_rate": pass_rate,
                        "by_type": {
                            "fix": {
                                "pass_rate": pass_rate,
                                "passed": int(pass_rate * 10),
                                "total": 10,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return d / "summary.json"


def _watch_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    return d


def test_scan_sorted_and_empty(tmp_path):
    d = _watch_dir(tmp_path)
    assert scan_summaries(d) == []
    _summary(d, "run-b")
    _summary(d, "run-a")
    assert [p.parent.name for p in scan_summaries(d)] == ["run-a", "run-b"]


def test_record_new_and_no_double(tmp_path):
    d = _watch_dir(tmp_path)
    s = _summary(d, "run-1", 0.6)
    out = record_new_summaries(d, str(tmp_path / "cache"))
    assert len(out) == 1
    assert out[0]["label"] == "run-1"
    assert out[0]["pass_rate"] == pytest.approx(0.6)
    # same content again: sha match, no double-record
    assert record_new_summaries(d, str(tmp_path / "cache")) == []


def test_rerecord_on_content_change(tmp_path):
    d = _watch_dir(tmp_path)
    s = _summary(d, "run-1", 0.6)
    cache = str(tmp_path / "cache")
    record_new_summaries(d, cache)
    s.write_text(s.read_text(encoding="utf-8").replace("0.6", "0.8"), encoding="utf-8")
    out = record_new_summaries(d, cache)
    assert len(out) == 1
    assert out[0]["pass_rate"] == pytest.approx(0.8)


def test_malformed_summary_not_consumed(tmp_path, caplog):
    d = _watch_dir(tmp_path)
    bad = d / "run-bad"
    bad.mkdir()
    (bad / "summary.json").write_text("{nope", encoding="utf-8")
    out = record_new_summaries(d, str(tmp_path / "cache"))
    assert out == []
    assert load_state(str(tmp_path / "cache"))["consumed"] == {}


def test_state_roundtrip(tmp_path):
    cache = str(tmp_path / "cache")
    save_state(cache, {"consumed": {"a": {"sha256": "x"}}})
    assert load_state(cache)["consumed"]["a"]["sha256"] == "x"


def test_watch_once_counts(tmp_path):
    d = _watch_dir(tmp_path)
    _summary(d, "run-1", 0.5)
    _summary(d, "run-2", 0.7)
    from fusion_trainer.flywheel_watch import watch

    n = watch(d, str(tmp_path / "cache"), gate=False, once=True)
    assert n == 2


def test_gate_discard_flow(tmp_path):
    d = _watch_dir(tmp_path)
    cache = str(tmp_path / "cache")
    _summary(d, "run-high", 0.9)
    record_new_summaries(d, cache, gate=True)
    _summary(d, "run-low", 0.3)
    out = record_new_summaries(d, cache, gate=True)
    assert len(out) == 1
    assert out[0]["status"] == "discarded"
    state = load_state(cache)
    assert (
        state["consumed"][str(d / "run-low" / "summary.json")]["status"] == "discarded"
    )
