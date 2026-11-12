"""Reward-hacking monitor tests (capability plan T7)."""

from __future__ import annotations

import json

from fusion_trainer.hacking_monitor import HackingMonitor, compute_hacking_metrics


def _feed(monitor, n, reward_fn, length_fn, share_fn, step0=0):
    for i in range(n):
        rewards = [reward_fn(i)] * 4
        comps = ["x" * int(length_fn(i))] * 4
        if share_fn(i):
            comps = ["<answer>" + c + "</answer>" for c in comps]
        monitor.observe(step0 + i, rewards, comps)


def test_flat_run_no_alarms():
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.5, lambda i: 100, lambda i: False)
    snap = m.snapshot()
    assert snap["reward_drift_alarm"] == 0
    assert snap["length_inflation_alarm"] == 0
    assert snap["format_gaming_alarm"] == 0
    assert snap["any_alarm"] == 0


def test_reward_drift_alarm():
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.2 if i < 10 else 0.9, lambda i: 100, lambda i: False)
    snap = m.snapshot()
    assert snap["reward_drift_alarm"] == 1
    assert snap["any_alarm"] == 1
    assert snap["reward_drift_abs"] > 0.5


def test_length_inflation_alarm():
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.5, lambda i: 50 if i < 10 else 200, lambda i: False)
    snap = m.snapshot()
    assert snap["length_inflation_alarm"] == 1
    assert snap["length_inflation_ratio"] > 1.5


def test_format_gaming_alarm():
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.5, lambda i: 100, lambda i: i >= 10)
    snap = m.snapshot()
    assert snap["format_gaming_alarm"] == 1
    assert snap["format_share_first"] == 0.0
    assert snap["format_share_last"] == 1.0


def test_insufficient_data_no_alarm():
    m = HackingMonitor()
    _feed(m, 3, lambda i: 0.0 + i, lambda i: 10, lambda i: True)
    snap = m.snapshot()
    assert snap["reward_points"] == 3
    assert snap["reward_drift_alarm"] == 0
    assert snap["length_inflation_alarm"] == 0
    assert snap["format_gaming_alarm"] == 0


def test_prom_series_keys_numeric():
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.5, lambda i: 100, lambda i: False)
    prom = m.prom_series()
    assert all(isinstance(v, float) for v in prom.values())
    assert "hacking_any_alarm" in prom
    assert "hacking_reward_drift_z" in prom


def test_compute_from_events_mean_reward():
    events = [
        {"step": i, "mean_reward": 0.2 if i < 10 else 0.9} for i in range(20)
    ]
    snap = compute_hacking_metrics(events)
    assert snap["reward_drift_alarm"] == 1
    assert snap["reward_points"] == 20


def test_compute_from_events_no_reward_key():
    events = [{"step": i, "train_loss": 1.0} for i in range(20)]
    snap = compute_hacking_metrics(events)
    assert snap["reward_points"] == 0
    assert snap["reward_drift_alarm"] == 0


def test_compute_mixed_streams():
    events = [
        {
            "step": i,
            "mean_reward": 0.5,
            "completion_length": 50 if i < 10 else 200,
        }
        for i in range(20)
    ]
    snap = compute_hacking_metrics(events)
    assert snap["length_inflation_alarm"] == 1
    assert snap["reward_drift_alarm"] == 0


def test_events_jsonl_roundtrip(tmp_path):
    m = HackingMonitor()
    _feed(m, 20, lambda i: 0.2 if i < 10 else 0.9, lambda i: 100, lambda i: False)
    events = [
        {"step": i, "mean_reward": 0.2 if i < 10 else 0.9} for i in range(20)
    ]
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    loaded = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]
    snap = compute_hacking_metrics(loaded)
    assert snap["reward_drift_alarm"] == m.snapshot()["reward_drift_alarm"] == 1
