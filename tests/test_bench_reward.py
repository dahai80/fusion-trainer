"""Bench-as-GRPO-verifier tests (capability plan 3.1)."""

from __future__ import annotations

import http.client
import json
from urllib.parse import urlparse

import pytest

from fusion_trainer.rlsl.reward_server import RewardServer


class FakeBenchClient:
    def __init__(self, score=0.75, fail=False):
        self.score = score
        self.fail = fail
        self.calls: list[dict] = []

    def run_benchmark(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("bench down")
        return {
            "status": "completed",
            "task_id": "t-1",
            "score": self.score,
            "pass_rate": self.score,
        }


DATASET = [
    {
        "prompt": "what is 2+2",
        "answer": "4",
        "bench_task": "math/basic",
        "bench_params": {"level": "L1"},
    }
]


def _post_reward(endpoint: str, body: dict) -> dict:
    parsed = urlparse(endpoint)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
    payload = json.dumps(body).encode("utf-8")
    conn.request(
        "POST",
        f"{parsed.path}?{parsed.query}",
        body=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read().decode("utf-8"))
    conn.close()
    return data


def test_bench_mode_requires_client():
    with pytest.raises(ValueError, match="requires a bench_client"):
        RewardServer(DATASET, mode="bench", bench_client=None).start()


def test_bench_scoring_happy_path():
    bench = FakeBenchClient(score=0.8)
    server = RewardServer(
        DATASET, mode="bench", scale=1.0, bench_client=bench
    )
    endpoint = server.start()
    try:
        out = _post_reward(
            endpoint, {"prompt": "what is 2+2", "completions": ["<answer>4</answer>"]}
        )
        assert out["rewards"] == [0.8]
        assert bench.calls[0]["params"]["bench_task"] == "math/basic"
        assert bench.calls[0]["params"]["completion"] == "<answer>4</answer>"
        assert server.bench_degraded is False
    finally:
        server.stop()


def test_bench_score_scaled_and_cached():
    bench = FakeBenchClient(score=0.5)
    server = RewardServer(
        DATASET, mode="bench", scale=2.0, bench_client=bench
    )
    endpoint = server.start()
    try:
        out = _post_reward(
            endpoint, {"prompt": "what is 2+2", "completions": ["4", "4"]}
        )
        assert out["rewards"] == [1.0, 1.0]
        assert len(bench.calls) == 1
    finally:
        server.stop()


def test_bench_outage_degrades_to_sandbox():
    bench = FakeBenchClient(fail=True)
    server = RewardServer(
        DATASET, mode="bench", scale=1.0, bench_client=bench
    )
    endpoint = server.start()
    try:
        out = _post_reward(
            endpoint, {"prompt": "what is 2+2", "completions": ["<answer>4</answer>"]}
        )
        # sandbox math reward on a correct answer, not a constant
        assert isinstance(out["rewards"][0], float)
        assert server.bench_degraded is True
    finally:
        server.stop()


def test_bench_sample_without_task_zero_reward():
    bench = FakeBenchClient(score=0.9)
    server = RewardServer(
        [{"prompt": "p", "answer": "a"}], mode="bench", bench_client=bench
    )
    endpoint = server.start()
    try:
        out = _post_reward(endpoint, {"prompt": "p", "completions": ["a"]})
        assert out["rewards"] == [0.0]
        assert bench.calls == []
    finally:
        server.stop()


def test_sandbox_mode_unchanged():
    server = RewardServer(DATASET, mode="math", scale=1.0)
    endpoint = server.start()
    try:
        out = _post_reward(
            endpoint, {"prompt": "what is 2+2", "completions": ["<answer>4</answer>"]}
        )
        assert out["rewards"] == [1.0]
    finally:
        server.stop()


def test_monitor_fed_by_bench_batches():
    from fusion_trainer.hacking_monitor import HackingMonitor

    monitor = HackingMonitor()
    bench = FakeBenchClient(score=0.8)
    server = RewardServer(
        DATASET, mode="bench", scale=1.0, bench_client=bench, monitor=monitor
    )
    endpoint = server.start()
    try:
        _post_reward(
            endpoint, {"prompt": "what is 2+2", "completions": ["4", "4"]}
        )
        snap = monitor.snapshot()
        assert snap["reward_points"] == 1
        assert snap["format_points"] == 1
    finally:
        server.stop()
