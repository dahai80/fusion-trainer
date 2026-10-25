"""Tests for the benchmark-based checkpoint selection loop (issue #75)."""

import json
from pathlib import Path

import httpx
import pytest

from fusion_trainer.config import RLSLConfig, SFTConfig, TrainerConfig, URLGuardError
from fusion_trainer.dataset.loader import DatasetLoader
from fusion_trainer.eval_loop import (
    BENCH_TASKS_PATH,
    DEFAULT_BENCH_URL,
    SELECTION_FILENAME,
    BenchClient,
    BenchError,
    list_checkpoints,
    nearest_val_loss,
    run_checkpoint_eval,
    select_best_checkpoint,
    stage_checkpoint_for_serving,
)

# ---------------------------------------------------------------------------
# list_checkpoints / nearest_val_loss / select_best_checkpoint
# ---------------------------------------------------------------------------


def _mk_checkpoints(tmp_path: Path, steps: list[int]):
    for s in steps:
        (tmp_path / f"{s:08d}_adapters.safetensors").write_bytes(b"x" * 16)
    (tmp_path / "adapters.safetensors").write_bytes(b"final")
    (tmp_path / "garbage.txt").write_text("nope")


def test_list_checkpoints_filters_and_sorts(tmp_path):
    _mk_checkpoints(tmp_path, [500, 100])
    cks = list_checkpoints(tmp_path)
    assert [c["step"] for c in cks] == [100, 500]
    assert all(c["path"].endswith("_adapters.safetensors") for c in cks)


def test_list_checkpoints_empty(tmp_path):
    assert list_checkpoints(tmp_path) == []


def test_nearest_val_loss():
    vl = [(100, 1.0), (300, 0.8), (500, 0.9)]
    assert nearest_val_loss(vl, 100) == 1.0
    assert nearest_val_loss(vl, 400) == 0.8
    assert nearest_val_loss(vl, 10**9) == 0.9
    assert nearest_val_loss([], 100) is None


def test_select_best_bench_score_wins():
    rows = [
        {"step": 100, "val_loss": 0.5, "bench_score": 0.4},
        {"step": 200, "val_loss": 0.9, "bench_score": 0.7},
        {"step": 300, "val_loss": 0.4, "bench_score": None},
    ]
    best, basis = select_best_checkpoint(rows)
    assert best["step"] == 200
    assert basis == "bench"


def test_select_best_val_loss_fallback_without_bench():
    rows = [
        {"step": 100, "val_loss": 0.6},
        {"step": 200, "val_loss": 0.4},
        {"step": 300, "val_loss": 0.5},
    ]
    best, basis = select_best_checkpoint(rows)
    assert best["step"] == 200
    assert basis == "val_loss"


def test_select_best_tie_prefers_later_step():
    rows = [
        {"step": 100, "val_loss": 0.4},
        {"step": 300, "val_loss": 0.4},
    ]
    best, basis = select_best_checkpoint(rows)
    assert best["step"] == 300
    assert basis == "val_loss"


def test_select_best_no_signal_takes_last():
    rows = [{"step": 100, "val_loss": None}, {"step": 200, "val_loss": None}]
    best, basis = select_best_checkpoint(rows)
    assert best["step"] == 200
    assert basis == "last_checkpoint"


def test_select_best_empty_raises():
    with pytest.raises(BenchError):
        select_best_checkpoint([])


# ---------------------------------------------------------------------------
# BenchClient (mocked HTTP, _Fake pattern from test_voice_trainer)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class _FakeHttp:
    """Capture httpx.post/get calls and replay canned responses."""

    def __init__(self):
        self.post_calls: list[dict] = []
        self.get_calls: list[dict] = []
        self._post_responses: list[_FakeResponse] = []
        self._get_responses: list[_FakeResponse] = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.post_calls.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self._post_responses.pop(0)

    def get(self, url, headers=None, timeout=None):
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self._get_responses.pop(0)


@pytest.fixture
def fake_http(monkeypatch):
    fake = _FakeHttp()
    monkeypatch.setattr(httpx, "post", fake.post)
    monkeypatch.setattr(httpx, "get", fake.get)
    return fake


def test_bench_client_default_url_and_headers(fake_http):
    client = BenchClient(api_key="jwt-token", tenant_id="default")
    assert client._base == DEFAULT_BENCH_URL
    assert client._headers["X-Tenant-Id"] == "default"
    assert client._headers["Authorization"] == "Bearer jwt-token"


def test_bench_client_env_url(monkeypatch, fake_http):
    monkeypatch.setenv("FUSION_BENCH_URL", "http://localhost:9999")
    client = BenchClient()
    assert client._base == "http://localhost:9999"


def test_bench_client_rejects_numeric_loopback():
    with pytest.raises(URLGuardError):
        BenchClient(base_url="http://127.0.0.1:11467")


def test_bench_client_rejects_bad_scheme():
    with pytest.raises(URLGuardError):
        BenchClient(base_url="ftp://localhost:11467")


def test_create_task_contract(fake_http):
    fake = fake_http
    fake._post_responses.append(_FakeResponse(json_data={"task_id": "t1", "status": "pending"}))
    client = BenchClient()
    task = client.create_task(model="qwen", executor_key="speed", suite="s1")
    call = fake.post_calls[0]
    assert call["url"] == f"{DEFAULT_BENCH_URL}{BENCH_TASKS_PATH}"
    assert call["json"]["model"] == "qwen"
    assert call["json"]["executor_key"] == "speed"
    assert call["json"]["suite"] == "s1"
    assert call["json"]["params"] == {}
    assert task["task_id"] == "t1"


def test_create_task_server_error_raises(fake_http):
    fake_http._post_responses.append(
        _FakeResponse(status_code=401, text="unauthorized")
    )
    client = BenchClient(api_key="bad")
    with pytest.raises(BenchError, match="401"):
        client.create_task(model="qwen")


def test_create_task_missing_task_id_raises(fake_http):
    fake_http._post_responses.append(_FakeResponse(json_data={"status": "pending"}))
    with pytest.raises(BenchError, match="no task_id"):
        BenchClient().create_task(model="qwen")


def test_wait_for_task_polls_until_terminal(fake_http, monkeypatch):
    fake = fake_http
    fake._get_responses.extend(
        [
            _FakeResponse(json_data={"task_id": "t1", "status": "pending"}),
            _FakeResponse(json_data={"task_id": "t1", "status": "running"}),
            _FakeResponse(
                json_data={
                    "task_id": "t1",
                    "status": "completed",
                    "result": {
                        "metric_name": "pass_rate",
                        "metric_value": 12.3,
                        "pass_rate": 0.75,
                    },
                }
            ),
        ]
    )
    monkeypatch.setattr("fusion_trainer.eval_loop.time.sleep", lambda s: None)
    task = BenchClient().wait_for_task("t1", poll_interval=0)
    assert task["status"] == "completed"
    assert fake.get_calls[-1]["url"].endswith(f"{BENCH_TASKS_PATH}/t1")


def test_wait_for_task_404_raises_fast(fake_http, monkeypatch):
    fake = fake_http
    fake._get_responses.append(_FakeResponse(status_code=404, text="gone"))
    with pytest.raises(BenchError, match="404"):
        BenchClient().wait_for_task("t1", poll_interval=0)


def test_wait_for_task_transient_error_retries(fake_http, monkeypatch):
    fake = fake_http
    fake._get_responses.extend(
        [
            _FakeResponse(status_code=503, text="boom"),
            _FakeResponse(json_data={"task_id": "t1", "status": "completed"}),
        ]
    )
    monkeypatch.setattr("fusion_trainer.eval_loop.time.sleep", lambda s: None)
    task = BenchClient().wait_for_task("t1", poll_interval=0)
    assert task["status"] == "completed"


def test_run_benchmark_extracts_score(fake_http, monkeypatch):
    fake = fake_http
    fake._post_responses.append(_FakeResponse(json_data={"task_id": "t9"}))
    fake._get_responses.append(
        _FakeResponse(
            json_data={
                "task_id": "t9",
                "status": "completed",
                "result": {"metric_name": "pass_rate", "pass_rate": 0.9},
            }
        )
    )
    monkeypatch.setattr("fusion_trainer.eval_loop.time.sleep", lambda s: None)
    out = BenchClient().run_benchmark(model="qwen", adapter_path="/tmp/a")
    assert out["status"] == "completed"
    assert out["score"] == 0.9
    assert fake.post_calls[0]["json"]["params"]["adapter_path"] == "/tmp/a"


def test_run_benchmark_failed_task_no_score(fake_http, monkeypatch):
    fake = fake_http
    fake._post_responses.append(_FakeResponse(json_data={"task_id": "t1"}))
    fake._get_responses.append(
        _FakeResponse(
            json_data={
                "task_id": "t1",
                "status": "failed",
                "result": {"errors": ["timeout"]},
            }
        )
    )
    monkeypatch.setattr("fusion_trainer.eval_loop.time.sleep", lambda s: None)
    out = BenchClient().run_benchmark(model="qwen")
    assert out["status"] == "failed"
    assert out["score"] is None
    assert out["errors"] == ["timeout"]


# ---------------------------------------------------------------------------
# run_checkpoint_eval end-to-end (mocked bench)
# ---------------------------------------------------------------------------


class _FakeBench:
    def __init__(self, scores: dict[int, float] | None = None, fail=False):
        self.scores = scores or {}
        self.fail = fail
        self.calls: list[str] = []

    def run_benchmark(self, model, adapter_path="", executor_key="speed", **kw):
        self.calls.append(adapter_path)
        if self.fail:
            raise BenchError("bench POST transport error: down")
        step = int(Path(adapter_path).name.rsplit("eval_stage_", 1)[-1])
        return {
            "status": "completed",
            "task_id": f"t-{step}",
            "metric_name": "pass_rate",
            "metric_value": self.scores.get(step),
            "pass_rate": self.scores.get(step),
            "score": self.scores.get(step),
            "errors": [],
        }


def test_run_checkpoint_eval_selects_by_bench(tmp_path):
    _mk_checkpoints(tmp_path, [100, 200])
    bench = _FakeBench(scores={100: 0.6, 200: 0.8})
    out = run_checkpoint_eval(
        model="qwen",
        adapter_dir=tmp_path,
        val_losses=[(100, 0.5), (200, 0.4)],
        bench=bench,
        poll_interval=0,
    )
    assert out["basis"] == "bench"
    assert out["chosen"]["step"] == 200
    assert out["bench_status"] == "ok"
    # every checkpoint staged + benchmarked
    assert len(bench.calls) == 2
    # table persisted atomically
    persisted = json.loads((tmp_path / SELECTION_FILENAME).read_text())
    assert persisted["basis"] == "bench"
    assert persisted["chosen"]["step"] == 200
    assert [r["step"] for r in persisted["table"]] == [100, 200]


def test_run_checkpoint_eval_bench_outage_falls_back(tmp_path):
    _mk_checkpoints(tmp_path, [100, 200])
    bench = _FakeBench(fail=True)
    out = run_checkpoint_eval(
        model="qwen",
        adapter_dir=tmp_path,
        val_losses=[(100, 0.5), (200, 0.4)],
        bench=bench,
        poll_interval=0,
    )
    assert out["basis"] == "val_loss"
    assert out["bench_status"] == "unavailable"
    assert out["fallback_reason"]
    assert out["chosen"]["step"] == 200  # lower val_loss
    # fallback recorded in the persisted table, not silent
    persisted = json.loads((tmp_path / SELECTION_FILENAME).read_text())
    assert all(r["bench_status"] == "unavailable" for r in persisted["table"])


def test_run_checkpoint_eval_no_bench_uses_val_loss(tmp_path):
    _mk_checkpoints(tmp_path, [100, 200, 300])
    out = run_checkpoint_eval(
        model="qwen",
        adapter_dir=tmp_path,
        val_losses=[(100, 0.5), (300, 0.3)],
        bench=None,
    )
    assert out["basis"] == "val_loss"
    assert out["chosen"]["step"] == 300
    assert out["bench_status"] == "not_configured"


def test_run_checkpoint_eval_single_checkpoint_table(tmp_path):
    (tmp_path / "adapters.safetensors").write_bytes(b"final")
    out = run_checkpoint_eval(
        model="qwen", adapter_dir=tmp_path, val_losses=[(100, 0.5)], bench=None
    )
    assert out["basis"] == "single_checkpoint"
    assert len(out["table"]) == 1
    assert out["chosen"]["checkpoint"] == "final"


def test_run_checkpoint_eval_never_raises_on_persist_error(tmp_path, monkeypatch):
    _mk_checkpoints(tmp_path, [100])
    monkeypatch.setattr(
        "fusion_trainer.eval_loop.Path.write_text",
        lambda self, *a, **kw: (_ for _ in ()).throw(OSError("disk full")),
    )
    out = run_checkpoint_eval(model="qwen", adapter_dir=tmp_path, bench=None)
    assert out["basis"] in ("val_loss", "last_checkpoint")


def test_stage_checkpoint_for_serving(tmp_path):
    src = tmp_path / "00100_adapters.safetensors"
    src.write_bytes(b"weights")
    stage = tmp_path / "stage"
    served = stage_checkpoint_for_serving(src, stage)
    assert (Path(served) / "adapters.safetensors").read_bytes() == b"weights"
    assert src.read_bytes() == b"weights"  # source untouched


# ---------------------------------------------------------------------------
# config wiring (issue #75: grpo_backend alias, eval fields)
# ---------------------------------------------------------------------------


def test_grpo_backend_policy_gradient_alias_normalized():
    cfg = TrainerConfig(rlsl=RLSLConfig(grpo_backend="policy_gradient"))
    assert cfg.rlsl.grpo_backend == "grpo"


def test_grpo_backend_rft_default_untouched():
    cfg = TrainerConfig()
    assert cfg.rlsl.grpo_backend == "rft"


def test_sft_config_eval_fields_defaults():
    c = SFTConfig()
    assert c.eval_select is False
    assert c.bench_url == ""
    assert c.bench_executor == "speed"
    assert c.bench_task_timeout == 1800.0


# ---------------------------------------------------------------------------
# loader conversation-level char budget (issue #75)
# ---------------------------------------------------------------------------


def _loader(tmp_path):
    return DatasetLoader(), tmp_path


def test_load_messages_short_conversation_untouched(tmp_path):
    loader, _ = _loader(tmp_path)
    f = tmp_path / "chat.jsonl"
    f.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]})
        + "\n"
    )
    train, val = loader.load_messages(path=str(f), val_ratio=0.0, max_seq_length=4096)
    assert len(train) == 1
    assert train[0]["messages"][0]["content"] == "hello"


def test_load_messages_budget_is_conversation_level(tmp_path):
    # issue #75: the old per-message cap truncated EVERY message at
    # max_seq_length CHARS (4096 tokens treated as 4096 chars = 4x
    # over-truncation). A 6000-char single message under a 4096-token
    # budget (16384 chars) must now survive intact.
    loader, _ = _loader(tmp_path)
    long = "x" * 6000
    f = tmp_path / "chat.jsonl"
    f.write_text(
        json.dumps({"messages": [{"role": "user", "content": long}]}) + "\n"
    )
    train, _ = loader.load_messages(path=str(f), val_ratio=0.0, max_seq_length=4096)
    assert train[0]["messages"][0]["content"] == long


def test_load_messages_trims_trailing_turns_on_budget(tmp_path):
    # two turns far over budget: only the trailing assistant answer fits
    loader, _ = _loader(tmp_path)
    f = tmp_path / "chat.jsonl"
    f.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "u" * 9000},
                    {"role": "assistant", "content": "a" * 1000},
                ]
            }
        )
        + "\n"
    )
    # max_seq_length 1024 -> budget 4096 chars
    train, _ = loader.load_messages(path=str(f), val_ratio=0.0, max_seq_length=1024)
    msgs = train[0]["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"
    assert len(msgs[0]["content"]) == 1000


def test_load_messages_keeps_last_message_when_nothing_fits(tmp_path):
    loader, _ = _loader(tmp_path)
    f = tmp_path / "chat.jsonl"
    f.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "u" * 9000}]}
        )
        + "\n"
    )
    train, _ = loader.load_messages(path=str(f), val_ratio=0.0, max_seq_length=1024)
    msgs = train[0]["messages"]
    assert len(msgs) == 1
    assert len(msgs[0]["content"]) == 4096
