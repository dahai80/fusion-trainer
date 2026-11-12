"""Tests for RunManager — async run lifecycle + live progress surface.

Targets: run_manager.py
Mocked MLXFineTuneClient (real training needs fusion-mlx up). Cleans up
temp dirs after each test (keep only final outputs + logs).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
from fusion_trainer.run_manager import RunManager, config_to_dict


def _make_dataset_file(tmpdir: str, n: int = 4) -> str:
    p = Path(tmpdir) / "train.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"input": f"Q{i}", "output": f"A{i}"}) + "\n")
    return str(p)


def _make_sft_config(
    cache_path: str, model: str = "test-model", dataset_path: str = ""
) -> TrainerConfig:
    if not dataset_path:
        dataset_path = _make_dataset_file(cache_path)
    return TrainerConfig(
        dataset=DatasetConfig(path=dataset_path, max_seq_length=128),
        sft=SFTConfig(
            model=model,
            method="lora",
            lora_rank=16,
            lora_alpha=32.0,
            run_name="rm_test_sft",
            max_steps=2,
        ),
        cache_path=cache_path,
    )


def _patch_sft_client(monkeypatch, events=None, final_status="completed"):
    """Patch MLXFineTuneClient so SFT runs without a server."""
    events = events or [
        {
            "type": "train_loss",
            "step": 1,
            "train_loss": 2.3,
            "learning_rate": 2e-5,
            "tokens_per_second": 100,
            "it_sec": 1.0,
            "peak_memory_gb": 5.0,
        },
        {
            "type": "train_loss",
            "step": 2,
            "train_loss": 1.1,
            "learning_rate": 2e-5,
            "tokens_per_second": 100,
            "it_sec": 1.0,
            "peak_memory_gb": 5.0,
        },
    ]

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

        def create_job(self, model_id, dataset, config=None, adapter_name=""):
            return {"job_id": "fake-job-123", "status": "queued"}

        def wait_for_job(self, job_id, on_progress=None, **kw):
            for ev in events:
                if on_progress is not None:
                    on_progress({"status": "running", "events": [ev]}, [ev])
            return {
                "status": final_status,
                "adapter_path": "/fake/adapters/m/adapter",
                "events": events,
                "error": "" if final_status == "completed" else "boom",
            }

        def health(self):
            return {"data": [{"id": "test-model"}]}

        def list_adapters(self, model_id=""):
            return [{"name": "a1", "model": "test-model"}]

    import fusion_trainer.mlx_client as mlx_mod
    import fusion_trainer.run_manager as rm_mod
    import fusion_trainer.sft.trainer as tr_mod

    _real_stage = mlx_mod.MLXFineTuneClient.stage_dataset
    monkeypatch.setattr(mlx_mod, "MLXFineTuneClient", _FakeClient)
    monkeypatch.setattr(tr_mod, "MLXFineTuneClient", _FakeClient, raising=False)
    monkeypatch.setattr(rm_mod, "MLXFineTuneClient", _FakeClient, raising=False)
    _FakeClient.stage_dataset = staticmethod(_real_stage)


def _wait_terminal(rm: RunManager, run_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = rm.run_status(run_id)
        if st["status"] in ("completed", "failed", "cancelled"):
            return st
        time.sleep(0.05)
    return rm.run_status(run_id)


# ===========================================================================
# info / presets / config_to_dict
# ===========================================================================


class TestInfoPresets:
    def test_list_presets_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            presets = rm.list_presets(kind="sft")
            assert len(presets) > 0
            assert all(p["kind"] == "sft" for p in presets)
            assert "memory_estimate_gb" in presets[0]

    def test_list_presets_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            presets = rm.list_presets()
            kinds = {p["kind"] for p in presets}
            assert "sft" in kinds and "rlsl" in kinds

    def test_info_unreachable_mlx(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            info = rm.info()
            assert info["version"]
            assert info["mlx_port"] == 11999
            assert info["mlx_alive"] is False
            assert any(w["id"] == "mxfp8" for w in info["warnings"])

    def test_config_to_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_sft_config(tmp)
            d = config_to_dict(cfg)
            assert d["sft"]["model"] == "test-model"
            assert "dataset" in d and "rlsl" in d


# ===========================================================================
# SFT run lifecycle + live progress
# ===========================================================================


class TestSFTRun:
    def test_start_and_complete(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            start = rm.start_sft(cfg)
            assert start["status"] == "running"
            run_id = start["run_id"]
            st = _wait_terminal(rm, run_id)
            assert st["status"] == "completed"
            assert st["adapter_name"] != ""

    def test_progress_streams_train_loss(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            _wait_terminal(rm, run_id)
            st = rm.run_status(run_id)
            assert st["metrics"].get("train_loss") == 1.1
            prog = rm.run_progress(run_id, since_step=0)
            assert len(prog["events"]) >= 1
            # R8: step-less events (job/status) are passed through by design;
            # only train_loss metric events carry a step, and those must be > 0.
            loss_events = [
                ev for ev in prog["events"] if ev.get("type") == "train_loss"
            ]
            assert loss_events, "expected at least one train_loss event"
            assert all(ev.get("step", 0) > 0 for ev in loss_events)

    def test_events_jsonl_written(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            _wait_terminal(rm, run_id)
            events_path = Path(tmp) / "runs" / run_id / "events.jsonl"
            assert events_path.exists()
            lines = events_path.read_text().strip().split("\n")
            assert any(json.loads(l).get("type") == "train_loss" for l in lines)

    def test_failed_run_visible(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch, final_status="failed")
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            st = _wait_terminal(rm, run_id)
            assert st["status"] == "failed"
            assert st["error"]

    def test_list_runs_records(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            _wait_terminal(rm, run_id)
            runs = rm.list_runs()
            assert any(r["run_id"] == run_id for r in runs)
            rec = next(r for r in runs if r["run_id"] == run_id)
            assert rec["status"] == "completed"


# ===========================================================================
# Persistence / rehydration
# ===========================================================================


class TestRehydrate:
    def test_terminal_run_reloaded(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            _wait_terminal(rm, run_id)
            rm2 = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            st = rm2.run_status(run_id)
            assert st["status"] == "completed"

    def test_zombie_marked_failed(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            run_dir = Path(tmp) / "runs" / "sft_zombie"
            run_dir.mkdir(parents=True)
            (run_dir / "run_meta.json").write_text(
                json.dumps(
                    {
                        "run_id": "sft_zombie",
                        "status": "running",
                        "created": 1.0,
                        "method": "sft",
                    }
                )
            )
            rm2 = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            st = rm2.run_status("sft_zombie")
            assert st["status"] == "failed"

    def test_meta_persisted_at_start(self, monkeypatch):
        # A4: run_meta.json must exist with status:running BEFORE the worker
        # thread completes, so a mid-run crash leaves a recoverable record.
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            meta_path = Path(tmp) / "runs" / run_id / "run_meta.json"
            # Written at start, before the (mocked) trainer completes.
            assert meta_path.exists()
            meta = json.loads(meta_path.read_text())
            assert meta["status"] == "running"
            _wait_terminal(rm, run_id)

    def test_reconstruct_meta_from_events_no_meta(self, monkeypatch):
        # A4: a run that crashed before terminal persist (no run_meta.json)
        # but with events.jsonl present must be reconstructed + marked failed
        # (was silently skipped → zombie-cancel dead code).
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "sft_crashed"
            run_dir.mkdir(parents=True)
            (run_dir / "events.jsonl").write_text(
                json.dumps({"type": "job", "job_id": "upstream-999", "kind": "sft"})
                + "\n"
                + json.dumps({"type": "train_loss", "step": 3, "train_loss": 1.5})
                + "\n"
            )
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            st = rm.run_status("sft_crashed")
            assert st["status"] == "failed"
            # job_id scavenged from events so zombie-cancel can target it.
            assert st["job_id"] == "upstream-999"

    def test_run_id_collision_proof(self, monkeypatch):
        # R5: two runs started in rapid succession must not collide on run_id.
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            id1 = rm.start_sft(cfg)["run_id"]
            id2 = rm.start_sft(cfg)["run_id"]
            assert id1 != id2
            # Non-daemon workers must reach terminal before the tmp dir is
            # torn down, else rmtree races the still-writing thread.
            _wait_terminal(rm, id1)
            _wait_terminal(rm, id2)


# ===========================================================================
# Datasets / adapters
# ===========================================================================


class TestDatasetsAdapters:
    def test_list_datasets_empty_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            assert rm.list_datasets() == []

    def test_list_datasets_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "datasets" / "sft"
            hub.mkdir(parents=True)
            (hub / "code_alpaca.jsonl").write_text(
                json.dumps({"input": "q", "output": "a"}) + "\n"
            )
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            ds = rm.list_datasets()
            assert any(d["name"] == "sft/code_alpaca" for d in ds)
            # P1: line-counted jsonl row count is correct.
            assert ds[0]["n_samples"] == 1

    def test_list_datasets_cached_by_mtime(self):
        # P1: a second list_datasets call must hit the cache, not re-read.
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "datasets" / "sft"
            hub.mkdir(parents=True)
            f = hub / "code_alpaca.jsonl"
            f.write_text(
                json.dumps({"input": "q", "output": "a"})
                + "\n"
                + json.dumps({"input": "q2", "output": "a2"})
                + "\n"
            )
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            first = rm.list_datasets()
            assert first[0]["n_samples"] == 2
            # Corrupt the file, then RESTORE its original (mtime, size) so the
            # cache key still matches — proving the second call serves cache
            # and does not re-read the now-garbage file.
            orig_stat = f.stat()
            f.write_text("x" * orig_stat.st_size)
            os.utime(f, (orig_stat.st_atime, orig_stat.st_mtime))
            second = rm.list_datasets()
            assert second[0]["n_samples"] == 2  # cached value, not re-read

    def test_list_datasets_parquet_row_count(self):
        # P1: parquet counted via pyarrow (was counted as text lines = wrong).
        pytest.importorskip("pyarrow")
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "datasets" / "sft"
            hub.mkdir(parents=True)
            table = pa.table(
                {"input": ["q1", "q2", "q3"], "output": ["a1", "a2", "a3"]}
            )
            pq.write_table(table, str(hub / "pq.parquet"))
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            ds = rm.list_datasets()
            ds = [d for d in ds if d["format"] == "parquet"]
            assert ds and ds[0]["n_samples"] == 3

    def test_preview_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            hub = Path(tmp) / "datasets" / "sft"
            hub.mkdir(parents=True)
            (hub / "code_alpaca.jsonl").write_text(
                json.dumps({"input": "q1", "output": "a1"})
                + "\n"
                + json.dumps({"input": "q2", "output": "a2"})
                + "\n"
            )
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            pv = rm.preview_dataset("sft/code_alpaca", limit=1)
            assert pv["n_total"] == 2
            assert len(pv["rows"]) == 1

    def test_list_adapters_proxies(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            res = rm.list_adapters(model="test-model")
            assert res["error"] is None
            assert res["items"] and res["items"][0]["name"] == "a1"

    def test_list_adapters_down_returns_error(self):
        # P2-6: fusion-mlx unreachable -> explicit error carrier, not empty list.
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            res = rm.list_adapters()
            assert res["items"] == []
            assert res["error"] is not None

    def test_delete_adapter_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            res = rm.delete_adapter("any")
            assert res["ok"] is False
            assert "error" in res


class TestRunRecordFields:
    def test_completed_run_reports_full_steps_and_model(self, monkeypatch):
        # Tiny runs finish before the first steps_per_report boundary —
        # a completed run must show step == total_steps (not 0/N), and
        # list_runs must carry the model for the GUI run table.
        with tempfile.TemporaryDirectory() as tmp:
            _patch_sft_client(monkeypatch)
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp)
            run_id = rm.start_sft(cfg)["run_id"]
            _wait_terminal(rm, run_id)
            rec = next(r for r in rm.list_runs() if r["run_id"] == run_id)
            assert rec["status"] == "completed"
            assert rec["model"] == cfg.sft.model
            assert rec["step"] == rec["total_steps"] > 0


# ===========================================================================
# T2 (capability plan): cross-process run lock + priority queue
# ===========================================================================


class TestRunLock:
    def test_exclusive_and_crash_safe(self, tmp_path):
        from fusion_trainer.run_queue import RunLock

        lock_path = tmp_path / "run.lock"
        a = RunLock(lock_path)
        b = RunLock(lock_path)
        assert a.acquire(timeout=1.0)
        assert b.acquire(timeout=0.3) is False  # busy
        a.release()
        assert b.acquire(timeout=1.0)
        b.release()

    def test_holder_info(self, tmp_path):
        from fusion_trainer.run_queue import RunLock

        lock = RunLock(tmp_path / "run.lock")
        with lock:
            holder = lock.holder()
            assert holder.get("pid") == os.getpid()
        assert lock.holder() == {} or "pid" in lock.holder()

    def test_release_idempotent(self, tmp_path):
        from fusion_trainer.run_queue import RunLock

        lock = RunLock(tmp_path / "run.lock")
        lock.release()  # no acquire yet — must not raise
        assert lock.acquire(timeout=1.0)
        lock.release()
        lock.release()


class TestRunQueue:
    def test_second_run_queues_then_runs(self, monkeypatch, tmp_path):
        _patch_sft_client(monkeypatch)
        import fusion_trainer.run_manager as rm_mod

        release_a = threading.Event()

        def _slow_wait(job_id, on_progress=None, **kw):
            release_a.wait(timeout=15)
            return {
                "status": "completed",
                "adapter_path": "/fake/a",
                "events": [],
                "error": "",
            }

        # patch the fake client's wait_for_job for the first run only via
        # run-order: run A uses slow waiter, run B uses the default one.
        calls = {"n": 0}
        orig_wait = rm_mod.MLXFineTuneClient.wait_for_job

        def _ordered_wait(self, job_id, on_progress=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _slow_wait(job_id, on_progress, **kw)
            return orig_wait(self, job_id, on_progress, **kw)

        monkeypatch.setattr(rm_mod.MLXFineTuneClient, "wait_for_job", _ordered_wait)

        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg_a = _make_sft_config(tmp, dataset_path=_make_dataset_file(tmp))
            cfg_b = _make_sft_config(tmp, dataset_path=_make_dataset_file(tmp))
            ra = rm.start_sft(cfg_a)
            assert ra["status"] == "running"
            rb = rm.start_sft(cfg_b)
            assert rb["status"] == "queued"
            assert rb["queue_position"] == 1
            assert rm.run_status(rb["run_id"])["status"] == "queued"
            release_a.set()
            _wait_terminal(rm, ra["run_id"])
            _wait_terminal(rm, rb["run_id"])
            assert rm.run_status(ra["run_id"])["status"] == "completed"
            assert rm.run_status(rb["run_id"])["status"] == "completed"

    def test_priority_order(self, monkeypatch, tmp_path):
        _patch_sft_client(monkeypatch)
        order = []
        import fusion_trainer.run_manager as rm_mod

        release = threading.Event()

        def _slow_wait(self, job_id, on_progress=None, **kw):
            order.append("first")
            release.wait(timeout=15)
            return {"status": "completed", "adapter_path": "/x", "events": [], "error": ""}

        monkeypatch.setattr(rm_mod.MLXFineTuneClient, "wait_for_job", _slow_wait)

        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp, dataset_path=_make_dataset_file(tmp))
            first = rm.start_sft(cfg)
            hi = rm.start_sft(cfg, priority=1)
            lo = rm.start_sft(cfg, priority=9)
            # stop the low-priority one while queued — high-priority must run
            rm.stop_run(lo["run_id"])
            assert rm.run_status(lo["run_id"])["status"] == "cancelled"
            release.set()
            _wait_terminal(rm, first["run_id"])
            _wait_terminal(rm, hi["run_id"])
            assert rm.run_status(hi["run_id"])["status"] == "completed"

    def test_stop_queued_never_spawns(self, monkeypatch, tmp_path):
        _patch_sft_client(monkeypatch)
        import fusion_trainer.run_manager as rm_mod

        release = threading.Event()

        def _slow_wait(job_id, on_progress=None, **kw):
            release.wait(timeout=15)
            return {"status": "completed", "adapter_path": "/x", "events": [], "error": ""}

        monkeypatch.setattr(rm_mod.MLXFineTuneClient, "wait_for_job", _slow_wait)

        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            cfg = _make_sft_config(tmp, dataset_path=_make_dataset_file(tmp))
            first = rm.start_sft(cfg)
            second = rm.start_sft(cfg)
            stopped = rm.stop_run(second["run_id"])
            assert stopped["status"] == "cancelled"
            release.set()
            _wait_terminal(rm, first["run_id"])
            time.sleep(0.3)
            # cancelled-while-queued run must stay terminal
            assert rm.run_status(second["run_id"])["status"] == "cancelled"
            assert rm._pending == []

    def test_info_reports_queue(self, tmp_path):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            info = rm.info()
            assert info["queue"]["lock_path"].endswith("run.lock")
            assert info["queue"]["pending"] == 0
