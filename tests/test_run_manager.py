"""Tests for RunManager — async run lifecycle + live progress surface.

Targets: run_manager.py
Mocked MLXFineTuneClient (real training needs fusion-mlx up). Cleans up
temp dirs after each test (keep only final outputs + logs).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

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
            assert all(ev.get("step", 0) > 0 for ev in prog["events"])

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
            adapters = rm.list_adapters(model="test-model")
            assert adapters and adapters[0]["name"] == "a1"

    def test_delete_adapter_unsupported(self):
        with tempfile.TemporaryDirectory() as tmp:
            rm = RunManager(cache_path=tmp, model_hub_path=tmp, mlx_port=11999)
            res = rm.delete_adapter("any")
            assert res["ok"] is False
            assert "error" in res
