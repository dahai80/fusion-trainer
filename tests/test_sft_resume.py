"""Tests for checkpoint resume + engine-restart detection (issue #65).

find_latest_checkpoint, _resolve_resume staging, iters subtraction,
job_state persistence, and 404 fast-fail in wait_for_job.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError
from fusion_trainer.sft.trainer import (
    MLX_ADAPTER_BASE_DIR,
    SFTTrainer,
    find_latest_checkpoint,
)


def _make_checkpoint(adapter_dir: Path, step: int) -> Path:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    p = adapter_dir / f"{step:07d}_adapters.safetensors"
    p.write_bytes(b"fake")
    return p


def _make_trainer(tmp_path: Path, **sft_kwargs) -> SFTTrainer:
    cfg = TrainerConfig(
        cache_path=str(tmp_path / "cache"),
        dataset=DatasetConfig(path=str(tmp_path / "data.jsonl")),
        sft=SFTConfig(model="test-model", run_name="run_a", **sft_kwargs),
    )
    return SFTTrainer(cfg)


class TestFindLatestCheckpoint:
    def test_returns_highest_step(self, tmp_path):
        _make_checkpoint(tmp_path, 300)
        _make_checkpoint(tmp_path, 700)
        _make_checkpoint(tmp_path, 100)
        ckpt = find_latest_checkpoint(tmp_path)
        assert ckpt is not None
        assert ckpt[1] == 700
        assert ckpt[0].name == "0000700_adapters.safetensors"

    def test_empty_dir_returns_none(self, tmp_path):
        assert find_latest_checkpoint(tmp_path) is None

    def test_ignores_non_numbered_files(self, tmp_path):
        (tmp_path / "adapters.safetensors").write_bytes(b"x")
        (tmp_path / "junk_adapters.safetensors").write_bytes(b"x")
        assert find_latest_checkpoint(tmp_path) is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert find_latest_checkpoint(tmp_path / "nope") is None


class TestMLXFineTuneErrorStatusCode:
    def test_default_none(self):
        assert MLXFineTuneError("boom").status_code is None

    def test_carries_status(self):
        exc = MLXFineTuneError("404", status_code=404)
        assert exc.status_code == 404

    def test_request_attaches_status_on_4xx(self):
        client = MLXFineTuneClient(port=11434)
        resp = MagicMock()
        resp.status_code = 404
        resp.text = "not found"
        client._http = MagicMock()
        client._http.request.return_value = resp
        with pytest.raises(MLXFineTuneError) as exc_info:
            client.get_job("gone")
        assert exc_info.value.status_code == 404
        client.close()


class TestWaitForJob404:
    def _client(self):
        return MLXFineTuneClient(port=11434)

    def test_404_raises_immediately(self):
        client = self._client()
        polls = []

        def fake_get(job_id):
            polls.append(job_id)
            raise MLXFineTuneError("GET failed 404: gone", status_code=404)

        client.get_job = fake_get
        with pytest.raises(MLXFineTuneError) as exc_info:
            client.wait_for_job("j1", poll_interval=0.01, timeout=5.0)
        assert exc_info.value.status_code == 404
        assert len(polls) == 1

    def test_transient_error_keeps_polling(self):
        client = self._client()
        calls = {"n": 0}

        def fake_get(job_id):
            calls["n"] += 1
            if calls["n"] < 3:
                raise MLXFineTuneError("connect blip")
            return {"status": "completed", "events": []}

        client.get_job = fake_get
        job = client.wait_for_job("j1", poll_interval=0.01, timeout=5.0)
        assert job["status"] == "completed"
        assert calls["n"] == 3


class TestResolveResume:
    def test_stages_newest_checkpoint(self, tmp_path):
        adapter_dir = tmp_path / "adapters" / "test-model" / "run_a"
        _make_checkpoint(adapter_dir, 300)
        _make_checkpoint(adapter_dir, 700)
        trainer = _make_trainer(
            tmp_path,
            init_adapter_path=str(adapter_dir),
            output_dir=str(tmp_path / "out"),
        )
        trainer._resolve_resume()
        assert trainer._resume_step == 700
        staged = Path(trainer._resume_dir) / "adapters.safetensors"
        assert staged.exists()

    def test_explicit_final_weights_dir(self, tmp_path):
        adapter_dir = tmp_path / "final"
        adapter_dir.mkdir()
        (adapter_dir / "adapters.safetensors").write_bytes(b"x")
        trainer = _make_trainer(
            tmp_path,
            init_adapter_path=str(adapter_dir),
            output_dir=str(tmp_path / "out"),
        )
        trainer._resolve_resume()
        assert trainer._resume_dir == str(adapter_dir)
        assert trainer._resume_step == 0

    def test_no_checkpoint_starts_fresh(self, tmp_path):
        adapter_dir = tmp_path / "empty"
        adapter_dir.mkdir()
        trainer = _make_trainer(
            tmp_path,
            init_adapter_path=str(adapter_dir),
            output_dir=str(tmp_path / "out"),
        )
        trainer._resolve_resume()
        assert trainer._resume_dir == ""
        assert trainer._resume_step == 0

    def test_unset_is_noop(self, tmp_path):
        trainer = _make_trainer(tmp_path, output_dir=str(tmp_path / "out"))
        trainer._resolve_resume()
        assert trainer._resume_dir == ""


class TestBuildMlxConfigResume:
    def test_iters_subtracted(self, tmp_path):
        trainer = _make_trainer(tmp_path, output_dir=str(tmp_path / "out"))
        trainer._train_samples = [{"messages": []} for _ in range(4)]
        trainer._resume_step = 700
        trainer._resume_dir = "/tmp/staged"
        cfg = trainer._build_mlx_config()
        # 4 samples / batch 1 * 3 epochs = 12; 12 - 700 clamps to 0
        assert cfg["iters"] == 0
        assert cfg["init_adapter_path"] == "/tmp/staged"

    def test_no_resume_leaves_iters(self, tmp_path):
        trainer = _make_trainer(tmp_path, output_dir=str(tmp_path / "out"))
        trainer._train_samples = [{"messages": []} for _ in range(4)]
        cfg = trainer._build_mlx_config()
        assert cfg["iters"] == 12
        assert "init_adapter_path" not in cfg


class TestPersistJobState:
    def test_writes_and_updates(self, tmp_path):
        out = tmp_path / "out"
        trainer = _make_trainer(tmp_path, output_dir=str(out))
        trainer._global_step = 42
        trainer._persist_job_state("running", "job123")
        p = out / "job_state.json"
        assert p.exists()
        state = json.loads(p.read_text())
        assert state["job_id"] == "job123"
        assert state["step"] == 42
        assert state["status"] == "running"
        assert state["adapter_dir"] == str(
            MLX_ADAPTER_BASE_DIR / "test-model" / "run_a"
        )
        # atomic tmp file cleaned up
        assert not (out / "job_state.json.tmp").exists()


class TestRunTrainingInterrupted:
    def test_404_yields_interrupted_result(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        trainer = _make_trainer(
            tmp_path, output_dir=str(out), num_epochs=1, batch_size=1
        )
        train_loader = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "hi2"}]},
        ]

        mock_client = MagicMock()
        mock_client.create_job.return_value = {"job_id": "joblost"}
        mock_client.wait_for_job.side_effect = MLXFineTuneError(
            "GET failed 404: job gone", status_code=404
        )
        mock_cls = MagicMock()
        mock_cls.stage_dataset.return_value = "staged/train.jsonl"
        mock_cls.return_value = mock_client
        monkeypatch.setattr("fusion_trainer.mlx_client.MLXFineTuneClient", mock_cls)

        result = trainer._run_training(train_loader, [])
        assert result["status"] == "interrupted"
        assert result["job_id"] == "joblost"
        assert result["resume_checkpoint"] == ""
        state = json.loads((out / "job_state.json").read_text())
        assert state["status"] == "interrupted"

    def test_resume_noop_when_already_trained(self, tmp_path, monkeypatch):
        out = tmp_path / "out"
        adapter_dir = tmp_path / "adapters" / "test-model" / "run_a"
        _make_checkpoint(adapter_dir, 10)
        trainer = _make_trainer(
            tmp_path,
            output_dir=str(out),
            num_epochs=1,
            batch_size=1,
            init_adapter_path=str(adapter_dir),
        )
        train_loader = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "hi2"}]},
        ]

        mock_client = MagicMock()
        mock_cls = MagicMock()
        mock_cls.stage_dataset.return_value = "staged/train.jsonl"
        mock_cls.return_value = mock_client
        monkeypatch.setattr("fusion_trainer.mlx_client.MLXFineTuneClient", mock_cls)

        result = trainer._run_training(train_loader, [])
        assert result["status"] == "completed"
        assert result["resumed_noop"] is True
        # no job submitted
        mock_client.create_job.assert_not_called()
        staged = out / "resume_stage" / "adapters.safetensors"
        assert staged.exists()
