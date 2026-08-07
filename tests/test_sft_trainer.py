"""Tests for SFTTrainer — training orchestration, checkpointing, dataloader.

Targets: sft/trainer.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
from fusion_trainer.sft import SFTUnsupportedError
from fusion_trainer.sft.trainer import SFTTrainer


def _make_config(
    method: str = "lora",
    model: str = "test-model",
    output_dir: str = "",
    cache_path: str = "",
    **kwargs,
) -> TrainerConfig:
    return TrainerConfig(
        dataset=DatasetConfig(path="", max_seq_length=128),
        sft=SFTConfig(
            model=model,
            method=method,  # type: ignore[arg-type]
            lora_rank=16,
            lora_alpha=32.0,
            output_dir=output_dir,
            use_mxfp8=kwargs.pop("use_mxfp8", False),  # default off; mxfp8 has its own test
            **kwargs,
        ),
        cache_path=cache_path or "/tmp",
    )


def _make_dataset_file(tmpdir: str, n: int = 10, field: str = "input") -> str:
    import json

    p = Path(tmpdir) / "train.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({field: f"Q{i}", "output": f"A{i}"}) + "\n")
    return str(p)


# ===========================================================================
# Initialization
# ===========================================================================


class TestSFTTrainerInit:
    def test_init_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            assert trainer.sft_cfg.method == "lora"
            assert trainer._lora_cfg.rank == 16
            assert trainer._output_dir.exists()

    def test_init_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="full", cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # lora_rank is overridden from SFTConfig (default 16), not from from_method
            assert trainer._lora_cfg.quantize_base is False
            assert trainer._lora_cfg.rank == 16

    def test_init_qlora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="qlora", cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            assert trainer._lora_cfg.quantize_base is True
            assert trainer._lora_cfg.quant_bits == 4

    def test_init_custom_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "my_output"
            cfg = _make_config(output_dir=str(out_dir))
            trainer = SFTTrainer(cfg)
            assert trainer._output_dir == out_dir
            assert out_dir.exists()

    def test_init_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(run_name="my_run", cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            assert trainer._global_step == 0
            assert trainer._epoch == 0
            assert trainer._model is None
            assert trainer._tokenizer is None


# ===========================================================================
# _prepare_dataloader
# ===========================================================================


class TestPrepareDataloader:
    def test_preserves_input_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            data = [{"input": "hello", "output": "world"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert result[0] == {"input": "hello", "output": "world"}

    def test_passthrough_messages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ho"}]
            data = [{"messages": msgs}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert result[0]["messages"] == msgs

    def test_passthrough_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            data = [{"text": "raw text sample"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert result[0]["text"] == "raw text sample"

    def test_uses_get_defensive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # Data with missing keys should not crash; empty strings used
            data = [{"other": "data"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert result[0] == {"input": "", "output": ""}


# ===========================================================================
# _load_model
# ===========================================================================


class TestLoadModel:
    def test_no_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(model="", cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            with pytest.raises(ValueError, match="No model specified"):
                trainer._load_model()

    def test_mlx_lm_not_installed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(model="test-model", cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # _load_model() now uses HTTP to fusion-mlx, not mlx_lm import.
            # When fusion-mlx is unreachable, it should still return (None, None)
            # gracefully (prints warning, does not raise).
            result = trainer._load_model()
            assert result == (None, None)


# ===========================================================================
# Checkpointing
# ===========================================================================


class TestCheckpointing:
    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            from fusion_trainer.experiment import ExperimentLogger

            logger = ExperimentLogger("test", tmpdir, enable_tensorboard=False)
            path = trainer._save_checkpoint(100, logger)
            assert path is not None
            ckpt_dir = Path(path)
            assert ckpt_dir.exists()
            assert (ckpt_dir / "checkpoint.json").exists()
            with open(ckpt_dir / "checkpoint.json") as f:
                meta = json.load(f)
            assert meta["step"] == 100

    def test_save_final_lora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            cfg = _make_config(method="lora", output_dir=str(out_dir))
            trainer = SFTTrainer(cfg)
            path = trainer._save_final()
            assert path == str(out_dir / "lora_adapter")
            assert (out_dir / "lora_adapter" / "adapter_config.json").exists()
            with open(out_dir / "lora_adapter" / "adapter_config.json") as f:
                cfg_data = json.load(f)
            assert cfg_data["lora_rank"] == 16
            assert cfg_data["base_model"] == "test-model"

    def test_save_final_qlora(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            cfg = _make_config(method="qlora", output_dir=str(out_dir))
            trainer = SFTTrainer(cfg)
            path = trainer._save_final()
            assert (out_dir / "lora_adapter" / "adapter_config.json").exists()
            with open(out_dir / "lora_adapter" / "adapter_config.json") as f:
                cfg_data = json.load(f)
            assert cfg_data["quantize_base"] is True

    def test_save_final_full(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "output"
            cfg = _make_config(method="full", output_dir=str(out_dir))
            trainer = SFTTrainer(cfg)
            path = trainer._save_final()
            # Full method saves to output dir directly (no lora_adapter subdir)
            assert path == str(out_dir)

    def test_cleanup_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir, save_total_limit=2)
            trainer = SFTTrainer(cfg)
            from fusion_trainer.experiment import ExperimentLogger

            logger = ExperimentLogger("test", tmpdir, enable_tensorboard=False)

            # Create 4 checkpoints
            for step in [100, 200, 300, 400]:
                trainer._save_checkpoint(step, logger)

            # After cleanup, only 2 should remain
            ckpt_base = trainer._output_dir / "checkpoints"
            remaining = sorted(ckpt_base.iterdir())
            assert len(remaining) == 2  # save_total_limit=2

    def test_cleanup_checkpoints_no_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # Should not raise if dir doesn't exist
            trainer._cleanup_checkpoints()  # no-op


# ===========================================================================
# _run_training — mocked HTTP client (real training needs fusion-mlx up)
# ===========================================================================


def _patch_mlx_client(
    monkeypatch,
    events: list[dict] | None = None,
    final_status: str = "completed",
    adapter_path: str = "/fake/adapters/m/adapter",
):
    """Patch MLXFineTuneClient so _run_training runs without a server.

    Returns a list that records create_job calls for assertions.
    """
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        def create_job(self, model_id, dataset, config=None, adapter_name=""):
            calls.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "config": config,
                    "adapter_name": adapter_name,
                }
            )
            return {"job_id": "fake-job-123", "status": "queued"}

        def wait_for_job(self, job_id, on_progress=None, **kw):
            for ev in events or [
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
            ]:
                if on_progress is not None:
                    on_progress({"status": "running", "events": [ev]}, [ev])
            return {
                "status": final_status,
                "adapter_path": adapter_path,
                "events": events or [],
                "error": "" if final_status == "completed" else "boom",
            }

    import fusion_trainer.mlx_client as mlx_mod
    import fusion_trainer.sft.trainer as tr_mod

    _real_stage = mlx_mod.MLXFineTuneClient.stage_dataset
    monkeypatch.setattr(mlx_mod, "MLXFineTuneClient", _FakeClient)
    monkeypatch.setattr(tr_mod, "MLXFineTuneClient", _FakeClient, raising=False)
    # rebind the staticmethod captured before patching
    _FakeClient.stage_dataset = staticmethod(_real_stage)
    return calls


class TestRunTraining:
    def test_run_training_basic(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora",
                cache_path=tmpdir,
                num_epochs=2,
                log_every=5,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(20)]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert result["total_steps"] > 0
            assert result["method"] == "lora"
            assert result["output_dir"] == str(trainer._output_dir)
            assert result["job_id"] == "fake-job-123"
            assert calls[0]["model_id"] == "test-model"
            assert calls[0]["config"]["fine_tune_type"] == "lora"

    def test_run_training_with_val(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="lora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
            val_data = [{"input": "vq", "output": "va"}]
            _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, val_data)
            assert result["status"] == "completed"

    def test_run_training_early_stop(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora",
                cache_path=tmpdir,
                max_steps=5,
                num_epochs=10,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(100)]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["iters"] == 5

    def test_run_training_full_method(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="full",
                cache_path=tmpdir,
                num_epochs=1,
                batch_size=2,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(10)]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["method"] == "full"
            assert result["lora_rank"] == 0
            assert calls[0]["config"]["fine_tune_type"] == "full"

    def test_run_training_qlora_raises_upstream_blocked(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="qlora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            _patch_mlx_client(monkeypatch)
            with pytest.raises(SFTUnsupportedError, match="fusion-mlx/issues/402"):
                trainer._run_training(train_loader, [])

    def test_run_training_mxfp8_warns_not_raises(self, monkeypatch, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora", cache_path=tmpdir, num_epochs=1, use_mxfp8=True
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            with caplog.at_level("WARNING"):
                result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["fine_tune_type"] == "lora"
            assert "MXFP8" in caplog.text
            assert "fusion-mlx/issues/402" in caplog.text

    def test_run_training_failure(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="lora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            _patch_mlx_client(monkeypatch, final_status="failed")
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "failed"

    def test_run_training_val_loss_event(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="lora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            events = [
                {
                    "type": "train_loss",
                    "step": 1,
                    "train_loss": 2.0,
                    "learning_rate": 1e-5,
                    "tokens_per_second": 50,
                    "it_sec": 1.0,
                    "peak_memory_gb": 3.0,
                },
                {"type": "val_loss", "step": 1, "val_loss": 1.8},
            ]
            _patch_mlx_client(monkeypatch, events=events)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"


# ===========================================================================
# train() — full pipeline
# ===========================================================================


class TestTrain:
    def test_train_no_model_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = _make_dataset_file(tmpdir)
            cfg = TrainerConfig(
                dataset=DatasetConfig(path=data_path, max_seq_length=128),
                sft=SFTConfig(model="", method="lora"),
                cache_path=tmpdir,
            )
            trainer = SFTTrainer(cfg)
            with pytest.raises(ValueError, match="No model specified"):
                trainer.train()

    def test_train_no_dataset_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainerConfig(
                dataset=DatasetConfig(path="", max_seq_length=128),
                sft=SFTConfig(model="test-model", method="lora"),
                cache_path=tmpdir,
            )
            trainer = SFTTrainer(cfg)
            with pytest.raises(ValueError, match="must be provided"):
                trainer.train()

    def test_train_with_mock_integration(self, monkeypatch):
        """Integration test: mock model loading + HTTP client, verify flow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = _make_dataset_file(tmpdir, n=5)
            cfg = TrainerConfig(
                dataset=DatasetConfig(path=data_path, max_seq_length=128),
                sft=SFTConfig(
                    model="test-model",
                    method="lora",
                    lora_rank=8,
                    num_epochs=1,
                    log_every=1,
                    save_every=5,
                ),
                cache_path=tmpdir,
            )
            trainer = SFTTrainer(cfg)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template.return_value = "<formatted>"

            def mock_load():
                return mock_model, mock_tokenizer

            trainer._load_model = mock_load

            calls = _patch_mlx_client(monkeypatch)
            result = trainer.train()
            assert result["status"] == "completed"
            assert result["total_steps"] > 0
            assert result["method"] == "lora"
            assert result["lora_rank"] == 8
            assert calls[0]["model_id"] == "test-model"


import os

from fusion_trainer.mlx_client import MLXFineTuneClient


def _real_model_env() -> bool:
    return os.environ.get("FUSION_TRAINER_REAL") == "1"


@pytest.mark.skipif(
    not _real_model_env(), reason="needs fusion-mlx server; set FUSION_TRAINER_REAL=1"
)
class TestRealFusionMLXIntegration:
    # Real-model verification (per project rule: 涉及到大模型测试，须真实加载模型).
    # Requires a running fusion-mlx server with the Qwen3-0.6B-4bit model
    # available. Start it standalone:
    #   FUSION_MLX_API_KEY=dahai168 FUSION_ROUTE_WARN_ONLY=true \
    #     python -m fusion_mlx serve --base-path ~/.fusion-mlx \
    #     --host 127.0.0.1 --port 11434
    # Then: FUSION_TRAINER_REAL=1 FUSION_MLX_ADMIN_TOKEN=dahai168 pytest \
    #   tests/test_sft_trainer.py::TestRealFusionMLXIntegration -v

    def test_real_lora_job_runs_and_writes_adapter(self, tmp_path):
        client = MLXFineTuneClient(port=11434)
        # health() hits /v1/models; reachable server returns a dict (not raises)
        health = client.health()
        assert isinstance(health, dict)

        samples = []
        for i in range(6):
            samples.append(
                {
                    "messages": [
                        {"role": "user", "content": f"What is {i}+1?"},
                        {"role": "assistant", "content": f"{i + 1}"},
                    ]
                }
            )
        data_dir = tmp_path / "staged"
        adapter_name = "ft_real_ci"
        dataset_dir = MLXFineTuneClient.stage_dataset(samples, data_dir)

        job = client.create_job(
            model_id="Qwen3-0.6B-4bit",
            dataset=dataset_dir,
            config={"fine_tune_type": "lora", "iters": 2, "batch_size": 1},
            adapter_name=adapter_name,
        )
        job_id = job["job_id"]
        assert job["status"] in ("queued", "running")

        final = client.wait_for_job(job_id, timeout=300.0, poll_interval=2.0)
        assert final["status"] == "completed", f"job did not complete: {final}"
        assert final.get("adapter_path"), "adapter_path missing on completed job"

        adapter_file = Path(final["adapter_path"])
        assert adapter_file.exists(), f"adapter not written: {adapter_file}"
        assert adapter_file.stat().st_size > 0

    def test_real_second_job_not_stuck_queued(self, tmp_path):
        # Regression for fusion-mlx issue #361: a second job submitted after
        # the first terminates must transition out of 'queued'.
        client = MLXFineTuneClient(port=11434)
        samples = [
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ]
            }
            for i in range(6)
        ]
        names = ["ft_real_ci_a", "ft_real_ci_b"]
        for name in names:
            data_dir = tmp_path / name
            dataset_dir = MLXFineTuneClient.stage_dataset(samples, data_dir)
            client.create_job(
                model_id="Qwen3-0.6B-4bit",
                dataset=dataset_dir,
                config={"fine_tune_type": "lora", "iters": 1, "batch_size": 1},
                adapter_name=name,
            )
        jobs = {j["adapter_name"]: j for j in client.list_jobs()}
        for name in names:
            job_id = jobs[name]["job_id"]
            final = client.wait_for_job(job_id, timeout=300.0, poll_interval=2.0)
            assert final["status"] == "completed", f"{name} stuck: {final}"
