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
            use_mxfp8=kwargs.pop(
                "use_mxfp8", False
            ),  # default off; mxfp8 has its own test
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
            msgs = [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ho"},
            ]
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
    # E7 (round-2 audit): _save_checkpoint / _cleanup_checkpoints removed —
    # checkpoints live server-side in fusion-mlx, never managed by the trainer.
    # Only _save_final (local adapter pointer) remains, exercised below.

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

        def close(self):
            pass

        def health(self):
            # E3: _load_model now checks loaded models via pooled client.health().
            return {"data": [{"id": "test-model"}]}

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

    def test_run_training_qlora_real(self, monkeypatch):
        # fusion-mlx #402 已落地:qlora = fine_tune_type=qlora + quantize_base=True。
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="qlora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["fine_tune_type"] == "qlora"
            assert calls[0]["config"]["quantize_base"] is True
            assert calls[0]["config"]["quant_bits"] == 4

    def test_run_training_qdora_real(self, monkeypatch):
        # #63: qdora = fine_tune_type=dora + quantize_base=True + quant_bits.
        # mlx_lm 0.31.3 DoRALinear.from_base accepts QuantizedLinear, so
        # fusion-mlx quantizes the frozen base then attaches DoRA adapters.
        # PRD baseline for Qwen2.5-32B BNUP training.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="qdora", cache_path=tmpdir, num_epochs=1, quant_bits=8
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["fine_tune_type"] == "dora"
            assert calls[0]["config"]["quantize_base"] is True
            assert calls[0]["config"]["quant_bits"] == 8
            # LoRA fields present on the dora path (adapters are DoRA-shaped
            # but rank/alpha/dropout config still applies)
            assert calls[0]["config"]["lora_rank"] == 16

    def test_run_training_qdora_quant_bits_clamped(self, monkeypatch):
        # #63: quant_bits outside (4,8) clamps to 4 with a warning —
        # same gate as qlora (fusion-mlx validates in (4,8)).
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="qdora", cache_path=tmpdir, num_epochs=1, quant_bits=3
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["quant_bits"] == 4

    def test_run_training_mxfp8_explicit_true_passes_through(self, monkeypatch):
        # #402/#425 已落地:upstream service.py now routes mxfp8=True to the
        # 8-bit QLoRA path (quantize_base=True, quant_bits=8, fine_tune_type=
        # qlora) rather than raising. MLX 0.32 has no native fp8 dtype, so
        # semantics = 8-bit frozen base + LoRA for memory saving, NOT fp8
        # compute. fusion-trainer passes the mxfp8 flag through verbatim; the
        # upstream decides the routing (Rule 5). No fail-stop on the lora path.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora", cache_path=tmpdir, num_epochs=1, use_mxfp8=True
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["mxfp8"] is True
            assert calls[0]["config"]["fine_tune_type"] == "lora"

    def test_run_training_mxfp8_with_full_fail_stops(self, monkeypatch):
        # #425: mxfp8 + full is a contradiction (full unfreezes the base, so
        # there is nothing to quantize). fusion-trainer fails visibly here
        # before the job is submitted so the operator gets a clear message
        # rather than an opaque upstream 400. Upstream also raises; this is
        # the defense-in-depth client-side guard.
        from fusion_trainer.mlx_client import MLXFineTuneError

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="full", cache_path=tmpdir, num_epochs=1, use_mxfp8=True
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            _patch_mlx_client(monkeypatch)
            with pytest.raises(MLXFineTuneError):
                trainer._run_training(train_loader, [])

    def test_run_training_mxfp8_default_off_runs_normally(self, monkeypatch):
        # L2: default use_mxfp8=False trains at default precision with no
        # warning, no fail-stop. No mxfp8 flag sent upstream.
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora", cache_path=tmpdir, num_epochs=1, use_mxfp8=False
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls = _patch_mlx_client(monkeypatch)
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert calls[0]["config"]["fine_tune_type"] == "lora"
            assert "mxfp8" not in calls[0]["config"]

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
# VLM fine-tune (model_type=="vlm", #55, fusion-mlx #797/#800)
# ===========================================================================


def _patch_mlx_client_vlm(
    monkeypatch,
    events: list[dict] | None = None,
    final_status: str = "completed",
    adapter_path: str = "/fake/adapters/m/vlm_adapter",
):
    """Patch MLXFineTuneClient for the VLM path.

    Records create_vlm_job calls (NOT create_job) and ensures stage_dataset
    is never invoked. Returns the call list.
    """
    calls: list[dict] = []
    staged_invocations: list[str] = []

    class _FakeVLMClient:
        def __init__(self, *a, **kw):
            pass

        def close(self):
            pass

        def health(self):
            return {"data": [{"id": "test-vlm"}]}

        def create_vlm_job(self, model_id, dataset, config=None, adapter_name=""):
            calls.append(
                {
                    "model_id": model_id,
                    "dataset": dataset,
                    "config": config,
                    "adapter_name": adapter_name,
                }
            )
            return {"job_id": "fake-vlm-job-1", "status": "queued"}

        def wait_for_vlm_job(self, job_id, on_progress=None, **kw):
            for ev in events or [
                {
                    "type": "train_loss",
                    "step": 1,
                    "train_loss": 3.1,
                    "learning_rate": 1e-4,
                    "it_sec": 0.5,
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

    # Capture stage_dataset so we can assert the VLM path never calls it.
    _real_stage = mlx_mod.MLXFineTuneClient.stage_dataset

    def _spy_stage(*a, **kw):
        staged_invocations.append("stage_dataset was called!")
        return _real_stage(*a, **kw)

    monkeypatch.setattr(mlx_mod, "MLXFineTuneClient", _FakeVLMClient)
    monkeypatch.setattr(tr_mod, "MLXFineTuneClient", _FakeVLMClient, raising=False)
    _FakeVLMClient.stage_dataset = staticmethod(_spy_stage)
    return calls, staged_invocations


def _make_vlm_config(
    method: str = "lora",
    cache_path: str = "",
    dataset_path: str = "/tmp/vlm_dataset",
    **kwargs,
) -> TrainerConfig:
    return TrainerConfig(
        dataset=DatasetConfig(path=dataset_path, max_seq_length=512),
        sft=SFTConfig(
            model="test-vlm",
            method=method,  # type: ignore[arg-type]
            model_type="vlm",
            lora_rank=16,
            lora_alpha=32.0,
            **kwargs,
        ),
        cache_path=cache_path or "/tmp",
    )


class TestVLMTraining:
    def test_run_training_vlm_submits_to_vlm_endpoint(self, monkeypatch):
        # #55: model_type=="vlm" routes to create_vlm_job (NOT create_job),
        # does NOT stage a chat-messages jsonl, returns model_type=="vlm".
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_vlm_config(
                method="lora", cache_path=tmpdir, dataset_path="/tmp/vlm_ds"
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": f"Q{i}", "output": f"A{i}"} for i in range(8)]
            calls, staged = _patch_mlx_client_vlm(monkeypatch)
            result = trainer._run_training(train_loader, [])

            assert result["status"] == "completed"
            assert result["model_type"] == "vlm"
            assert result["job_id"] == "fake-vlm-job-1"
            assert len(calls) == 1
            assert calls[0]["model_id"] == "test-vlm"
            assert calls[0]["dataset"] == "/tmp/vlm_ds"
            cfg_sent = calls[0]["config"]
            assert cfg_sent["lora_layers"] == 16
            assert cfg_sent["full_finetune"] is False
            assert cfg_sent["train_vision"] is False
            assert "mask_prompt" not in cfg_sent
            assert "lr_schedule" not in cfg_sent
            assert "quant_bits" not in cfg_sent
            assert "quantize_base" not in cfg_sent
            assert "mxfp8" not in cfg_sent
            assert staged == []  # VLM path must NOT stage a chat-messages jsonl

    def test_run_training_vlm_qlora_warns_no_quantize(self, monkeypatch, caplog):
        # VLMFineTuneConfig has no quantize field; method="qlora" on a VLM
        # run degrades to plain LoRA with a loud warning (fail-visible), and
        # no quantize_base/quant_bits are sent upstream.
        import logging

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_vlm_config(
                method="qlora", cache_path=tmpdir, dataset_path="/tmp/vlm_ds"
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"input": "Q", "output": "A"}]
            calls, _ = _patch_mlx_client_vlm(monkeypatch)
            with caplog.at_level(logging.WARNING, logger="fusion_trainer.sft.trainer"):
                result = trainer._run_training(train_loader, [])

            assert result["status"] == "completed"
            cfg_sent = calls[0]["config"]
            assert cfg_sent["full_finetune"] is False
            assert "quantize_base" not in cfg_sent
            assert "quant_bits" not in cfg_sent
            assert any(
                "qlora" in rec.message.lower() and "vlm" in rec.message.lower()
                for rec in caplog.records
            )

    def test_build_vlm_config_field_names(self):
        # Assert the exact VLM field set — no text-SFT-only keys leak
        # (would 400 upstream: VLMFineTuneConfig(**body) TypeError on unknown).
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_vlm_config(
                method="lora", cache_path=tmpdir, dataset_path="/tmp/vlm_ds"
            )
            trainer = SFTTrainer(cfg)
            trainer._train_samples = [
                {"input": f"Q{i}", "output": f"A{i}"} for i in range(8)
            ]
            built = trainer._build_vlm_config()

            expected_keys = {
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_layers",
                "full_finetune",
                "train_vision",
                "optimizer",
                "learning_rate",
                "batch_size",
                "iters",
                "val_batches",
                "steps_per_report",
                "steps_per_eval",
                "steps_per_save",
                "max_seq_length",
                "grad_checkpoint",
                "grad_accumulation_steps",
                "seed",
                "num_processes",
            }
            assert set(built.keys()) == expected_keys
            assert built["iters"] > 0
            assert built["batch_size"] > 0


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


def _real_mlx_port() -> int:
    # 真实微调测试走 fusion-mlx 的 admin fine-tune API (/admin/api/fine-tune/*)。
    # 该 admin API 仅在 fusion-mlx 直连端口暴露,fusion-gateway(11432)不代理
    # /admin/api/fine-tune/*(gateway 的 /admin/api/* 是其自身密钥/渠道管理 API)。
    # 故默认直连 11434;若 fusion-mlx 监听非标准端口,用 FUSION_MLX_PORT 覆盖。
    # 注:这与 config._default_mlx_port() 的 11432(推理走 gateway)有意不同。
    return int(os.environ.get("FUSION_MLX_PORT", "11434"))


@pytest.mark.skipif(
    not _real_model_env(), reason="needs fusion-mlx server; set FUSION_TRAINER_REAL=1"
)
class TestRealFusionMLXIntegration:
    # Real-model verification (per project rule: 涉及到大模型测试，须真实加载模型).
    # Requires a running fusion-mlx server (direct, port 11434 by default) with
    # the Qwen3-0.6B-4bit model available. 用 FUSION_MLX_PORT 覆盖非标准端口。
    # 启动示例:
    #   FUSION_MLX_API_KEY=dahai168 FUSION_ROUTE_WARN_ONLY=true \
    #     ~/claude-home/fusion-mlx/start.sh start
    # 注意:避免在已加载 VL(VLM)模型的环境冷启动跑(上游 fusion-mlx#413 VLM
    # 冷启动崩溃),干净启动(不预加载 VLM)即可正常 SFT 训练。
    # Then: FUSION_TRAINER_REAL=1 FUSION_MLX_ADMIN_TOKEN=dahai168 pytest \
    #   tests/test_sft_trainer.py::TestRealFusionMLXIntegration -v

    def test_real_lora_job_runs_and_writes_adapter(self, tmp_path):
        client = MLXFineTuneClient(port=_real_mlx_port())
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
        client = MLXFineTuneClient(port=_real_mlx_port())
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

    def test_real_dpo_job_runs_and_writes_adapter(self, tmp_path):
        # fusion-mlx #399 已落地:真实 DPO 训练。
        client = MLXFineTuneClient(port=_real_mlx_port())
        assert isinstance(client.health(), dict)

        pairs = [
            {
                "prompt": f"What is {i}+1?",
                "chosen": f"The answer is {i + 1}.",
                "rejected": f"The answer is {i + 100}.",
            }
            for i in range(4)
        ]
        job = client.create_dpo_job(
            model_id="Qwen3-0.6B-4bit",
            preference_pairs=pairs,
            config={"iters": 2, "batch_size": 1, "beta": 0.1},
            adapter_name="ft_real_dpo",
        )
        job_id = job["job_id"]
        assert job["status"] in ("queued", "running")

        final = client.wait_for_dpo_job(job_id, timeout=300.0, poll_interval=2.0)
        assert final["status"] == "completed", f"DPO job did not complete: {final}"
        assert final.get("adapter_path"), "adapter_path missing on completed DPO job"
        adapter_file = Path(final["adapter_path"])
        assert adapter_file.exists(), f"DPO adapter not written: {adapter_file}"
        assert adapter_file.stat().st_size > 0

    def test_real_orpo_job_runs_and_writes_adapter(self, tmp_path):
        # fusion-mlx #399 已落地:真实 ORPO 训练。
        client = MLXFineTuneClient(port=_real_mlx_port())
        assert isinstance(client.health(), dict)

        pairs = [
            {
                "prompt": f"What is {i}+1?",
                "chosen": f"The answer is {i + 1}.",
                "rejected": f"The answer is {i + 100}.",
            }
            for i in range(4)
        ]
        job = client.create_orpo_job(
            model_id="Qwen3-0.6B-4bit",
            preference_pairs=pairs,
            config={"iters": 2, "batch_size": 1, "lambda_odds": 1.0},
            adapter_name="ft_real_orpo",
        )
        job_id = job["job_id"]
        assert job["status"] in ("queued", "running")

        final = client.wait_for_dpo_job(job_id, timeout=300.0, poll_interval=2.0)
        assert final["status"] == "completed", f"ORPO job did not complete: {final}"
        assert final.get("adapter_path"), "adapter_path missing on completed ORPO job"
        adapter_file = Path(final["adapter_path"])
        assert adapter_file.exists(), f"ORPO adapter not written: {adapter_file}"
        assert adapter_file.stat().st_size > 0

    def test_real_qlora_job_runs_and_writes_adapter(self, tmp_path):
        # fusion-mlx #402 已落地:真实 QLoRA 训练(quantized-base LoRA)。
        client = MLXFineTuneClient(port=_real_mlx_port())
        assert isinstance(client.health(), dict)

        samples = [
            {
                "messages": [
                    {"role": "user", "content": f"What is {i}+1?"},
                    {"role": "assistant", "content": f"{i + 1}"},
                ]
            }
            for i in range(6)
        ]
        dataset_dir = MLXFineTuneClient.stage_dataset(samples, tmp_path / "qlora_data")

        job = client.create_job(
            model_id="Qwen3-0.6B-4bit",
            dataset=dataset_dir,
            config={
                "fine_tune_type": "qlora",
                "quantize_base": True,
                "quant_bits": 4,
                "iters": 2,
                "batch_size": 1,
            },
            adapter_name="ft_real_qlora",
        )
        job_id = job["job_id"]
        assert job["status"] in ("queued", "running")

        final = client.wait_for_job(job_id, timeout=300.0, poll_interval=2.0)
        assert final["status"] == "completed", f"QLoRA job did not complete: {final}"
        assert final.get("adapter_path"), "adapter_path missing on completed QLoRA job"
        adapter_file = Path(final["adapter_path"])
        assert adapter_file.exists(), f"QLoRA adapter not written: {adapter_file}"
        assert adapter_file.stat().st_size > 0

    def test_real_reward_job_runs_and_writes_adapter(self, tmp_path):
        # fusion-mlx #424 已落地:真实 reward-model 训练(Bradley-Terry value head)。
        client = MLXFineTuneClient(port=_real_mlx_port())
        assert isinstance(client.health(), dict)

        pairs = [
            {
                "prompt": f"What is {i}+1?",
                "chosen": f"The answer is {i + 1}.",
                "rejected": f"The answer is {i + 100}.",
            }
            for i in range(4)
        ]
        job = client.create_reward_job(
            model_id="Qwen3-0.6B-4bit",
            preference_pairs=pairs,
            config={"iters": 4, "batch_size": 1, "learning_rate": 1e-4},
            adapter_name="ft_real_reward",
        )
        job_id = job["job_id"]
        assert job["status"] in ("queued", "running")

        final = client.wait_for_reward_job(job_id, timeout=300.0, poll_interval=2.0)
        assert final["status"] == "completed", f"reward job did not complete: {final}"
        assert final.get("adapter_path"), "adapter_path missing on reward job"
        adapter_file = Path(final["adapter_path"])
        assert adapter_file.exists(), f"reward adapter not written: {adapter_file}"
        assert adapter_file.stat().st_size > 0
        progress = final.get("progress", {}) or {}
        assert progress.get("reward_margin") is not None, "reward_margin missing"

    def test_real_rlsl_closed_loop_runs(self, tmp_path):
        # RLSL 闭环真实集成: Phase1 RM 训练(#424) -> Phase2 真实 GRPO(#363),
        # reward 信号经本机 RM reward server -> 上游 reward/score(#431)。
        # 验证 mean_reward 为真实 RM 分数(非 length fallback 58.0)。
        # 启动:
        #   FUSION_TRAINER_REAL=1 FUSION_MLX_ADMIN_TOKEN=dahai168 \
        #     FUSION_MLX_PORT=11434 pytest \
        #     tests/test_sft_trainer.py::TestRealFusionMLXIntegration::test_real_rlsl_closed_loop_runs -v -s
        from fusion_trainer.config import (
            DatasetConfig,
            RLSLConfig,
            SFTConfig,
            TrainerConfig,
        )
        from fusion_trainer.rlsl.pipeline import RLSLEngine

        data_path = tmp_path / "rlsl_pref.jsonl"
        import json

        with open(data_path, "w") as f:
            for i in range(3):
                f.write(
                    json.dumps(
                        {
                            "prompt": f"What is {i}+1?",
                            "chosen": f"The answer is {i + 1}.",
                            "rejected": f"The answer is {i + 100}.",
                        }
                    )
                    + "\n"
                )
        config = TrainerConfig(
            dataset=DatasetConfig(path=str(data_path)),
            sft=SFTConfig(
                model="Qwen3-0.6B-4bit",
                lora_rank=4,
                lora_alpha=8.0,
                lora_dropout=0.0,
            ),
            rlsl=RLSLConfig(
                method="rlsl",
                run_name="ft_real_rlsl",
                group_size=2,
                grpo_iterations=1,
                reward_num_epochs=1,
                batch_size=1,
                response_length=32,
                temperature=0.7,
            ),
            cache_path=str(tmp_path),
            mlx_port=_real_mlx_port(),
        )
        engine = RLSLEngine(config)
        result = engine.run()
        assert result["status"] == "completed", f"RLSL closed loop failed: {result}"
        assert result["method"] == "rlsl"
        assert result["reward_model"]["status"] == "completed"
        assert result["phase2"]["method"] == "grpo"
        assert result["phase2"]["backend"] == "grpo"
        mean_reward = result["phase2"].get("mean_reward")
        assert mean_reward is not None, "mean_reward missing on phase2"
        # 真实 RM 打分不为 length fallback(58.0);闭环走的是 RM。
        assert mean_reward != 58.0, (
            f"mean_reward is length fallback, not RM: {mean_reward}"
        )
