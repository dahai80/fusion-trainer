"""Tests for SFTTrainer — training orchestration, checkpointing, dataloader.

Targets: sft/trainer.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fusion_trainer.config import TrainerConfig, SFTConfig, DatasetConfig
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
    def test_with_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # Mock tokenizer with apply_chat_template
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template.return_value = "<formatted>"
            trainer._tokenizer = mock_tokenizer

            data = [{"input": "hello", "output": "world"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert result[0]["text"] == "<formatted>"
            mock_tokenizer.apply_chat_template.assert_called_once()

    def test_with_tokenizer_fallback_on_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # Tokenizer that raises an error
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template.side_effect = Exception("template error")
            trainer._tokenizer = mock_tokenizer

            data = [{"input": "hi", "output": "ho"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert "|user|" in result[0]["text"]
            assert "hi" in result[0]["text"]
            assert "ho" in result[0]["text"]

    def test_without_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            trainer._tokenizer = None

            data = [{"input": "hello", "output": "world"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert "|user|" in result[0]["text"]
            assert "hello" in result[0]["text"]
            assert "world" in result[0]["text"]

    def test_uses_get_defensive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            # Data with missing keys should not crash
            data = [{"other": "data"}]
            result = trainer._prepare_dataloader(data)
            assert len(result) == 1
            assert "|user|" in result[0]["text"]


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
            # Simulate mlx_lm not installed
            import sys
            orig = sys.modules.get("mlx_lm", None)
            sys.modules["mlx_lm"] = None  # type: ignore[arg-type]
            try:
                with pytest.raises(ImportError, match="mlx-lm is required"):
                    trainer._load_model()
            finally:
                if orig:
                    sys.modules["mlx_lm"] = orig
                else:
                    del sys.modules["mlx_lm"]


# ===========================================================================
# Checkpointing
# ===========================================================================

class TestCheckpointing:
    def test_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(cache_path=tmpdir)
            trainer = SFTTrainer(cfg)
            from fusion_trainer.experiment import ExperimentLogger
            logger = ExperimentLogger("test", tmpdir)
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
            logger = ExperimentLogger("test", tmpdir)

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
# _run_training
# ===========================================================================

class TestRunTraining:
    def test_run_training_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora",
                cache_path=tmpdir,
                num_epochs=2,
                log_every=5,
                save_every=10,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"text": f"sample {i}"} for i in range(20)]
            result = trainer._run_training(train_loader, [])
            assert result["status"] == "completed"
            assert result["total_steps"] > 0
            assert result["method"] == "lora"
            assert result["output_dir"] == str(trainer._output_dir)

    def test_run_training_with_val(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(method="lora", cache_path=tmpdir, num_epochs=1)
            trainer = SFTTrainer(cfg)
            train_loader = [{"text": f"sample {i}"} for i in range(10)]
            val_data = [{"text": "val1"}, {"text": "val2"}]
            result = trainer._run_training(train_loader, val_data)
            assert result["status"] == "completed"

    def test_run_training_early_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="lora",
                cache_path=tmpdir,
                max_steps=5,
                num_epochs=10,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"text": f"sample {i}"} for i in range(100)]
            result = trainer._run_training(train_loader, [])
            assert result["total_steps"] == 5  # stopped at max_steps

    def test_run_training_full_method(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _make_config(
                method="full",
                cache_path=tmpdir,
                num_epochs=1,
                batch_size=2,
            )
            trainer = SFTTrainer(cfg)
            train_loader = [{"text": f"sample {i}"} for i in range(10)]
            result = trainer._run_training(train_loader, [])
            assert result["method"] == "full"
            assert result["lora_rank"] == 0


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

    def test_train_with_mock_integration(self):
        """Integration test: mock model loading, verify end-to-end flow."""
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

            # Mock the model loading
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_tokenizer.apply_chat_template.return_value = "<formatted>"
            trainer._model = mock_model
            trainer._tokenizer = mock_tokenizer

            # We need to patch _load_model to avoid the actual import
            original_load = trainer._load_model

            def mock_load():
                return mock_model, mock_tokenizer
            trainer._load_model = mock_load

            result = trainer.train()
            assert result["status"] == "completed"
            assert result["total_steps"] > 0
            assert result["method"] == "lora"
            assert result["lora_rank"] == 8