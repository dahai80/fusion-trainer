"""Tests for BNUP two-stage training config + presets (issue #59).

Tests config YAML loading, preset validation, and CLI command structure.
Pipeline orchestration (SFT → DPO → merge) is tested via mocked trainers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from fusion_trainer.cli import app
from fusion_trainer.rlsl.presets import RLSL_PRESETS
from fusion_trainer.sft.presets import SFT_PRESETS


class TestBnupSftConfig:
    def test_yaml_loads(self):
        config_path = Path("configs/bnup_sft_32b.yaml")
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        sft = data["sft"]
        # #63: PRD baseline = 8-bit QDoRA (was qlora fallback).
        assert sft["method"] == "qdora"
        assert sft["quant_bits"] == 8
        assert sft["learning_rate"] == 1.0e-4
        assert sft["num_epochs"] == 3
        assert sft["max_seq_length"] == 4096
        assert sft["batch_size"] == 2
        assert sft["gradient_accumulation_steps"] == 8
        assert sft["lr_scheduler"] == "cosine"

    def test_yaml_dataset_section(self):
        config_path = Path("configs/bnup_sft_32b.yaml")
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["dataset"]["max_seq_length"] == 4096


class TestBnupDpoConfig:
    def test_yaml_loads(self):
        config_path = Path("configs/bnup_dpo_32b.yaml")
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        rlsl = data["rlsl"]
        assert rlsl["method"] == "dpo"
        assert rlsl["beta"] == 0.1
        assert rlsl["learning_rate"] == 5.0e-6
        assert rlsl["num_epochs"] == 2
        assert rlsl["batch_size"] == 1


class TestBnupPresets:
    def test_sft_preset_exists(self):
        assert "bnup_sft_32b" in SFT_PRESETS
        p = SFT_PRESETS["bnup_sft_32b"]
        # #63: PRD baseline = 8-bit QDoRA (was qlora fallback).
        assert p["method"] == "qdora"
        assert p["quant_bits"] == 8
        assert p["learning_rate"] == 1e-4
        assert p["num_epochs"] == 3
        assert p["max_seq_length"] == 4096
        assert "memory_estimate_gb" in p

    def test_dpo_preset_exists(self):
        assert "bnup_dpo_32b" in RLSL_PRESETS
        p = RLSL_PRESETS["bnup_dpo_32b"]
        assert p["method"] == "dpo"
        assert p["beta"] == 0.1
        assert p["learning_rate"] == 5e-6
        assert p["num_epochs"] == 2
        assert "memory_estimate_gb" in p


class TestBnupPrepCli:
    def test_bnup_prep_sft(self, tmp_path):
        import json

        raw = [
            {
                "question": "q1",
                "thinking": "t1",
                "dsl": {
                    "meta": {
                        "id": "x",
                        "grade": 6,
                        "topic": "t",
                        "visual_type": "geometry_2d",
                        "fallback_type": "tape_diagram",
                    },
                    "parameters": {"a": {"label": "a", "value": 1, "unit": "m"}},
                    "entities": [
                        {"id": "e", "type": "t", "label": "l", "properties": {}}
                    ],
                    "pipeline": [
                        {
                            "step": 1,
                            "title": "s",
                            "formula": "f",
                            "eval": "1",
                            "result_unit": "m",
                            "visual_state": {
                                "progress_range": [0, 1],
                                "active_entities": ["e"],
                                "action": "highlight",
                            },
                        }
                    ],
                },
            }
        ]
        input_file = tmp_path / "raw.json"
        input_file.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        output_file = str(tmp_path / "sft.jsonl")

        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "bnup-prep",
                "--stage",
                "sft",
                "--input",
                str(input_file),
                "--output",
                output_file,
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path(output_file).exists()

    def test_bnup_prep_invalid_stage(self):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            app, ["bnup-prep", "--stage", "invalid", "--input", "x", "--output", "y"]
        )
        assert result.exit_code == 1


class TestBnupTrainCli:
    def test_invalid_stage(self):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            app, ["bnup-train", "--stage", "invalid", "--model", "m", "--sft-data", "d"]
        )
        assert result.exit_code == 1

    def test_dpo_without_data(self):
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            app, ["bnup-train", "--stage", "dpo", "--model", "m", "--sft-data", "d"]
        )
        assert result.exit_code == 1

    def test_sft_stage_invokes_trainer(self, tmp_path):
        from typer.testing import CliRunner

        runner = CliRunner()
        with (
            patch("fusion_trainer.cli.SFTTrainer")
            if hasattr(
                __import__("fusion_trainer.cli", fromlist=["SFTTrainer"]), "SFTTrainer"
            )
            else patch("fusion_trainer.sft.trainer.SFTTrainer") as mock_trainer_cls
        ):
            mock_trainer = MagicMock()
            mock_trainer.train.return_value = {
                "status": "completed",
                "adapter_name": "bnup_sft",
            }
            mock_trainer_cls.return_value = mock_trainer
            result = runner.invoke(
                app,
                [
                    "bnup-train",
                    "--stage",
                    "sft",
                    "--model",
                    "test-model",
                    "--sft-data",
                    str(tmp_path / "data.jsonl"),
                    "--sft-config",
                    "configs/bnup_sft_32b.yaml",
                ],
            )
            assert result.exit_code == 0, result.output
            assert "BNUP SFT done" in result.output
