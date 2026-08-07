"""Tests for CLI edge cases — argument parsing, info command, error paths.

Targets: cli.py
"""

from __future__ import annotations

import os
import tempfile

import click
import typer
from typer.testing import CliRunner

from fusion_trainer.cli import _parse_dataset_arg, app

runner = CliRunner()


def _cli_params(command: str) -> dict[str, click.Parameter]:
    # 渲染后的 --help 文本宽度随终端/rich 版本变化(CI 非 TTY 环境下 option 行会折叠),
    # 用 click 注册参数做确定性断言,不依赖渲染文本。见 Rule 5。
    return {p.name: p for p in typer.main.get_command(app).commands[command].params}


# ===========================================================================
# _parse_dataset_arg
# ===========================================================================


class TestParseDatasetArg:
    def test_direct_path(self):
        path, hub = _parse_dataset_arg("/data/file.jsonl")
        assert path == "/data/file.jsonl"
        assert hub == ""

    def test_hub_prefix(self):
        path, hub = _parse_dataset_arg("hub:sft/my_data")
        assert path == ""
        assert hub == "sft/my_data"

    def test_hub_prefix_deep(self):
        path, hub = _parse_dataset_arg("hub:preference/ultrafeedback/binarized")
        assert path == ""
        assert hub == "preference/ultrafeedback/binarized"

    def test_empty_string(self):
        path, hub = _parse_dataset_arg("")
        assert path == ""
        assert hub == ""

    def test_relative_path(self):
        path, hub = _parse_dataset_arg("data/train.jsonl")
        assert path == "data/train.jsonl"
        assert hub == ""

    def test_hub_without_category(self):
        path, hub = _parse_dataset_arg("hub:my_data")
        assert path == ""
        assert hub == "my_data"


# ===========================================================================
# CLI info command
# ===========================================================================


class TestCLIInfo:
    def test_info_success(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "Fusion-Trainer" in result.stdout
        assert "Python" in result.stdout
        assert "Model hub path" in result.stdout

    def test_info_mlx_not_installed(self):
        # Simulate fusion-mlx not being installed
        import sys

        orig_mod = sys.modules.get("fusion_mlx", None)
        sys.modules["fusion_mlx"] = None  # type: ignore[arg-type]
        try:
            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
            assert "NOT INSTALLED" in result.stdout
        finally:
            if orig_mod:
                sys.modules["fusion_mlx"] = orig_mod
            else:
                del sys.modules["fusion_mlx"]

    def test_info_fusion_mlx_not_installed(self):
        # Simulate fusion-mlx not being available
        import sys

        orig_mod = sys.modules.get("fusion_mlx", None)
        sys.modules["fusion_mlx"] = None  # type: ignore[arg-type]
        try:
            result = runner.invoke(app, ["info"])
            assert result.exit_code == 0
            assert "NOT INSTALLED" in result.stdout
        finally:
            if orig_mod:
                sys.modules["fusion_mlx"] = orig_mod
            else:
                del sys.modules["fusion_mlx"]


# ===========================================================================
# CLI command help
# ===========================================================================


class TestCLIHelp:
    def test_sft_help(self):
        result = runner.invoke(app, ["sft", "--help"])
        assert result.exit_code == 0
        assert "Run supervised fine-tuning" in result.stdout
        assert "hub:" in result.stdout
        # 渲染宽度敏感,改用注册参数确定性断言
        assert "model_hub_path" in _cli_params("sft")

    def test_rlsl_help(self):
        result = runner.invoke(app, ["rlsl", "--help"])
        assert result.exit_code == 0
        assert "Run RLSL" in result.stdout
        assert "hub:" in result.stdout
        assert "model_hub_path" in _cli_params("rlsl")

    def test_no_dataset_command(self):
        result = runner.invoke(app, ["dataset", "--help"])
        assert result.exit_code != 0
        # Check stderr for typer error message
        assert "No such command" in result.stdout or "Error" in result.stderr

    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Fusion-Trainer" in result.stdout
        assert "sft" in result.stdout
        assert "rlsl" in result.stdout
        assert "info" in result.stdout

    def test_no_args_shows_help(self):
        result = runner.invoke(app, [])
        # With no_args_is_help=True, typer exits with code 2 showing help
        assert "Fusion-Trainer" in result.stdout


# ===========================================================================
# CLI preset option
# ===========================================================================


class TestCLIPreset:
    """--preset loads SFT_PRESETS into SFTConfig; CLI flags override."""

    def test_sft_help_lists_preset(self):
        result = runner.invoke(app, ["sft", "--help"])
        assert result.exit_code == 0
        assert "preset" in _cli_params("sft")

    def test_unknown_preset_rejected(self, monkeypatch):
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                called["trained"] = True

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        result = runner.invoke(
            app,
            ["sft", "--preset", "no_such_preset", "--model", "m", "--dataset", "/x.jsonl"],
        )
        assert result.exit_code != 0
        assert "Unknown preset" in result.output

    def test_preset_seeds_config(self, monkeypatch):
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                called["trained"] = True

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        # sft_7b_full: method=full, gradient_accumulation_steps=8, lr=1e-5
        result = runner.invoke(
            app,
            [
                "sft",
                "--preset",
                "sft_7b_full",
                "--model",
                "qwen2.5-7b",
                "--dataset",
                "/data.jsonl",
                "--run-name",
                "preset_test",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = called["cfg"]
        assert cfg.sft.method == "full"
        assert cfg.sft.gradient_accumulation_steps == 8
        assert cfg.sft.learning_rate == 1e-5
        assert cfg.sft.lora_alpha == 32.0  # preset extra field applied
        assert called.get("trained") is True

    def test_cli_flag_overrides_preset(self, monkeypatch):
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                pass

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        # sft_7b_lora has lora_rank=16; CLI --lora-rank 32 wins
        result = runner.invoke(
            app,
            [
                "sft",
                "--preset",
                "sft_7b_lora",
                "--model",
                "m",
                "--dataset",
                "/d.jsonl",
                "--lora-rank",
                "32",
            ],
        )
        assert result.exit_code == 0, result.output
        assert called["cfg"].sft.lora_rank == 32


# ===========================================================================
# CLI edge cases
# ===========================================================================


class TestCLIEdgeCases:
    def test_sft_invalid_method(self):
        """Should fail gracefully with unknown method."""
        result = runner.invoke(app, ["sft", "--method", "invalid"])
        # typer validates via type hint, so this will fail at runtime
        # We just check it doesn't crash
        assert (
            result.exit_code != 0
            or "Error" in result.stderr
            or "usage" in result.stdout.lower()
        )

    def test_rlsl_invalid_method(self):
        result = runner.invoke(app, ["rlsl", "--method", "invalid"])
        assert result.exit_code != 0 or "Error" in result.stderr

    def test_info_with_env(self):
        """Test info with FUSION_MODEL_HUB set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["FUSION_MODEL_HUB"] = tmpdir
            try:
                result = runner.invoke(app, ["info"])
                assert result.exit_code == 0
                assert tmpdir in result.stdout
            finally:
                del os.environ["FUSION_MODEL_HUB"]
