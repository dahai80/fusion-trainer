"""Tests for the CLI module.

Fusion-Trainer CLI provides: sft, rlsl, info.
Dataset management commands are handled by the companion fusion-model-hub project.
"""

from __future__ import annotations

from typer.testing import CliRunner

from fusion_trainer.cli import app

runner = CliRunner()


def test_cli_info():
    """Test that `fusion-trainer info` runs without error."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Fusion-Trainer" in result.stdout


def test_cli_version_long_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    from fusion_trainer._version import __version__

    assert __version__ in result.stdout
    assert "ft" in result.stdout


def test_cli_version_short_flag():
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    from fusion_trainer._version import __version__

    assert __version__ in result.stdout


def test_cli_version_exits_zero_no_usage_error():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Usage" not in result.stdout
    assert "Error" not in result.stdout


def test_cli_sft_help():
    """Test that `fusion-trainer sft --help` shows help."""
    result = runner.invoke(app, ["sft", "--help"])
    assert result.exit_code == 0
    assert "Run supervised fine-tuning" in result.stdout


def test_cli_rlsl_help():
    """Test that `fusion-trainer rlsl --help` shows help."""
    result = runner.invoke(app, ["rlsl", "--help"])
    assert result.exit_code == 0
    assert "Run RLSL / DPO / ORPO alignment" in result.stdout


def test_cli_no_dataset_command():
    """Dataset management is NOT part of Fusion-Trainer CLI."""
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code != 0  # dataset command should not exist
    # Typer outputs error to stderr when command not found
    assert "No such command" in result.stdout or "Error" in result.stderr


def test_cli_parse_hub_dataset_arg():
    """Test the hub: prefix parser via CLI help."""
    result = runner.invoke(app, ["sft", "--help"])
    assert result.exit_code == 0
    assert "hub:" in result.stdout


def test_cli_sft_init_adapter_path_flag():
    """#76: `ft sft --init-adapter-path` exists for flywheel round chaining."""
    result = runner.invoke(app, ["sft", "--help"])
    assert result.exit_code == 0
    assert "--init-adapter-path" in result.stdout
