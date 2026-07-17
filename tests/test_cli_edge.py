"""Tests for CLI edge cases — argument parsing, info command, error paths.

Targets: cli.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fusion_trainer.cli import _parse_dataset_arg, app
from typer.testing import CliRunner

runner = CliRunner()


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
        # Simulate MLX not being available
        import sys
        orig_mod = sys.modules.get("mlx.core", None)
        sys.modules["mlx.core"] = None  # type: ignore[arg-type]
        try:
            result = runner.invoke(app, ["info"])
            # Should still show info, just with MLX NOT INSTALLED
            assert result.exit_code == 0
            assert "NOT INSTALLED" in result.stdout
        finally:
            if orig_mod:
                sys.modules["mlx.core"] = orig_mod
            else:
                del sys.modules["mlx.core"]

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
        assert "--model-hub-path" in result.stdout
        assert "hub:" in result.stdout

    def test_rlsl_help(self):
        result = runner.invoke(app, ["rlsl", "--help"])
        assert result.exit_code == 0
        assert "Run RLSL" in result.stdout
        assert "--model-hub-path" in result.stdout
        assert "hub:" in result.stdout

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
# CLI edge cases
# ===========================================================================

class TestCLIEdgeCases:
    def test_sft_invalid_method(self):
        """Should fail gracefully with unknown method."""
        result = runner.invoke(app, ["sft", "--method", "invalid"])
        # typer validates via type hint, so this will fail at runtime
        # We just check it doesn't crash
        assert result.exit_code != 0 or "Error" in result.stderr or "usage" in result.stdout.lower()

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