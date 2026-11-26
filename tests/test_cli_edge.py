"""Tests for CLI edge cases — argument parsing, info command, error paths.

Targets: cli.py
"""

from __future__ import annotations

import json
import os
import tempfile

import click
import typer
from typer.testing import CliRunner

from fusion_trainer.cli import _parse_dataset_arg, app

runner = CliRunner()


def _cli_params(command: str) -> dict[str, click.Parameter]:
    # Rendered --help text width varies with terminal/rich version (option
    # lines wrap in CI's non-TTY environment); assert deterministically on
    # click-registered params instead of rendered text. See Rule 5.
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
        # Rendered width is sensitive; use deterministic registered-param assertions instead
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
            [
                "sft",
                "--preset",
                "no_such_preset",
                "--model",
                "m",
                "--dataset",
                "/x.jsonl",
            ],
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
                return {
                    "status": "completed",
                    "adapter_name": "fake",
                    "adapter_path": "/fake",
                }

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
                return {
                    "status": "completed",
                    "adapter_name": "fake",
                    "adapter_path": "/fake",
                }

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
# CLI --config YAML loading (A2 round-2 audit)
# ===========================================================================


class TestCLIConfig:
    """--config loads YAML; CLI flags override; missing file is graceful."""

    def test_config_seeds_yaml_fields(self, monkeypatch, tmp_path):
        # A2 (round-2 audit): --config was previously dead; YAML hyperparams
        # were silently dropped. Now the YAML section populates SFTConfig.
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                return {
                    "status": "completed",
                    "adapter_name": "fake",
                    "adapter_path": "/fake",
                }

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        cfg_yaml = tmp_path / "my.yaml"
        cfg_yaml.write_text(
            "sft:\n"
            "  method: lora\n"
            "  lora_rank: 64\n"
            "  learning_rate: 0.0003\n"
            "  num_epochs: 7\n"
            "  lora_alpha: 128.0\n"
            "  weight_decay: 0.02\n"
            "dataset:\n"
            "  hub_dataset: sft/my_data\n"
            "  max_seq_length: 2048\n"
        )
        result = runner.invoke(
            app,
            [
                "sft",
                "--config",
                str(cfg_yaml),
                "--model",
                "qwen2.5-7b",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = called["cfg"]
        assert cfg.sft.lora_rank == 64
        assert cfg.sft.learning_rate == 0.0003
        assert cfg.sft.num_epochs == 7
        # non-CLI-exposed extras flow from YAML
        assert cfg.sft.lora_alpha == 128.0
        assert cfg.sft.weight_decay == 0.02
        # dataset section fills hub_dataset when --dataset not passed
        assert cfg.dataset.hub_dataset == "sft/my_data"
        assert cfg.dataset.max_seq_length == 2048

    def test_cli_flag_overrides_yaml(self, monkeypatch, tmp_path):
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                return {
                    "status": "completed",
                    "adapter_name": "fake",
                    "adapter_path": "/fake",
                }

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        cfg_yaml = tmp_path / "my.yaml"
        cfg_yaml.write_text("sft:\n  lora_rank: 64\n  learning_rate: 0.0003\n")
        result = runner.invoke(
            app,
            [
                "sft",
                "--config",
                str(cfg_yaml),
                "--model",
                "m",
                "--dataset",
                "/d.jsonl",
                "--lora-rank",
                "8",
            ],
        )
        assert result.exit_code == 0, result.output
        cfg = called["cfg"]
        assert cfg.sft.lora_rank == 8  # CLI flag wins over YAML
        assert cfg.sft.learning_rate == 0.0003  # YAML fills (no flag)

    def test_missing_config_is_graceful(self, monkeypatch):
        # A missing --config file must not crash — it warns and uses defaults.
        called: dict = {}

        class FakeTrainer:
            def __init__(self, cfg):
                called["cfg"] = cfg

            def train(self):
                return {
                    "status": "completed",
                    "adapter_name": "fake",
                    "adapter_path": "/fake",
                }

        monkeypatch.setattr("fusion_trainer.sft.trainer.SFTTrainer", FakeTrainer)
        result = runner.invoke(
            app,
            [
                "sft",
                "--config",
                "/nonexistent/path.yaml",
                "--model",
                "m",
                "--dataset",
                "/d.jsonl",
            ],
        )
        assert result.exit_code == 0, result.output


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


# ===========================================================================
# DPO adapter chaining (fusion-mlx #1149)
# ===========================================================================


class TestDPOAdapterChaining:
    def test_rlsl_has_init_adapter_path_flag(self):
        assert "init_adapter_path" in _cli_params("rlsl")

    def test_bnup_train_has_init_adapter_path_flag(self):
        assert "init_adapter_path" in _cli_params("bnup-train")

    def test_rlsl_init_adapter_path_flows_to_config(self, monkeypatch, tmp_path):
        # --init-adapter-path reaches RLSLConfig.init_adapter_path.
        captured: dict = {}

        class FakeEngine:
            def __init__(self, cfg):
                captured["rlsl"] = cfg.rlsl

            def run(self):
                return {"status": "completed", "adapter_name": "x"}

        monkeypatch.setattr("fusion_trainer.rlsl.pipeline.RLSLEngine", FakeEngine)
        result = runner.invoke(
            app,
            [
                "rlsl",
                "--model",
                "m",
                "--dataset",
                "/x.jsonl",
                "--method",
                "dpo",
                "--init-adapter-path",
                "/sft/adapter",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["rlsl"].init_adapter_path == "/sft/adapter"

    def test_rlsl_empty_init_adapter_path_omitted(self, monkeypatch, tmp_path):
        # No flag → init_adapter_path empty (backward compat, fresh LoRA).
        captured: dict = {}

        class FakeEngine:
            def __init__(self, cfg):
                captured["rlsl"] = cfg.rlsl

            def run(self):
                return {"status": "completed", "adapter_name": "x"}

        monkeypatch.setattr("fusion_trainer.rlsl.pipeline.RLSLEngine", FakeEngine)
        result = runner.invoke(
            app,
            ["rlsl", "--model", "m", "--dataset", "/x.jsonl", "--method", "dpo"],
        )
        assert result.exit_code == 0, result.output
        assert captured["rlsl"].init_adapter_path == ""

    def test_rlsl_yaml_hyperparams_flow_to_config(self, monkeypatch, tmp_path):
        # #83: YAML rlsl section must reach RLSLConfig — num_epochs: 2 used to
        # be silently dropped (dataclass default 1 ran; live-observed iters 22
        # not 44 on the round-5 DPO flywheel).
        captured: dict = {}
        cfg_file = tmp_path / "rlsl.yaml"
        cfg_file.write_text(
            "rlsl:\n"
            "  method: dpo\n"
            "  num_epochs: 2\n"
            "  beta: 0.2\n"
            "  batch_size: 3\n"
            "  learning_rate: 1.0e-5\n"
            "  log_every: 5\n"
            "  save_every: 50\n"
            "  run_name: yaml-run\n"
            "dataset:\n"
            "  max_seq_length: 1024\n"
        )

        class FakeEngine:
            def __init__(self, cfg):
                captured["rlsl"] = cfg.rlsl

            def run(self):
                return {"status": "completed", "adapter_name": "x"}

        monkeypatch.setattr("fusion_trainer.rlsl.pipeline.RLSLEngine", FakeEngine)
        result = runner.invoke(
            app,
            [
                "rlsl",
                "--config",
                str(cfg_file),
                "--model",
                "m",
                "--dataset",
                "/x.jsonl",
                "--method",
                "dpo",
            ],
        )
        assert result.exit_code == 0, result.output
        r = captured["rlsl"]
        assert r.num_epochs == 2
        assert r.beta == 0.2
        assert r.batch_size == 3
        assert r.learning_rate == 1.0e-5
        assert r.log_every == 5
        assert r.save_every == 50

    def test_rlsl_defaults_without_yaml(self, monkeypatch, tmp_path):
        # No YAML/preset → dataclass defaults unchanged (backward compat).
        captured: dict = {}

        class FakeEngine:
            def __init__(self, cfg):
                captured["rlsl"] = cfg.rlsl

            def run(self):
                return {"status": "completed", "adapter_name": "x"}

        monkeypatch.setattr("fusion_trainer.rlsl.pipeline.RLSLEngine", FakeEngine)
        result = runner.invoke(
            app,
            ["rlsl", "--model", "m", "--dataset", "/x.jsonl", "--method", "dpo"],
        )
        assert result.exit_code == 0, result.output
        r = captured["rlsl"]
        assert r.num_epochs == 1
        assert r.beta == 0.1
        assert r.batch_size == 1


def test_from_rollouts_failures_output(tmp_path):
    from typer.testing import CliRunner

    from fusion_trainer.cli import app

    sink = tmp_path / "tap.jsonl"
    ok = {
        "schema_version": 1,
        "rollout_id": "ro-1",
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "done"},
        ],
        "prompt": "go",
        "outcome": {"signal": "task_success"},
    }
    bad = {
        "schema_version": 1,
        "rollout_id": "ro-2",
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "oops"},
        ],
        "prompt": "go",
        "outcome": {"signal": "user_rejected"},
    }
    sink.write_text(
        json.dumps(ok) + "\n" + json.dumps(bad) + "\n", encoding="utf-8"
    )
    fail_out = tmp_path / "failures.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "datasets",
            "from-rollouts",
            "--sink",
            str(sink),
            "--failures-output",
            str(fail_out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Failures:     1" in result.output
    assert "user_rejected" in result.output
    lines = [json.loads(l) for l in fail_out.read_text().splitlines()]
    assert lines[0]["rollout_id"] == "ro-2"


def test_from_rollouts_no_output_errors():
    from typer.testing import CliRunner

    from fusion_trainer.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["datasets", "from-rollouts", "--sink", "/nonexistent/tap.jsonl"],
    )
    assert result.exit_code == 1
    assert "nothing to write" in result.output
