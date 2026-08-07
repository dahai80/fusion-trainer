#!/usr/bin/env python3
"""CLI for Fusion-Trainer.

Commands:
    fusion-trainer sft           Run supervised fine-tuning
    fusion-trainer rlsl          Run RLSL/DPO/ORPO alignment
    fusion-trainer info          System info & hardware check

Dataset management (import, list, info) is handled by the companion
project fusion-model-hub, NOT by Fusion-Trainer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import typer

app = typer.Typer(
    name="fusion-trainer",
    help="Fusion-Trainer — lightweight MLX-based training workbench",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the fusion-trainer version and exit.",
        is_eager=True,
    ),
):
    if version:
        typer.echo(f"ft {_get_version()}")
        raise typer.Exit(0)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


# ---------------------------------------------------------------------------
# SFT command
# ---------------------------------------------------------------------------


@app.command()
def sft(
    config: str = typer.Option(
        "configs/sft_7b.yaml",
        "--config",
        "-c",
        help="Path to SFT training config YAML",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Base model name or path (overrides config)",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        "-d",
        help="Dataset file path, or 'hub:category/name' to load from model hub",
    ),
    preset: str = typer.Option(
        "",
        "--preset",
        "-p",
        help="Named preset from SFT_PRESETS (e.g. sft_7b_lora). Loads hyperparameters; CLI flags still override the preset.",
    ),
    output_dir: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output directory for checkpoints",
    ),
    method: str = typer.Option(
        "lora",
        "--method",
        "-M",
        help="Training method: full, lora, qlora",
    ),
    lora_rank: int = typer.Option(
        16,
        "--lora-rank",
        "-r",
        help="LoRA rank",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Per-device batch size",
    ),
    learning_rate: float = typer.Option(
        2e-5,
        "--learning-rate",
        "-l",
        help="Learning rate",
    ),
    num_epochs: int = typer.Option(
        3,
        "--epochs",
        "-e",
        help="Number of training epochs",
    ),
    max_seq_length: int = typer.Option(
        4096,
        "--max-seq-length",
        help="Maximum sequence length",
    ),
    run_name: str = typer.Option(
        "",
        "--run-name",
        "-n",
        help="Experiment run name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed",
    ),
    use_mxfp8: bool = typer.Option(
        True,
        "--mxfp8/--no-mxfp8",
        help="Use MXFP8 mixed precision",
    ),
    model_hub_path: str = typer.Option(
        "",
        "--model-hub-path",
        "-H",
        help="fusion-model-hub path (read-only, default: $FUSION_MODEL_HUB or ~/fusion-model-hub)",
    ),
):
    """Run supervised fine-tuning (SFT) on a base model.

    Dataset can be specified as:
      - A local file path:  --dataset /path/to/data.jsonl
      - A hub reference:    --dataset hub:sft/my_dataset
        Resolved as: {model_hub_path}/datasets/sft/my_dataset.jsonl

    Use --preset to load a named hyperparameter bundle (e.g. sft_7b_lora,
    sft_14b_qlora) from SFT_PRESETS. Explicit CLI flags override the preset.
    """
    import logging

    from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
    from fusion_trainer.sft.presets import SFT_PRESETS
    from fusion_trainer.sft.trainer import SFTTrainer

    log = logging.getLogger(__name__)

    # --preset seeds SFTConfig from a named bundle. Preset values fill the
    # exposed CLI options only when the user did NOT pass that flag (detected
    # by comparing to the option default), and always fill the fields the CLI
    # does not expose (lora_alpha, dropout, target_modules, lr_scheduler,
    # warmup_steps, weight_decay, max_grad_norm, gradient_accumulation_steps).
    preset_vals: dict = {}
    if preset:
        if preset not in SFT_PRESETS:
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available: {', '.join(sorted(SFT_PRESETS))}"
            )
        preset_vals = SFT_PRESETS[preset]
        log.info("Loaded preset '%s' (%s)", preset, preset_vals.get("description", ""))
        method = _preset_or_flag(method, preset_vals, "method", "lora")
        lora_rank = _preset_or_flag(lora_rank, preset_vals, "lora_rank", 16)
        batch_size = _preset_or_flag(batch_size, preset_vals, "batch_size", 1)
        learning_rate = _preset_or_flag(
            learning_rate, preset_vals, "learning_rate", 2e-5
        )
        num_epochs = _preset_or_flag(num_epochs, preset_vals, "num_epochs", 3)
        max_seq_length = _preset_or_flag(
            max_seq_length, preset_vals, "max_seq_length", 4096
        )

    _path, _hub_dataset = _parse_dataset_arg(dataset)

    sft_cfg = SFTConfig(
        model=model,
        method=method,  # type: ignore[arg-type]
        lora_rank=lora_rank,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_mxfp8=use_mxfp8,
        run_name=run_name,
        output_dir=output_dir,
    )
    if preset_vals:
        _apply_preset_extras(sft_cfg, preset_vals)

    cfg = TrainerConfig(
        seed=seed,
        model_hub_path=model_hub_path or os.environ.get("FUSION_MODEL_HUB", ""),
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
            max_seq_length=max_seq_length,
        ),
        sft=sft_cfg,
    )

    trainer = SFTTrainer(cfg)
    trainer.train()


# ---------------------------------------------------------------------------
# RLSL command
# ---------------------------------------------------------------------------


@app.command()
def rlsl(
    config: str = typer.Option(
        "configs/rlsl_14b.yaml",
        "--config",
        "-c",
        help="Path to RLSL config YAML",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Base model name or path",
    ),
    dataset: str = typer.Option(
        "",
        "--dataset",
        "-d",
        help="Preference dataset path, or 'hub:category/name'",
    ),
    method: str = typer.Option(
        "dpo",
        "--method",
        "-M",
        help="Alignment method: dpo, orpo, rlsl, grpo",
    ),
    reward_model: str = typer.Option(
        "",
        "--reward-model",
        "-r",
        help="Reward model (required for rlsl method)",
    ),
    sandbox: str = typer.Option(
        "auto",
        "--sandbox",
        help="GRPO sandbox reward: off, code, math, auto",
    ),
    group_size: int = typer.Option(
        8,
        "--group-size",
        help="GRPO samples per prompt (G)",
    ),
    grpo_iterations: int = typer.Option(
        1,
        "--grpo-iters",
        help="GRPO outer iterations (sample -> score -> SFT-update)",
    ),
    run_name: str = typer.Option(
        "",
        "--run-name",
        "-n",
        help="Experiment run name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed",
    ),
    model_hub_path: str = typer.Option(
        "",
        "--model-hub-path",
        "-H",
        help="fusion-model-hub path (read-only, default: $FUSION_MODEL_HUB or ~/fusion-model-hub)",
    ),
):
    """Run RLSL / DPO / ORPO alignment.

    Dataset can be specified as:
      - A local file path:  --dataset /path/to/preference.jsonl
      - A hub reference:    --dataset hub:preference/my_data
        Resolved as: {model_hub_path}/datasets/preference/my_data.jsonl
    """
    from fusion_trainer.config import (
        DatasetConfig,
        RLSLConfig,
        SFTConfig,
        TrainerConfig,
    )
    from fusion_trainer.rlsl.pipeline import RLSLEngine

    _path, _hub_dataset = _parse_dataset_arg(dataset)

    cfg = TrainerConfig(
        seed=seed,
        model_hub_path=model_hub_path or os.environ.get("FUSION_MODEL_HUB", ""),
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
        ),
        sft=SFTConfig(model=model),
        rlsl=RLSLConfig(
            method=method,  # type: ignore[arg-type]
            reward_model=reward_model,
            run_name=run_name,
            sandbox_mode=sandbox,  # type: ignore[arg-type]
            group_size=group_size,
            grpo_iterations=grpo_iterations,
        ),
    )

    engine = RLSLEngine(cfg)
    engine.run()


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------


@app.command()
def info():
    """Show system information and hardware check."""
    import platform

    import psutil

    typer.echo(f"{'Fusion-Trainer':30s} v{_get_version()}")
    typer.echo(f"{'Python':30s} {platform.python_version()}")
    typer.echo(f"{'Platform':30s} {platform.platform()}")
    typer.echo(f"{'Processor':30s} {platform.processor()}")
    typer.echo(f"{'Memory':30s} {psutil.virtual_memory().total / 1024**3:.1f} GB")

    # Check MLX via fusion-mlx
    try:
        import httpx

        from .config import _default_mlx_port

        mlx_port = _default_mlx_port()
        resp = httpx.get(f"http://localhost:{mlx_port}/v1/models", timeout=5.0)
        if resp.status_code == 200:
            typer.echo(f"{'fusion-mlx':30s} running")
            # Try to get MLX version from fusion-mlx package
            try:
                from fusion_mlx import __version__ as fm_version

                typer.echo(f"{'fusion-mlx version':30s} {fm_version}")
            except ImportError:
                pass
        else:
            typer.echo(f"{'fusion-mlx':30s} NOT RUNNING")
    except Exception:
        typer.echo(f"{'fusion-mlx':30s} NOT RUNNING")

    # Check fusion-mlx
    try:
        from fusion_mlx import __version__ as fm_version

        typer.echo(f"{'fusion-mlx version':30s} {fm_version}")
    except ImportError:
        typer.echo(f"{'fusion-mlx':30s} NOT INSTALLED")

    # Report model hub path
    hub_path = os.environ.get("FUSION_MODEL_HUB", str(Path.home() / "fusion-model-hub"))
    exists = "✓" if os.path.isdir(hub_path) else "✗"
    typer.echo(f"{'Model hub path':30s} {hub_path}  ({exists})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preset_or_flag(
    flag_val: Any,
    preset_vals: dict,
    key: str,
    option_default: Any,
) -> Any:
    """Return the preset value when the user did NOT pass the CLI flag.

    Typer fills in the option default when a flag is absent, so "not passed"
    is detected by equality with option_default. If the user passed the flag
    (flag_val != option_default), their value wins.
    """
    if flag_val == option_default:
        return preset_vals.get(key, flag_val)
    return flag_val


def _apply_preset_extras(sft_cfg: Any, preset_vals: dict) -> None:
    """Fill SFTConfig fields the CLI does not expose, from the preset.

    Only sets fields present in the preset; leaves SFTConfig defaults intact
    otherwise. Mutates sft_cfg in place.
    """
    import logging

    log = logging.getLogger(__name__)
    extras = [
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "lr_scheduler",
        "warmup_steps",
        "weight_decay",
        "max_grad_norm",
        "gradient_accumulation_steps",
        "model_type",
    ]
    for key in extras:
        if key in preset_vals:
            setattr(sft_cfg, key, preset_vals[key])
    log.info(
        "Preset extras applied: method=%s rank=%s alpha=%s grad_accum=%s",
        sft_cfg.method,
        sft_cfg.lora_rank,
        sft_cfg.lora_alpha,
        sft_cfg.gradient_accumulation_steps,
    )


def _parse_dataset_arg(arg: str) -> tuple[str, str]:
    """Parse a --dataset argument.

    The hub: prefix is resolved by DatasetLoader._resolve_path() as:

        {model_hub_path}/datasets/{hub_dataset}.jsonl

    Example:
        --dataset hub:sft/code_alpaca
        → hub_dataset="sft/code_alpaca"
        → resolved path: {model_hub_path}/datasets/sft/code_alpaca.jsonl

    Returns:
        (path, hub_dataset) — one will be empty, the other populated.
    """
    if arg.startswith("hub:"):
        return "", arg[4:]  # strip "hub:" prefix
    return arg, ""


def _get_version() -> str:
    try:
        from fusion_trainer._version import __version__

        return __version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app()


if __name__ == "__main__":
    main()
