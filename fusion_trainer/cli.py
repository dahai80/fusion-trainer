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
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase log verbosity (-v INFO, -vv DEBUG). Default WARNING. "
        "Also settable via LOG_LEVEL env (DEBUG/INFO/WARNING/ERROR).",
    ),
):
    _configure_logging(verbose)
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
    ctx: typer.Context,
    config: str = typer.Option(
        "configs/sft_7b.yaml",
        "--config",
        "-c",
        help="Path to SFT training config YAML (loaded first; --preset and CLI flags override).",
    ),
    model: str = typer.Option(
        "",
        "--model",
        "-m",
        help="Base model name or path (overrides config/preset)",
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
        help="Training method: full, lora, dora, qlora",
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
        False,
        "--mxfp8/--no-mxfp8",
        help="Use MXFP8 mixed precision (upstream-blocked, fusion-mlx#402; "
        "explicit --mxfp8 fail-stops until upstream lands)",
    ),
    model_hub_path: str = typer.Option(
        "",
        "--model-hub-path",
        "-H",
        help="fusion-model-hub path (read-only, default: $FUSION_MODEL_HUB or ~/fusion-model-hub)",
    ),
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Publish the trained LoRA adapter to fusion-model-hub on completion (issue #49). Failures warn but do not block the run.",
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

    # A2 (round-2 audit): --config is honored. Hierarchy is real now:
    #   YAML file → preset → CLI flags (highest wins).
    # Source detection via ctx.get_parameter_source tells whether the user
    # actually passed a flag (COMMANDLINE) vs got the option default (DEFAULT),
    # so a YAML/preset value is only overridden when the flag is explicit.
    src = _flag_sources(ctx)

    # Step 1: YAML file → dict of overrides (only keys present in the file).
    yaml_vals = _load_yaml_config(config, section="sft")
    yaml_dataset = _load_yaml_config(config, section="dataset")

    # Step 2: preset → dict (fills CLI-exposed + non-exposed fields).
    preset_vals: dict = {}
    if preset:
        if preset not in SFT_PRESETS:
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available: {', '.join(sorted(SFT_PRESETS))}"
            )
        preset_vals = SFT_PRESETS[preset]
        log.info("Loaded preset '%s' (%s)", preset, preset_vals.get("description", ""))

    # Step 3: resolve each CLI-exposed field by precedence:
    #   flag (COMMANDLINE) > preset > YAML > option default.
    method = _resolve(src, "method", method, preset_vals, yaml_vals, default="lora")
    lora_rank = int(
        _resolve(src, "lora_rank", lora_rank, preset_vals, yaml_vals, default=16)
    )
    batch_size = int(
        _resolve(src, "batch_size", batch_size, preset_vals, yaml_vals, default=1)
    )
    learning_rate = float(
        _resolve(
            src, "learning_rate", learning_rate, preset_vals, yaml_vals, default=2e-5
        )
    )
    num_epochs = int(
        _resolve(src, "num_epochs", num_epochs, preset_vals, yaml_vals, default=3)
    )
    max_seq_length = int(
        _resolve(src, "max_seq_length", max_seq_length, yaml_dataset, default=4096)
    )
    model = _resolve(src, "model", model, preset_vals, yaml_vals, default="")
    run_name = _resolve(src, "run_name", run_name, preset_vals, yaml_vals, default="")
    output_dir = _resolve(
        src, "output_dir", output_dir, preset_vals, yaml_vals, default=""
    )

    _path, _hub_dataset = _parse_dataset_arg(dataset)
    # YAML dataset.hub_dataset fills in only when --dataset not passed.
    if not _path and not _hub_dataset and yaml_dataset.get("hub_dataset"):
        _hub_dataset = str(yaml_dataset["hub_dataset"])

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
    # Non-CLI-exposed fields: preset > YAML > dataclass default.
    _apply_extras(sft_cfg, preset_vals, yaml_vals)

    cfg = TrainerConfig(
        seed=seed,
        model_hub_path=model_hub_path or os.environ.get("FUSION_MODEL_HUB", ""),
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
            max_seq_length=max_seq_length,
        ),
        sft=sft_cfg,
        publish_adapter=publish,
    )

    trainer = SFTTrainer(cfg)
    result = trainer.train()
    # P0-6 (round-3 audit): a failed/empty result must exit non-zero and print
    # the error — previously the return dict was discarded and the CLI exited 0
    # even when training failed, so a failure was silent after terminal scroll.
    if not result or result.get("status") not in ("completed", "succeeded", "ok"):
        err = (result or {}).get("error", "training failed (no result returned)")
        typer.echo(f"✗ SFT failed: {err}")
        raise typer.Exit(1)
    typer.echo(
        f"✓ SFT completed: {result.get('adapter_name', result.get('adapter_path', ''))}"
    )
    if publish:
        from fusion_trainer.hub_client import publish_adapter_if_enabled

        pub = publish_adapter_if_enabled(cfg, result)
        if pub.get("published"):
            typer.echo(
                f"✓ Adapter published to hub: {pub.get('model', {}).get('id', '')}"
            )
        elif pub.get("error"):
            typer.echo(f"⚠ Adapter publish failed (non-blocking): {pub['error']}")
        elif pub.get("reason"):
            typer.echo(f"○ Adapter publish skipped: {pub['reason']}")


# ---------------------------------------------------------------------------
# RLSL command
# ---------------------------------------------------------------------------


@app.command()
def rlsl(
    ctx: typer.Context,
    config: str = typer.Option(
        "configs/rlsl_14b.yaml",
        "--config",
        "-c",
        help="Path to RLSL config YAML (loaded first; CLI flags override).",
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
    preset: str = typer.Option(
        "",
        "--preset",
        "-p",
        help="Named preset from RLSL_PRESETS (e.g. rlsl_grpo_7b). Loads GRPO "
        "hyperparameters incl. grpo_backend; CLI flags still override.",
    ),
    grpo_backend: str = typer.Option(
        "",
        "--grpo-backend",
        help="GRPO backend: rft (reward-filtered SFT, default) or grpo (real "
        "policy-gradient via fusion-mlx #363 Phase 2). Only used when "
        "--method grpo.",
    ),
    lora_rank: int = typer.Option(
        0,
        "--lora-rank",
        "-r",
        help="LoRA rank for the alignment adapter (0 = use preset/default).",
    ),
    lora_alpha: float = typer.Option(
        0.0,
        "--lora-alpha",
        help="LoRA alpha for the alignment adapter (0 = use preset/default).",
    ),
    lora_dropout: float = typer.Option(
        -1.0,
        "--lora-dropout",
        help="LoRA dropout for the alignment adapter (negative = use preset/default).",
    ),
    reward_model: str = typer.Option(
        "",
        "--reward-model",
        "-r",
        help="Reward model (required for rlsl method)",
    ),
    reward_learning_rate: float = typer.Option(
        -1.0,
        "--reward-lr",
        help="Reward-model training learning rate (negative = use default 1e-5).",
    ),
    reward_batch_size: int = typer.Option(
        0,
        "--reward-batch-size",
        help="Reward-model training batch size (0 = use default 1).",
    ),
    reward_num_epochs: int = typer.Option(
        0,
        "--reward-epochs",
        help="Reward-model training epochs (0 = use default 2).",
    ),
    sandbox: str = typer.Option(
        "off",
        "--sandbox",
        help="GRPO sandbox reward: off, code, math, auto. Default off: code "
        "mode execs model/data-generated Python with no real isolation "
        "(C2); enable only on fully trusted data with "
        "FUSION_CODE_SANDBOX_TRUSTED=1.",
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
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Publish the trained LoRA adapter to fusion-model-hub on completion (issue #49). Failures warn but do not block the run.",
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
    from fusion_trainer.rlsl.presets import RLSL_PRESETS

    # A2 (round-2 audit): --config honored. YAML → CLI flags (flag wins on
    # explicit pass, YAML fills otherwise).
    src = _flag_sources(ctx)
    yaml_vals = _load_yaml_config(config, section="rlsl")
    yaml_dataset = _load_yaml_config(config, section="dataset")

    # P0-3 (round-3 audit): --preset + grpo_backend resolution, so the
    # headline true-GRPO backend is reachable from the CLI (was CLI-unreachable:
    # no flag, no preset, YAML field dropped by the old _resolve call set).
    preset_vals: dict = {}
    if preset:
        if preset not in RLSL_PRESETS:
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available: {', '.join(sorted(RLSL_PRESETS))}"
            )
        preset_vals = RLSL_PRESETS[preset]

    method = _resolve(src, "method", method, preset_vals, yaml_vals, default="dpo")
    reward_model = _resolve(
        src, "reward_model", reward_model, preset_vals, yaml_vals, default=""
    )
    # P3-4 (round-3 audit): reward-model hyperparams tunable from CLI (were
    # locked to RLSLConfig defaults). Sentinel values = not passed -> default.
    reward_learning_rate = float(
        _resolve(
            src,
            "reward_learning_rate",
            reward_learning_rate,
            preset_vals,
            yaml_vals,
            default=-1.0,
        )
    )
    reward_batch_size = int(
        _resolve(
            src,
            "reward_batch_size",
            reward_batch_size,
            preset_vals,
            yaml_vals,
            default=0,
        )
    )
    reward_num_epochs = int(
        _resolve(
            src,
            "reward_num_epochs",
            reward_num_epochs,
            preset_vals,
            yaml_vals,
            default=0,
        )
    )
    sandbox = _resolve(src, "sandbox", sandbox, preset_vals, yaml_vals, default="off")
    group_size = int(
        _resolve(src, "group_size", group_size, preset_vals, yaml_vals, default=8)
    )
    grpo_iterations = int(
        _resolve(
            src, "grpo_iterations", grpo_iterations, preset_vals, yaml_vals, default=1
        )
    )
    # P0-3: grpo_backend now flows through _resolve (preset > YAML > default).
    grpo_backend = _resolve(
        src, "grpo_backend", grpo_backend, preset_vals, yaml_vals, default="rft"
    )
    run_name = _resolve(src, "run_name", run_name, preset_vals, yaml_vals, default="")
    model = _resolve(src, "model", model, preset_vals, yaml_vals, default="")

    _path, _hub_dataset = _parse_dataset_arg(dataset)
    if not _path and not _hub_dataset and yaml_dataset.get("hub_dataset"):
        _hub_dataset = str(yaml_dataset["hub_dataset"])
    ds_max_seq = (
        int(_resolve(src, "max_seq_length", 4096, yaml_dataset, default=4096))
        if yaml_dataset.get("max_seq_length")
        else 4096
    )

    # P1-6 (round-3 audit): expose LoRA hyperparams so alignment adapters are
    # tunable (were locked to SFTConfig defaults rank16/alpha32). 0/negative
    # sentinels = "not passed" → fall back to SFTConfig default.
    sft_kwargs: dict[str, Any] = {"model": model}
    if lora_rank and lora_rank > 0:
        sft_kwargs["lora_rank"] = lora_rank
    if lora_alpha and lora_alpha > 0:
        sft_kwargs["lora_alpha"] = lora_alpha
    if lora_dropout >= 0:
        sft_kwargs["lora_dropout"] = lora_dropout

    # P3-4: reward-model hyperparams (sentinel -> omit -> dataclass default).
    rlsl_kwargs: dict[str, Any] = {}
    if reward_learning_rate >= 0:
        rlsl_kwargs["reward_learning_rate"] = reward_learning_rate
    if reward_batch_size > 0:
        rlsl_kwargs["reward_batch_size"] = reward_batch_size
    if reward_num_epochs > 0:
        rlsl_kwargs["reward_num_epochs"] = reward_num_epochs

    cfg = TrainerConfig(
        seed=seed,
        model_hub_path=model_hub_path or os.environ.get("FUSION_MODEL_HUB", ""),
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
            max_seq_length=ds_max_seq,
        ),
        sft=SFTConfig(**sft_kwargs),
        rlsl=RLSLConfig(
            method=method,  # type: ignore[arg-type]
            reward_model=reward_model,
            run_name=run_name,
            sandbox_mode=sandbox,  # type: ignore[arg-type]
            group_size=group_size,
            grpo_iterations=grpo_iterations,
            grpo_backend=grpo_backend,  # type: ignore[arg-type]
            **rlsl_kwargs,
        ),
        publish_adapter=publish,
    )

    engine = RLSLEngine(cfg)
    result = engine.run()
    # P0-6 (round-3 audit): exit non-zero on failure (was discarded → exit 0).
    if not result or result.get("status") not in ("completed", "succeeded", "ok"):
        err = (result or {}).get("error", "alignment failed (no result returned)")
        typer.echo(f"✗ RLSL failed: {err}")
        raise typer.Exit(1)
    typer.echo(
        f"✓ RLSL completed: {result.get('adapter_name', result.get('adapter_path', ''))}"
    )
    if publish:
        from fusion_trainer.hub_client import publish_adapter_if_enabled

        pub = publish_adapter_if_enabled(cfg, result)
        if pub.get("published"):
            typer.echo(
                f"✓ Adapter published to hub: {pub.get('model', {}).get('id', '')}"
            )
        elif pub.get("error"):
            typer.echo(f"⚠ Adapter publish failed (non-blocking): {pub['error']}")
        elif pub.get("reason"):
            typer.echo(f"○ Adapter publish skipped: {pub['reason']}")


# ---------------------------------------------------------------------------
# Publish command (issue #49)
# ---------------------------------------------------------------------------


@app.command()
def publish(
    run_id: str = typer.Argument(
        ..., help="Run ID of a completed run whose adapter to publish."
    ),
    hub_url: str = typer.Option(
        "",
        "--hub-url",
        help="fusion-model-hub API base URL (default: $FUSION_MODEL_HUB_URL or http://localhost:11432).",
    ),
    hub_api_key: str = typer.Option(
        "",
        "--hub-api-key",
        help="API key for model-hub (default: $FUSION_MODEL_HUB_API_KEY).",
    ),
):
    """Publish a completed run's LoRA adapter to fusion-model-hub (issue #49).

    Retroactively publishes a run that was trained without --publish. Reads the
    run's recorded adapter_name + base model from the run registry. Unlike the
    in-training --publish flag (non-blocking), this is a STANDALONE CLI op —
    E6 (round-2 audit): on failure it exits non-zero so CI / `&&` chains can
    detect it. (The --publish during training stays non-blocking by design.)
    """
    from fusion_trainer.run_manager import get_run_manager

    mgr = get_run_manager()
    result = mgr.publish_run(run_id, hub_url=hub_url, api_key=hub_api_key)
    if result.get("published"):
        typer.echo(
            f"✓ Adapter published to hub: {result.get('model', {}).get('id', '')}"
        )
        raise typer.Exit(0)
    err = result.get("error", "unknown")
    typer.echo(f"⚠ Publish failed: {err}")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Info command
# ---------------------------------------------------------------------------


@app.command()
def info():
    """Show system information and hardware check."""
    import logging
    import platform

    import psutil

    from .config import _default_mlx_port

    typer.echo(f"{'Fusion-Trainer':30s} v{_get_version()}")
    typer.echo(f"{'Python':30s} {platform.python_version()}")
    typer.echo(f"{'Platform':30s} {platform.platform()}")
    typer.echo(f"{'Processor':30s} {platform.processor()}")
    typer.echo(f"{'Memory':30s} {psutil.virtual_memory().total / 1024**3:.1f} GB")

    # Check MLX via the pooled MLXFineTuneClient (P0-4 round-3 audit): was a
    # raw httpx.get that bypassed the unified HTTP boundary + retry/guard.
    mlx_port = _default_mlx_port()
    mlx_alive = False
    n_models = 0
    try:
        from .mlx_client import MLXFineTuneClient

        client = MLXFineTuneClient(port=mlx_port)
        try:
            health = client.health()
            mlx_alive = bool(health)
            models = health.get("data", []) if isinstance(health, dict) else []
            n_models = len(models) if isinstance(models, list) else 0
        finally:
            client.close()
    except Exception as exc:
        logging.getLogger(__name__).debug("fusion-mlx health probe failed: %s", exc)

    typer.echo(f"{'fusion-mlx port':30s} {mlx_port}")
    if mlx_alive:
        typer.echo(f"{'fusion-mlx':30s} running ({n_models} model(s) loaded)")
        if n_models == 0:
            typer.echo(
                f"{'  ⚠ warning':30s} no models loaded — a training job will fail until a model is loaded"
            )
    else:
        typer.echo(f"{'fusion-mlx':30s} NOT RUNNING")

    # Check fusion-mlx package version
    try:
        from fusion_mlx import __version__ as fm_version

        typer.echo(f"{'fusion-mlx version':30s} {fm_version}")
    except ImportError:
        typer.echo(f"{'fusion-mlx version':30s} NOT INSTALLED")

    # Report model hub path
    hub_path = os.environ.get("FUSION_MODEL_HUB", str(Path.home() / "fusion-model-hub"))
    exists = "✓" if os.path.isdir(hub_path) else "✗"
    typer.echo(f"{'Model hub path':30s} {hub_path}  ({exists})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _configure_logging(verbose: int) -> None:
    """P0-5 (round-3 audit): wire up logging so INFO/DEBUG reach the user and
    run.log. Without this the root logger stays at WARNING and every
    logger.info in trainers is dropped — failures become undiagnosable.

    Precedence: --verbose flag > LOG_LEVEL env > default WARNING.
    """
    import logging

    env_level = os.environ.get("LOG_LEVEL", "").upper()
    if verbose >= 2 or env_level == "DEBUG":
        level = logging.DEBUG
    elif verbose >= 1 or env_level == "INFO":
        level = logging.INFO
    elif env_level == "ERROR":
        level = logging.ERROR
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fusion_trainer").setLevel(level)


def _flag_sources(ctx: typer.Context) -> dict[str, Any]:
    """Map each CLI option name → whether the user passed it explicitly.

    A2 (round-2 audit): replaces the old equality-with-default guess in
    _preset_or_flag, which could not distinguish "user passed the default
    value" from "user passed nothing". Click's ParameterSource is exact.
    """
    from click.core import ParameterSource

    out: dict[str, Any] = {}
    try:
        params = ctx.command.params
    except Exception:
        return out
    for p in params:
        try:
            out[p.name] = ctx.get_parameter_source(p.name)
        except Exception:
            out[p.name] = ParameterSource.DEFAULT
    return out


def _resolve(
    src: dict[str, Any],
    flag_name: str,
    flag_val: Any,
    *higher: dict,
    default: Any = None,
) -> Any:
    """Resolve a config field by precedence: flag > higher[0] > ... > default.

    A2 (round-2 audit): flag wins ONLY when the user passed it explicitly
    (ParameterSource.COMMANDLINE). Otherwise the first dict that has the key
    wins; finally the option default. `higher` dicts are ordered
    highest-precedence-first (e.g. preset, then YAML).
    """
    got = src.get(flag_name)
    # Typer vendors click (typer._click.core) whose ParameterSource enum is a
    # DIFFERENT class than click.core's (and uses different int values:
    # COMMANDLINE is 1 vendored, 2 upstream). `==`/`is`/`.value` all fail
    # across the two classes. Compare by enum member NAME (string
    # "COMMANDLINE"), which is stable regardless of which copy produced the
    # value.
    is_cmd = getattr(got, "name", None) == "COMMANDLINE"
    if is_cmd:
        return flag_val
    for d in higher:
        if d and flag_name in d and d[flag_name] is not None:
            return d[flag_name]
    if default is not None:
        return default
    return flag_val


def _apply_extras(sft_cfg: Any, preset_vals: dict, yaml_vals: dict) -> None:
    """Fill SFTConfig fields the CLI does not expose: preset > YAML > default.

    A2 (round-2 audit): unifies the old preset-only path with YAML. Mutates
    sft_cfg in place; only sets fields present in preset or YAML.
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
        # preset wins over YAML (preset is a higher-precedence named bundle).
        if key in preset_vals:
            setattr(sft_cfg, key, preset_vals[key])
        elif key in yaml_vals:
            setattr(sft_cfg, key, yaml_vals[key])
    log.info(
        "Config extras applied: method=%s rank=%s alpha=%s grad_accum=%s",
        sft_cfg.method,
        sft_cfg.lora_rank,
        sft_cfg.lora_alpha,
        sft_cfg.gradient_accumulation_steps,
    )


def _load_yaml_config(path: str, section: str) -> dict:
    """Load one top-level section (sft/rlsl/dataset) from a YAML config file.

    A2 (round-2 audit): --config was previously declared but never read, so
    operator-authored YAML hyperparameters were silently dropped — a silent
    mis-configuration failure. Returns {} when the file is missing or the
    section is absent (graceful: the documented default configs live in
    configs/, but the file is optional). Logs loudly on a parse error so a
    malformed YAML is never silently ignored.
    """
    import logging

    log = logging.getLogger(__name__)
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        log.warning("--config %s not found; using preset/flag defaults", path)
        return {}
    try:
        import yaml

        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        log.error("--config %s failed to parse: %s", path, exc)
        raise typer.BadParameter(f"Config YAML parse error in {path}: {exc}")
    except OSError as exc:
        log.error("--config %s unreadable: %s", path, exc)
        raise typer.BadParameter(f"Config YAML unreadable: {path}: {exc}")
    section_data = data.get(section, {})
    if not isinstance(section_data, dict):
        log.warning(
            "--config %s: section '%s' is not a mapping; ignored", path, section
        )
        return {}
    log.info(
        "--config %s: loaded section '%s' (%d keys)", path, section, len(section_data)
    )
    return section_data


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
