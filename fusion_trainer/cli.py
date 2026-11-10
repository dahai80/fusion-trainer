#!/usr/bin/env python3
"""CLI for Fusion-Trainer.

Commands:
    fusion-trainer sft           Run supervised fine-tuning
    fusion-trainer rlsl          Run RLSL/DPO/ORPO alignment
    fusion-trainer info          System info & hardware check
    fusion-trainer datasets      Build trainer datasets from eval artifacts (#76)
    fusion-trainer flywheel      Eval-flywheel round registry (#76)

Dataset management (import, list, info) is handled by the companion
project fusion-model-hub, NOT by Fusion-Trainer. The `datasets` group only
converts fusion-eval trajectory exports into training-ready files.
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
        help="Training method: full, lora, dora, qdora (quantized DoRA), qlora",
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
        help="Use MXFP8 mixed precision (routes to 8-bit QLoRA, fusion-mlx#425)",
    ),
    pack_sequences: bool = typer.Option(
        False,
        "--pack-sequences/--no-pack-sequences",
        help="Sequence packing: concatenate short samples into rows of at most "
        "max_seq_length tokens with a block-diagonal attention mask "
        "(fusion-mlx#1400, requires fusion-mlx >= 0.10.17). Changes batch "
        "composition — results not byte-comparable with unpacked runs.",
    ),
    gradient_checkpointing: bool = typer.Option(
        False,
        "--grad-checkpoint/--no-grad-checkpoint",
        help="Gradient (activation) checkpointing: recompute activations in "
        "backward to cut peak memory at ~30% step-time cost. Required for "
        "LoRA on very large 8-bit models (27B-8bit + seq 4096 without it "
        "exhausts unified memory and panics macOS).",
    ),
    init_adapter_path: str = typer.Option(
        "",
        "--init-adapter-path",
        help="SFT adapter dir to initialize from (fusion-mlx #1182). Loads "
        "adapters.safetensors (or the newest numbered checkpoint) into the "
        "fresh LoRA layers before training; step counter restarts at 0. "
        "Use for multi-round flywheel chaining (#76). Empty = fresh LoRA.",
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
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an interrupted run from the newest checkpoint in the "
        "fusion-mlx adapter dir (~/.fusion-mlx/adapters/<model>/<run-name>/, "
        "issue #65; requires fusion-mlx >= 0.10.12 / #1182). No checkpoint "
        "found = loud warning + fresh start.",
    ),
    eval_select: bool = typer.Option(
        False,
        "--eval-select",
        help="After training, benchmark every numbered checkpoint via "
        "fusion-bench and pick the winner (issue #75). Table persisted to "
        "checkpoint_selection.json; bench outage falls back to lowest "
        "val_loss (loudly, never silently).",
    ),
    bench_url: str = typer.Option(
        "",
        "--bench-url",
        help="fusion-bench base URL for --eval-select (default "
        "$FUSION_BENCH_URL or http://localhost:11467).",
    ),
    bench_api_key: str = typer.Option(
        "",
        "--bench-api-key",
        help="Bearer token for fusion-bench (fusion-identity JWT).",
    ),
    bench_tenant_id: str = typer.Option(
        "",
        "--bench-tenant-id",
        help="X-Tenant-Id for fusion-bench's tenant middleware.",
    ),
    bench_executor: str = typer.Option(
        "speed",
        "--bench-executor",
        help="fusion-bench executor key (speed/accuracy/...).",
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
        pack_sequences=pack_sequences
        or bool(
            _resolve(
                src, "pack_sequences", False, preset_vals, yaml_vals, default=False
            )
        ),
        gradient_checkpointing=gradient_checkpointing
        or bool(
            _resolve(
                src,
                "gradient_checkpointing",
                False,
                preset_vals,
                yaml_vals,
                default=False,
            )
        ),
        init_adapter_path=init_adapter_path,
        run_name=run_name,
        output_dir=output_dir,
        # issue #75: checkpoint eval/selection wiring
        eval_select=eval_select
        or bool(
            _resolve(src, "eval_select", False, preset_vals, yaml_vals, default=False)
        ),
        bench_url=bench_url
        or str(_resolve(src, "bench_url", "", preset_vals, yaml_vals, default="")),
        bench_api_key=bench_api_key
        or str(_resolve(src, "bench_api_key", "", preset_vals, yaml_vals, default="")),
        bench_tenant_id=bench_tenant_id
        or str(
            _resolve(src, "bench_tenant_id", "", preset_vals, yaml_vals, default="")
        ),
        bench_executor=str(
            _resolve(
                src,
                "bench_executor",
                bench_executor,
                preset_vals,
                yaml_vals,
                default="speed",
            )
        ),
    )
    # Non-CLI-exposed fields: preset > YAML > dataclass default.
    _apply_extras(sft_cfg, preset_vals, yaml_vals)

    # #65: checkpoint-aware restart. Point init_adapter_path at the
    # fusion-mlx adapter dir; SFTTrainer._resolve_resume stages the newest
    # numbered checkpoint and subtracts its step from iters.
    if resume:
        from fusion_trainer.sft.trainer import (
            MLX_ADAPTER_BASE_DIR,
            find_latest_checkpoint,
        )

        run_name_eff = run_name or "sft_run"
        adapter_dir = MLX_ADAPTER_BASE_DIR / model / run_name_eff
        ckpt = find_latest_checkpoint(adapter_dir)
        if ckpt:
            sft_cfg.init_adapter_path = str(adapter_dir)
            typer.echo(f"↻ Resume: checkpoint step {ckpt[1]} found ({ckpt[0]})")
        else:
            typer.echo(
                f"⚠ --resume: no checkpoint in {adapter_dir}; starting fresh",
                err=True,
            )

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
    init_adapter_path: str = typer.Option(
        "",
        "--init-adapter-path",
        help="SFT adapter path to chain DPO/ORPO on (fusion-mlx #1149). The "
        "policy model loads with this adapter before alignment training; "
        "reference model stays base-only. Empty = fresh LoRA.",
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
    # #1149: DPO/ORPO adapter chaining on a prior SFT adapter.
    if init_adapter_path:
        rlsl_kwargs["init_adapter_path"] = init_adapter_path

    # #83: alignment hyperparams have no CLI flags, so they must flow from
    # preset/YAML — the old code never forwarded them and RLSLConfig ran on
    # dataclass defaults (live-observed: YAML num_epochs: 2 trained 1 epoch,
    # iters 22 not 44; beta/lr/batch only matched by coincidence). Resolution
    # order mirrors bnup-train's _pick: preset > YAML > dataclass default.
    for _key, _caster, _default in (
        ("beta", float, 0.1),
        ("batch_size", int, 1),
        ("learning_rate", float, 5e-6),
        ("num_epochs", int, 1),
        ("log_every", int, 10),
        ("save_every", int, 200),
        ("output_dir", str, ""),
    ):
        if _key in preset_vals:
            rlsl_kwargs[_key] = _caster(preset_vals[_key])
        elif _key in yaml_vals:
            rlsl_kwargs[_key] = _caster(yaml_vals[_key])

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
        # #65 live-verify found YAML save_every silently dropped here — the
        # job ran with the default 500 and wrote no numbered checkpoint, so
        # --resume had nothing to resume from. Checkpointing cadence, logging
        # cadence, quant bits, and the step cap are all config-file concerns.
        "save_every",
        "log_every",
        "quant_bits",
        "max_steps",
        # issue #75: eval/selection fields without dedicated CLI flags.
        # bench_url/api_key/tenant/executor/eval_select are handled in the
        # sft command body (flag-aware); these remain YAML/preset-only.
        "bench_suite",
        "bench_params",
        "bench_task_timeout",
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


# ---------------------------------------------------------------------------
# BNUP data prep (issue #56)
# ---------------------------------------------------------------------------


@app.command(name="bnup-prep")
def bnup_prep(
    stage: str = typer.Option(
        ...,
        "--stage",
        help="sft or dpo — prepare SFT JSONL or DPO preference pairs",
    ),
    input_file: str = typer.Option(
        ..., "--input", "-i", help="Input file (raw JSON for sft, SFT JSONL for dpo)"
    ),
    output_file: str = typer.Option(..., "--output", "-o", help="Output JSONL path"),
    system_prompt: str = typer.Option(
        "", "--system-prompt", help="System prompt (default: BNUP Socratic lecturer)"
    ),
    neg_ratio: int = typer.Option(
        3, "--neg-ratio", help="DPO negatives per sample (stage=dpo only)"
    ),
):
    """Prepare BNUP training data (SFT validation or DPO pair generation).

    Stage sft: reads raw {question, thinking, dsl} JSON, validates DSL
    via fusion-core, outputs <thinking>+<dsl> ShareGPT JSONL.

    Stage dpo: reads SFT JSONL, generates neg_ratio negative variants
    per sample, outputs ShareGPT preference-pair JSONL.
    """
    if stage not in ("sft", "dpo"):
        typer.echo(f"Error: --stage must be 'sft' or 'dpo', got: {stage}", err=True)
        raise typer.Exit(1)

    if stage == "sft":
        from fusion_trainer.dataset.bnup_prep import (
            DEFAULT_SYSTEM_PROMPT,
            process_raw_sft,
        )

        prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        result = process_raw_sft(input_file, output_file, system_prompt=prompt)
        typer.echo(
            f"SFT: valid={result['valid']} skipped={result['skipped']} "
            f"total={result['total']}"
        )
    else:
        from fusion_trainer.dataset.bnup_dpo_prep import create_dpo_pairs

        result = create_dpo_pairs(input_file, output_file, neg_ratio=neg_ratio)
        typer.echo(
            f"DPO: pairs={result['pairs']} samples={result['samples']} "
            f"neg_ratio={result['neg_ratio']}"
        )


# ---------------------------------------------------------------------------
# Model merge + export (issue #58)
# ---------------------------------------------------------------------------


@app.command(name="merge")
def merge(
    base_model: str = typer.Option(
        ..., "--base", help="Base model name/alias (fusion-mlx model id)"
    ),
    adapter_path: str = typer.Option(
        ..., "--adapter", help="Local path to trained LoRA/DoRA adapter"
    ),
    output_dir: str = typer.Option(
        "", "--output", "-o", help="Output dir for merged SafeTensors (auto if empty)"
    ),
    quantize: str = typer.Option(
        "", "--quantize", help="Post-merge quantization: 4bit or 8bit (none if empty)"
    ),
    dequantize: bool = typer.Option(
        False, "--dequantize", help="Dequantize frozen base to float during merge"
    ),
    publish: bool = typer.Option(
        False, "--publish", help="Register merged model in fusion-model-hub after merge"
    ),
):
    """Merge a trained adapter into the base model via fusion-mlx.

    Calls fusion-mlx POST /v1/merge-adapter, optionally quantizes for
    Apple Silicon deployment, and (with --publish) registers the result
    in fusion-model-hub.
    """
    import logging

    from fusion_trainer.merge import merge_and_export
    from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError

    logger = logging.getLogger("fusion_trainer")
    mlx_port = _default_mlx_port_safe()
    client = MLXFineTuneClient(port=mlx_port)
    try:
        result = merge_and_export(
            client=client,
            base_model=base_model,
            adapter_path=adapter_path,
            output_dir=output_dir,
            quantize=quantize,
            dequantize=dequantize,
        )
        merge_res = result.get("merge", {})
        quant_res = result.get("quantize", {})
        val_res = result.get("validation", {})
        final_path = result.get("final_model_path", "")
        typer.echo(f"✅ Merge complete: {final_path}")
        typer.echo(f"   Status: {merge_res.get('status', '?')}")
        if quant_res.get("status") != "skipped":
            typer.echo(f"   Quantize: {quant_res}")
        if val_res.get("status") == "completed":
            tag = "✓" if val_res.get("valid") else "⚠"
            typer.echo(
                f"   {tag} Validation: thinking={val_res.get('has_thinking')} "
                f"dsl={val_res.get('has_dsl')} json={val_res.get('dsl_valid_json')}"
            )
        elif val_res.get("status") == "failed":
            typer.echo(f"   ⚠ Validation: {val_res.get('reason', 'failed')}")

        if publish:
            from fusion_trainer.config import _default_hub_api_key, _default_hub_url
            from fusion_trainer.hub_client import HubPublishError, ModelHubClient

            merged_name = (
                Path(output_dir).name if output_dir else f"{base_model}-merged"
            )
            hub_url = _default_hub_url()
            api_key = _default_hub_api_key()
            try:
                hub = ModelHubClient(base_url=hub_url, api_key=api_key)
                pub = hub.publish_merged_model(
                    name=merged_name,
                    base_model=base_model,
                )
                if pub.get("published"):
                    typer.echo(
                        f"✓ Merged model published to hub: "
                        f"{pub.get('model', {}).get('id', '')}"
                    )
            except HubPublishError as exc:
                typer.echo(f"⚠ Merged model publish failed (non-blocking): {exc}")
    except MLXFineTuneError as exc:
        typer.echo(f"Error: merge failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        client.close()


# ---------------------------------------------------------------------------
# BNUP two-stage pipeline (issue #59)
# ---------------------------------------------------------------------------


@app.command(name="bnup-train")
def bnup_train(
    stage: str = typer.Option(
        "sft",
        "--stage",
        help="sft | dpo | pipeline (pipeline = SFT → DPO → merge)",
    ),
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Base model (fusion-mlx alias, e.g. Qwen2.5-32B-Instruct)",
    ),
    sft_config: str = typer.Option(
        "configs/bnup_sft_32b.yaml", "--sft-config", help="SFT config YAML"
    ),
    dpo_config: str = typer.Option(
        "configs/bnup_dpo_32b.yaml", "--dpo-config", help="DPO config YAML"
    ),
    sft_dataset: str = typer.Option(
        ..., "--sft-data", help="SFT dataset (hub:sft/xxx or /path/to/sft.jsonl)"
    ),
    dpo_dataset: str = typer.Option(
        "", "--dpo-data", help="DPO dataset (required for dpo/pipeline stages)"
    ),
    merge_output: str = typer.Option(
        "", "--merge-output", help="Merged model output dir (pipeline stage)"
    ),
    preset: str = typer.Option(
        "", "--preset", help="Override preset (bnup_sft_32b / bnup_dpo_32b)"
    ),
    init_adapter_path: str = typer.Option(
        "",
        "--init-adapter-path",
        help="SFT adapter path to chain DPO on (dpo stage; fusion-mlx #1149). "
        "Pipeline stage auto-fills this from SFT output.",
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume an interrupted SFT stage from the newest checkpoint "
        "(issue #65; fusion-mlx #1182). Applies to the sft/pipeline stage.",
    ),
):
    """Run BNUP two-stage training (SFT → DPO → merge).

    Stage sft: run BNUP SFT (8-bit QLoRA, <thinking>+<dsl> format).
    Stage dpo: run BNUP DPO preference alignment (beta=0.1).
    Stage pipeline: SFT → DPO → merge sequentially.

    DPO adapter chaining (fusion-mlx #1149, closed via PR #1152): the SFT
    adapter path flows into DPO's init_adapter_path, so the policy model
    loads with the SFT adapter before DPO training. The reference model
    stays base-only (DPO theory). The pipeline then merges the DPO adapter
    (which carries the SFT-then-DPO signal) with the base model.
    """
    if stage not in ("sft", "dpo", "pipeline"):
        typer.echo(f"Error: --stage must be sft/dpo/pipeline, got: {stage}", err=True)
        raise typer.Exit(1)

    if stage in ("dpo", "pipeline") and not dpo_dataset:
        typer.echo("Error: --dpo-data required for dpo/pipeline stage", err=True)
        raise typer.Exit(1)

    import logging

    logger = logging.getLogger("fusion_trainer")

    def _run_sft_stage() -> dict:
        typer.echo("─" * 50)
        typer.echo("BNUP Stage 1: SFT (8-bit QLoRA)")
        typer.echo("─" * 50)
        sft_preset = preset or "bnup_sft_32b"
        result = _invoke_sft(
            config=sft_config,
            model=model,
            dataset=sft_dataset,
            preset=sft_preset,
            logger=logger,
            resume=resume,
        )
        return {"stage": "sft", "status": "completed", "result": result}

    def _run_dpo_stage(init_adapter_path: str = "") -> dict:
        typer.echo("─" * 50)
        typer.echo("BNUP Stage 2: DPO preference alignment")
        typer.echo("─" * 50)
        if init_adapter_path:
            typer.echo(f"       Chaining on SFT adapter: {init_adapter_path}")
        dpo_preset = preset or "bnup_dpo_32b"
        result = _invoke_rlsl(
            config=dpo_config,
            model=model,
            dataset=dpo_dataset,
            preset=dpo_preset,
            method="dpo",
            logger=logger,
            init_adapter_path=init_adapter_path,
        )
        return {"stage": "dpo", "status": "completed", "result": result}

    if stage == "sft":
        _run_sft_stage()
    elif stage == "dpo":
        _run_dpo_stage(init_adapter_path=init_adapter_path)
    else:
        sft_outcome = _run_sft_stage()
        sft_adapter = (sft_outcome.get("result") or {}).get("final_model_path", "")
        if not sft_adapter:
            logger.warning(
                "BNUP pipeline: SFT produced no adapter_path — DPO will train "
                "a fresh LoRA (no chaining). Check SFT logs."
            )
        dpo_outcome = _run_dpo_stage(init_adapter_path=sft_adapter)
        if merge_output:
            typer.echo("─" * 50)
            typer.echo("BNUP Stage 3: Merge adapter → base model")
            typer.echo("─" * 50)
            from fusion_trainer.merge import merge_adapter
            from fusion_trainer.mlx_client import MLXFineTuneClient, MLXFineTuneError

            dpo_adapter = (dpo_outcome.get("result") or {}).get("adapter_path", "")
            merge_adapter_path = dpo_adapter or sft_adapter
            if not merge_adapter_path:
                typer.echo(
                    "Error: no adapter path to merge (both SFT and DPO empty)",
                    err=True,
                )
                raise typer.Exit(1)
            typer.echo(f"       Merging adapter: {merge_adapter_path}")
            client = MLXFineTuneClient(port=_default_mlx_port_safe())
            try:
                result = merge_adapter(
                    client=client,
                    base_model=model,
                    adapter_path=merge_adapter_path,
                    output_dir=merge_output,
                )
                typer.echo(f"✅ Merge: {result.get('output_path', '?')}")
            except MLXFineTuneError as exc:
                typer.echo(f"Error: merge failed: {exc}", err=True)
                raise typer.Exit(1) from exc
            finally:
                client.close()

    typer.echo("✅ BNUP training complete.")


def _default_mlx_port_safe() -> int:
    try:
        from .config import _default_mlx_port

        return _default_mlx_port()
    except Exception:
        return 11434


def _invoke_sft(
    config: str,
    model: str,
    dataset: str,
    preset: str,
    logger: Any,
    resume: bool = False,
) -> dict:
    """Invoke the SFT training path (reuses SFTTrainer).

    Returns the SFTTrainer result dict (carries adapter_path for DPO chaining).
    """
    from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
    from fusion_trainer.sft.presets import SFT_PRESETS
    from fusion_trainer.sft.trainer import SFTTrainer

    yaml_sft = _load_yaml_config(config, section="sft")
    yaml_dataset = _load_yaml_config(config, section="dataset")

    preset_vals: dict = {}
    if preset:
        if preset not in SFT_PRESETS:
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available: {', '.join(sorted(SFT_PRESETS))}"
            )
        preset_vals = SFT_PRESETS[preset]
        logger.info(
            "Loaded preset '%s' (%s)", preset, preset_vals.get("description", "")
        )

    def _pick(key: str, default: Any = None) -> Any:
        if key in preset_vals:
            return preset_vals[key]
        if key in yaml_sft:
            return yaml_sft[key]
        return default

    sft_cfg = SFTConfig(
        model=model or str(_pick("model", "")),
        model_type=str(_pick("model_type", "lm")),  # type: ignore[arg-type]
        method=str(_pick("method", "lora")),  # type: ignore[arg-type]
        lora_rank=int(_pick("lora_rank", 16)),
        lora_alpha=float(_pick("lora_alpha", 32.0)),
        lora_dropout=float(_pick("lora_dropout", 0.05)),
        quant_bits=int(_pick("quant_bits", 4)),
        batch_size=int(_pick("batch_size", 1)),
        gradient_accumulation_steps=int(_pick("gradient_accumulation_steps", 4)),
        learning_rate=float(_pick("learning_rate", 2e-5)),
        lr_scheduler=str(_pick("lr_scheduler", "cosine")),
        warmup_steps=int(_pick("warmup_steps", 100)),
        num_epochs=int(_pick("num_epochs", 3)),
        weight_decay=float(_pick("weight_decay", 0.01)),
        max_grad_norm=float(_pick("max_grad_norm", 1.0)),
        use_mxfp8=bool(_pick("use_mxfp8", False)),
        pack_sequences=bool(_pick("pack_sequences", False)),
        save_every=int(_pick("save_every", 500)),
        log_every=int(_pick("log_every", 10)),
        run_name=str(_pick("run_name", "")),
        output_dir=str(_pick("output_dir", "")),
    )

    # #65: checkpoint-aware restart (mirrors the sft command's --resume).
    if resume:
        from fusion_trainer.sft.trainer import (
            MLX_ADAPTER_BASE_DIR,
            find_latest_checkpoint,
        )

        run_name_eff = sft_cfg.run_name or "sft_run"
        adapter_dir = MLX_ADAPTER_BASE_DIR / sft_cfg.model / run_name_eff
        ckpt = find_latest_checkpoint(adapter_dir)
        if ckpt:
            sft_cfg.init_adapter_path = str(adapter_dir)
            logger.info(
                "BNUP SFT resume: checkpoint step %d found (%s)",
                ckpt[1],
                ckpt[0],
            )
        else:
            logger.warning(
                "BNUP SFT --resume: no checkpoint in %s; starting fresh",
                adapter_dir,
            )

    _path, _hub_dataset = _parse_dataset_arg(dataset)
    if not _path and not _hub_dataset and yaml_dataset.get("hub_dataset"):
        _hub_dataset = str(yaml_dataset["hub_dataset"])

    cfg = TrainerConfig(
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
            max_seq_length=int(yaml_dataset.get("max_seq_length", 4096)),
        ),
        sft=sft_cfg,
    )

    trainer = SFTTrainer(cfg)
    result = trainer.train()
    if not result or result.get("status") not in ("completed", "succeeded", "ok"):
        err = (result or {}).get("error", "training failed")
        typer.echo(f"✗ BNUP SFT failed: {err}", err=True)
        raise typer.Exit(1)
    typer.echo(f"✓ BNUP SFT done: {result.get('adapter_name', '')}")
    return result


def _invoke_rlsl(
    config: str,
    model: str,
    dataset: str,
    preset: str,
    method: str,
    logger: Any,
    init_adapter_path: str = "",
) -> dict:
    """Invoke the RLSL/DPO training path (reuses RLSLEngine).

    init_adapter_path: SFT adapter path to chain DPO/ORPO on (fusion-mlx #1149).
    Empty = fresh LoRA. Returns the engine result dict (carries adapter_path).
    """
    from fusion_trainer.config import DatasetConfig, SFTConfig, TrainerConfig
    from fusion_trainer.rlsl.pipeline import RLSLEngine
    from fusion_trainer.rlsl.presets import RLSL_PRESETS

    yaml_rlsl = _load_yaml_config(config, section="rlsl")
    yaml_sft = _load_yaml_config(config, section="sft")
    yaml_dataset = _load_yaml_config(config, section="dataset")

    preset_vals: dict = {}
    if preset:
        if preset not in RLSL_PRESETS:
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available: {', '.join(sorted(RLSL_PRESETS))}"
            )
        preset_vals = RLSL_PRESETS[preset]
        logger.info(
            "Loaded preset '%s' (%s)", preset, preset_vals.get("description", "")
        )

    def _pick(key: str, default: Any = None) -> Any:
        if key in preset_vals:
            return preset_vals[key]
        if key in yaml_rlsl:
            return yaml_rlsl[key]
        return default

    rlsl_kwargs: dict[str, Any] = {
        "method": method,
        "beta": float(_pick("beta", 0.1)),
        "batch_size": int(_pick("batch_size", 1)),
        "learning_rate": float(_pick("learning_rate", 5e-6)),
        "num_epochs": int(_pick("num_epochs", 1)),
        "log_every": int(_pick("log_every", 10)),
        "save_every": int(_pick("save_every", 200)),
        "run_name": str(_pick("run_name", "")),
        "output_dir": str(_pick("output_dir", "")),
    }
    # #1149: chain DPO/ORPO on a prior SFT adapter when provided.
    # RLSLConfig.init_adapter_path flows into the fusion-mlx DPO job config,
    # so the policy model loads with the SFT adapter (reference stays base-only).
    if init_adapter_path:
        rlsl_kwargs["init_adapter_path"] = init_adapter_path
        logger.info("BNUP %s: chaining on SFT adapter %s", method, init_adapter_path)

    from fusion_trainer.config import RLSLConfig

    rlsl_cfg = RLSLConfig(**rlsl_kwargs)

    sft_cfg = SFTConfig(
        model=model or str(yaml_sft.get("model", "")),
        lora_rank=int(yaml_sft.get("lora_rank", 16)),
        lora_alpha=float(yaml_sft.get("lora_alpha", 32.0)),
        lora_dropout=float(yaml_sft.get("lora_dropout", 0.05)),
    )

    _path, _hub_dataset = _parse_dataset_arg(dataset)
    if not _path and not _hub_dataset and yaml_dataset.get("hub_dataset"):
        _hub_dataset = str(yaml_dataset["hub_dataset"])

    cfg = TrainerConfig(
        dataset=DatasetConfig(
            path=_path,
            hub_dataset=_hub_dataset,
            max_seq_length=int(yaml_dataset.get("max_seq_length", 4096)),
        ),
        sft=sft_cfg,
        rlsl=rlsl_cfg,
    )

    engine = RLSLEngine(cfg)
    result = engine.run()
    if not result or result.get("status") not in ("completed", "succeeded", "ok"):
        err = (result or {}).get("error", "training failed")
        typer.echo(f"✗ BNUP {method} failed: {err}", err=True)
        raise typer.Exit(1)
    typer.echo(f"✓ BNUP {method} done: {result.get('adapter_name', '')}")
    return result


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
# Voice fine-tune (issue #74, fusion-mlx #1336)
# ---------------------------------------------------------------------------


@app.command(name="voice-train")
def voice_train(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Kokoro-family TTS model id (fusion-mlx model id, e.g. mlx-community/Kokoro-82M-bf16)",
    ),
    dataset: str = typer.Option(
        ...,
        "--dataset",
        "-d",
        help="Voice dataset dir: metadata.csv (file|text), metadata.jsonl ({audio,text}), or <name>.wav + <name>.txt pairs",
    ),
    voice_id: str = typer.Option(
        "",
        "--voice-id",
        help="Name for the trained voice package (auto-generated if empty; must not already exist)",
    ),
    steps: int = typer.Option(300, "--steps", help="SPSA optimization steps"),
    base_voice: str = typer.Option(
        "af_heart", "--base-voice", help="Stock Kokoro voice to start from"
    ),
    lang_code: str = typer.Option(
        "a", "--lang-code", help="Kokoro language code (a=US English, b=UK, ...)"
    ),
    max_samples: int = typer.Option(
        0, "--max-samples", help="Cap dataset pairs used (0 = all)"
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed (0 = unseeded)"),
    tenant_id: str = typer.Option(
        "default", "--tenant-id", help="Tenant owning the voice package"
    ),
    publish: bool = typer.Option(
        False,
        "--publish",
        help="Register the voice package in fusion-model-hub after training",
    ),
):
    """Train a TTS voice package (voice cloning) via fusion-mlx #1336.

    Delegates to POST /admin/api/fine-tune/voice/jobs — pure HTTP, no
    mlx_audio import. The trained package is served directly by fusion-mlx
    as voice_ref on POST /v1/audio/speech.
    """
    import logging

    from fusion_trainer.mlx_client import MLXFineTuneError
    from fusion_trainer.voice import VoiceTrainConfig, VoiceTrainer

    logger = logging.getLogger("fusion_trainer")
    cfg = VoiceTrainConfig(
        steps=steps,
        base_voice=base_voice,
        lang_code=lang_code,
        max_samples=max_samples,
        seed=seed,
    )
    trainer = VoiceTrainer(
        model=model,
        dataset_dir=dataset,
        voice_id=voice_id,
        tenant_id=tenant_id,
        config=cfg,
    )
    try:
        result = trainer.train()
    except MLXFineTuneError as exc:
        typer.echo(f"Error: voice training failed: {exc}", err=True)
        raise typer.Exit(1) from exc
    finally:
        trainer.close()

    status = result.get("status", "")
    if status == "completed":
        metrics = result.get("metrics", {})
        typer.echo("✅ Voice training complete")
        typer.echo(f"   voice_ref:  {result.get('voice_ref', '')}")
        typer.echo(f"   voice_path: {result.get('voice_path', '')}")
        typer.echo(
            f"   spectral distance: {metrics.get('initial_spectral_distance')} -> "
            f"{metrics.get('final_spectral_distance')} "
            f"(improved={metrics.get('improved')})"
        )
        if publish:
            from fusion_trainer.config import _default_hub_api_key, _default_hub_url
            from fusion_trainer.hub_client import HubPublishError, ModelHubClient

            hub = ModelHubClient(
                base_url=_default_hub_url(), api_key=_default_hub_api_key()
            )
            try:
                pub = hub.publish_voice_package(
                    name=result.get("voice_ref", "") or f"{model}-voice",
                    base_model=model,
                    voice_ref=result.get("voice_ref", ""),
                )
                if pub.get("published"):
                    typer.echo(
                        f"✓ Voice package published to hub: "
                        f"{pub.get('model', {}).get('id', '')}"
                        + (" (idempotent)" if pub.get("idempotent") else "")
                    )
            except HubPublishError as exc:
                typer.echo(f"⚠ Voice package publish failed (non-blocking): {exc}")
    else:
        typer.echo(
            f"Error: voice job ended {status}: {result.get('error', '')}",
            err=True,
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Eval flywheel (#76): fusion-eval trajectory ingestion + round registry
# ---------------------------------------------------------------------------


datasets_app = typer.Typer(help="Build trainer datasets from eval artifacts")
flywheel_app = typer.Typer(
    help="Flywheel round registry: checkpoint -> eval run -> pass rate"
)
app.add_typer(datasets_app, name="datasets")
app.add_typer(flywheel_app, name="flywheel")


@datasets_app.command(name="from-eval")
def datasets_from_eval(
    export_file: str = typer.Option(
        ...,
        "--export",
        "-e",
        help="fusion-eval trajectory export JSONL (fusion-eval export --run <id>)",
    ),
    output_file: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output SFT dataset path (messages-JSONL, fed to `ft sft --dataset`)",
    ),
    dpo_output: str = typer.Option(
        "",
        "--dpo-output",
        help="Optional: also mine DPO preference pairs to this path. Requires "
        "multiple trajectories per task (passed + failed siblings) — merge "
        "several single-arm exports into one JSONL, or run a multi-arm eval "
        "once fusion-eval supports it (fusion-eval#17). A one-trajectory-"
        "per-task export is structurally empty and raises (#78).",
    ),
    dpo_anyway: bool = typer.Option(
        False,
        "--dpo-anyway",
        help="Write the DPO output even when mining is structurally empty "
        "(single-arm export) — writes a 0-pair file with a loud warning",
    ),
):
    """Build an SFT dataset from passed eval trajectories (#76).

    Passed-only records become {"messages": [...]} samples the existing SFT
    path trains on as-is. A sidecar manifest `<output>.from_eval.json`
    records the export sha256 — `ft sft` picks it up automatically and
    reports the run's eval source.
    """
    from pathlib import Path

    from fusion_trainer.dataset.eval_ingest import (
        EvalIngestError,
        build_from_eval_dataset,
    )

    try:
        manifest = build_from_eval_dataset(
            Path(export_file),
            Path(output_file),
            Path(dpo_output) if dpo_output else None,
            dpo_allow_empty=dpo_anyway,
        )
    except EvalIngestError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"SFT dataset: {manifest['sft_samples']} samples -> {manifest['sft_path']} "
        f"({manifest['records_passed']} passed / {manifest['records_total']} records)"
    )
    if manifest["dpo_path"]:
        note = (
            " (STRUCTURALLY EMPTY — single-arm export)"
            if manifest.get("dpo_structurally_empty")
            else ""
        )
        typer.echo(
            f"DPO pairs:    {manifest['dpo_pairs']} -> {manifest['dpo_path']}{note}"
        )
    typer.echo(f"Arms:         {', '.join(manifest.get('arms', []))}")
    typer.echo(f"Export sha256: {manifest['export_sha256']}")
    typer.echo(
        f"Manifest:      {manifest['sft_path']}.from_eval.json "
        "(ft sft records this run's eval source automatically)"
    )


@datasets_app.command(name="from-rollouts")
def datasets_from_rollouts(
    sink_file: str = typer.Option(
        ...,
        "--sink",
        "-s",
        help="Gateway rollout-tap sink JSONL (schema: docs/rollout-tap.md, #86)",
    ),
    output_file: str = typer.Option(
        "",
        "--output",
        "-o",
        help="Output SFT dataset path (positive rollouts -> messages-JSONL, "
        "fed to `ft sft --dataset`)",
    ),
    grpo_output: str = typer.Option(
        "",
        "--grpo-output",
        help="Optional: also write a GRPO prompt set (every record with a "
        "prompt -> {prompt, reward, metadata}) for `ft rlsl --method grpo`",
    ),
    min_score: float = typer.Option(
        1.0,
        "--min-score",
        help="Minimum reward for the SFT positive set (default 1.0: "
        "task_success/user_accepted only)",
    ),
):
    """Build RFT/GRPO datasets from a gateway rollout-tap sink (#86).

    Positive rollouts (reward >= min-score) become {"messages": [...]}
    SFT samples; all prompted records become GRPO prompts with the
    observed outcome carried as the reward ref field. A sidecar manifest
    `<output>.from_rollouts.json` records the sink sha256 — `ft sft`
    picks it up automatically and reports the run's rollout source.
    """
    from pathlib import Path

    from fusion_trainer.dataset.rollout_ingest import (
        RolloutIngestError,
        build_from_rollouts,
    )

    if not output_file and not grpo_output:
        typer.echo(
            "Error: nothing to write — pass --output (SFT) and/or --grpo-output (GRPO prompts)",
            err=True,
        )
        raise typer.Exit(1)
    try:
        manifest = build_from_rollouts(
            Path(sink_file),
            Path(output_file) if output_file else None,
            Path(grpo_output) if grpo_output else None,
            min_score=min_score,
        )
    except RolloutIngestError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if manifest["sft_path"]:
        typer.echo(
            f"SFT dataset:  {manifest['sft_samples']} samples -> {manifest['sft_path']}"
        )
    if manifest["grpo_path"]:
        typer.echo(
            f"GRPO prompts: {manifest['grpo_prompts']} -> {manifest['grpo_path']}"
        )
    typer.echo(f"Signals:      {manifest['signals']}")
    typer.echo(f"Sink sha256:  {manifest['sink_sha256']}")
    typer.echo(
        f"Manifest:     {manifest['sft_path'] or manifest['grpo_path']}"
        ".from_rollouts.json "
        "(ft sft records this run's rollout source automatically)"
    )


@flywheel_app.command(name="record")
def flywheel_record(
    label: str = typer.Option(
        ..., "--label", help="Round label (e.g. baseline, round-1-sft, round-2-sft)"
    ),
    summary: str = typer.Option(
        ...,
        "--summary",
        help="fusion-eval run summary.json (results/<run_id>/summary.json)",
    ),
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        help="Trained adapter path (or base model id for a baseline round)",
    ),
    arm: str = typer.Option(
        "", "--arm", help="Eval arm to record (default: first arm)"
    ),
    export: str = typer.Option(
        "",
        "--export",
        help="Trajectory export JSONL this round trained on (sha256 recorded)",
    ),
    dataset: str = typer.Option(
        "", "--dataset", help="SFT dataset path used by this round"
    ),
    run_id: str = typer.Option(
        "", "--run-id", help="fusion-trainer run id for this round"
    ),
    baseline: bool = typer.Option(
        False,
        "--baseline",
        help="Baseline round: its export is the seed, not consumed training data (#82)",
    ),
    gate: bool = typer.Option(
        False,
        "--gate",
        help="Eval-gated promotion (#82): discard the round if pass_rate regresses "
        "vs the incumbent (best prior non-discarded round)",
    ),
    allow_reconsume: bool = typer.Option(
        False,
        "--allow-reconsume",
        help="Override the fresh-data check: re-train on an already-consumed "
        "export (entry flagged reconsumed, never silent)",
    ),
    cache_path: str = typer.Option(
        "",
        "--cache-path",
        help="Cache dir holding flywheel.json (default: FUSION_TRAINER_CACHE)",
    ),
):
    """Record a flywheel round from a fusion-eval summary (#76)."""
    from pathlib import Path

    from fusion_trainer.config import _default_cache_path
    from fusion_trainer.flywheel import FlywheelError, record_round

    try:
        entry = record_round(
            cache_path or _default_cache_path(),
            label=label,
            checkpoint=checkpoint,
            summary_path=Path(summary),
            arm=arm,
            export_path=export,
            dataset_path=dataset,
            run_id=run_id,
            baseline=baseline,
            gate=gate,
            allow_reconsume=allow_reconsume,
        )
    except FlywheelError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Recorded round {label!r}: eval_run={entry['eval_run_id']} "
        f"pass_rate={entry['pass_rate']:.4f} ({entry['passed']}/{entry['total']})"
        + (f" export={entry['export_sha256'][:12]}" if entry["export_sha256"] else "")
    )
    if entry.get("status") == "discarded":
        typer.echo(
            f"GATE DISCARDED round {label!r}: pass_rate {entry['pass_rate']:.4f} "
            f"regresses vs incumbent {entry['incumbent_pass_rate']:.4f} "
            f"({entry['regression']:+.4f}) — checkpoint NOT promoted (#82)"
        )
    elif entry.get("status") == "promoted":
        typer.echo(f"Gate PROMOTED round {label!r}")


@flywheel_app.command(name="mark")
def flywheel_mark(
    label: str = typer.Option(
        ..., "--label", help="Round label to mark (must be unique)"
    ),
    status: str = typer.Option(
        ..., "--status", help="promoted | discarded"
    ),
    cache_path: str = typer.Option(
        "",
        "--cache-path",
        help="Cache dir holding flywheel.json (default: FUSION_TRAINER_CACHE)",
    ),
):
    """Set a round's promotion status (#82) — backfill for pre-gate rounds."""
    from fusion_trainer.config import _default_cache_path
    from fusion_trainer.flywheel import FlywheelError, mark_round_status

    try:
        entry = mark_round_status(
            cache_path or _default_cache_path(), label=label, status=status
        )
    except FlywheelError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"Marked round {label!r} status={status} "
        f"(pass_rate={entry['pass_rate']:.4f})"
    )


@flywheel_app.command(name="status")
def flywheel_status(
    cache_path: str = typer.Option(
        "",
        "--cache-path",
        help="Cache dir holding flywheel.json (default: FUSION_TRAINER_CACHE)",
    ),
):
    """Print the pass-rate-vs-round curve (#76 acceptance evidence)."""
    from fusion_trainer.config import _default_cache_path
    from fusion_trainer.flywheel import format_status

    typer.echo(format_status(cache_path or _default_cache_path()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app()


if __name__ == "__main__":
    main()
