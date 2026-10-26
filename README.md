# Fusion-Trainer

**[English](README.md)** | **[中文](README_CN.md)**

![version](https://img.shields.io/badge/version-1.0.0rc2-blue)
![status](https://img.shields.io/badge/API-stable%20(SemVer)-green)

**Lightweight supervised + reinforcement learning trainer on Apple MLX, purpose-built for Mac M5 unified memory.**

Fusion-Trainer is a specialized training workbench that sits on top of [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) (the inference and model management layer). It provides a complete local training pipeline — dataset management, SFT fine-tuning, RLSL/DPO/ORPO alignment, and experiment tracking — without any cloud dependency.

---

## Architecture

```
fusion-model-hub (storage layer)
│
├── models/       ← Base model weights, LoRA adapters
├── datasets/     ← Dataset repository (sft, preference, reward, multimodal, domain)
└── snapshots/    ← Version snapshots & metadata
        │
        ▼
Fusion-Trainer (computation layer)
│
├── dataset/      ← Dataset loading, transforms, versioning
├── sft/          ← Supervised Fine-Tuning (full, LoRA, QLoRA)
├── rlsl/         ← RLSL / DPO / ORPO alignment pipeline
└── experiment/   ← Logging, metrics, visualization, tracking
        │
        ▼
fusion-* products (consumption layer)
    fusion-science, fusion-security, fusion-financial, fusion-code, ...
```

---

## Features

### Dataset Hub
- Import datasets from JSONL, Parquet, JSON, CSV
- Automatic deduplication, cleaning, format conversion
- Version snapshots for reproducibility
- Category-based organization (sft, preference, reward, multimodal, domain)
- Designed to interface with `fusion-model-hub` for shared storage

### Supervised Fine-Tuning (SFT)
- Real SFT loop over fusion-mlx HTTP fine-tune API (LoRA / DoRA / full)
- LoRA fine-tuning (rank 8–32) for 7B–32B models
- QLoRA ✅ — 4-bit quantized-base LoRA via fusion-mlx `FineTuneConfig` (`fine_tune_type=qlora`, `quantize_base=True`, `quant_bits=4`), landed in [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402). `_build_mlx_config` maps `method=qlora` to these fields.
- Domain-specific presets (code, finance, science)
- **fusion-router-light presets** — 1.5B LoRA + 3B full, for the gateway intent classifier / spec-decoding draft model
- MXFP8 mixed precision ✅ — [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)/#425 CLOSED: `use_mxfp8=True` passes through to fusion-mlx, which routes to the 8-bit QLoRA path (`quantize_base=True`, `quant_bits=8`, `fine_tune_type=qlora`) — 8-bit frozen base + LoRA for memory saving, NOT fp8 compute (MLX 0.32 has no fp8 dtype). `mxfp8+full` is a contradiction (full unfreezes the base) and still fails visibly. `use_mxfp8` defaults to `False`.
- **Sequence packing ✅** — `--pack-sequences` (or YAML `pack_sequences: true`): short samples are concatenated into rows of at most `max_seq_length` tokens with a block-diagonal attention mask (no cross-sample attention) and a per-token loss mask that still honours `mask_prompt` offsets. Big throughput win on short-sample corpora. Requires fusion-mlx >= 0.10.17 ([fusion-mlx#1400](https://github.com/dahai80/fusion-mlx/issues/1400) closed, `pack_sequences` field). Off by default: packing changes batch composition, so results are not byte-comparable with unpacked runs.
- **VLM fine-tune ✅** — [#55](https://github.com/dahai80/fusion-trainer/issues/55) closed, consuming the upstream VLM endpoint ([fusion-mlx#797](https://github.com/dahai80/fusion-mlx/issues/797)→[#800](https://github.com/dahai80/fusion-mlx/pull/800), v0.8.80). Set `model_type="vlm"` (or use the `sft_vlm_7b_qlora` preset) and `SFTTrainer` routes to `POST /admin/api/fine-tune/vlm/jobs`, which drives `mlx_vlm.trainer` SFT/LoRA inside fusion-mlx over HTTP — same pure-HTTP-delegation rule, fusion-trainer never imports `mlx-vlm`. The VLM path does NOT stage a chat-messages jsonl: `mlx_vlm.lora.load_dataset` consumes a HF dataset id or a local dir of images + jsonl directly, so pass the dataset reference via `dataset.path` (HF id or local dir) or `dataset.hub_dataset`. **VLM QLoRA caveat:** `VLMFineTuneConfig` has no quantize field, so `method="qlora"` on a VLM run degrades to plain LoRA with a loud warning — no base quantization occurs (mlx-vlm QLoRA semantics differ from mlx-lm and are not exposed upstream, #797).

### Reinforcement Learning from Supervised Learning (RLSL)
- **GRPO** ✅ — Group Relative Policy Optimization with sandbox reward (math + code). Two backends: `rft` (reward-filtered SFT approximation, default) and `grpo` (true policy-gradient via fusion-mlx `/api/fine-tune/grpo/jobs`, #363 Phase 2). See [GRPO backend](#5-run-grpo-with-sandbox-reward).
- **DPO** ✅ — Direct Preference Optimization via fusion-mlx `/admin/api/fine-tune/dpo/jobs` (landed in [fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399), DPO-list bug fixed in [#420](https://github.com/dahai80/fusion-mlx/issues/420)). `DPOTrainer.train` builds a `DPOConfig` (iters, beta, lora params from SFT config), submits preference pairs, and polls to completion; the adapter is written to `output_dir`.
- **ORPO** ✅ — Odds Ratio Preference Optimization via `/admin/api/fine-tune/orpo/jobs` ([fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)). Same shape as DPO but uses `lambda_odds` (mapped from `RLSLConfig.beta`) instead of the DPO temperature; no reference model.
- **Reward Model** ✅ — Bradley-Terry value-head training via fusion-mlx `/admin/api/fine-tune/reward/jobs` (landed in [fusion-mlx#424](https://github.com/dahai80/fusion-mlx/issues/424)). `RewardTrainer.train` builds a `RewardConfig` (iters from `reward_num_epochs`, lora params from SFT config), submits preference pairs, and polls to completion; the adapter is written with `reward_model=true`. `RewardTrainer.predict` (scoring) delegates to the reward/score endpoint ([fusion-mlx#431](https://github.com/dahai80/fusion-mlx/issues/431)), loading the trained RM adapter and returning scalar rewards.
- **RLSL** ✅ — Full closed-loop pipeline: Phase 1 trains a reward model ✅ (#424) → Phase 2 runs a **real policy-gradient GRPO** ✅ (#363) whose reward signal comes from that RM through the reward/score endpoint (#431). `_run_rlsl` hosts a local RM reward server (`rm_reward_server.py`) that GRPO calls back to score completions; the loop is `RM train (#424) → reward/score (#431) → GRPO (#363, PPO-clipped policy gradient)`.
- **RLSL presets** — `RLSL_PRESETS` (`fusion_trainer.rlsl.presets`) provides named GRPO hyperparameter bundles with `memory_estimate_gb` for M5 96GB planning: `rlsl_grpo_7b` (~22 GB, `rft` backend), `rlsl_grpo_14b` (~62 GB, `rft` backend), `rlsl_grpo_14b_true` (~62 GB, `grpo` backend, needs fusion-mlx #363 Phase 2). DPO/ORPO presets are intentionally omitted (hyperparameters map from SFT config). CLI `--preset` support is SFT-only today; RLSL presets are for programmatic use and as documented reference configs.
- Designed for M5 96GB unified memory (14B–32B models)

> **RLSL closed loop:** per the project rule, fusion-trainer talks **only** to fusion-mlx over HTTP and never imports `mlx-lm` directly. The full loop is now real — Phase 1 reward-model training via [fusion-mlx#424](https://github.com/dahai80/fusion-mlx/issues/424), reward/score endpoint via [#431](https://github.com/dahai80/fusion-mlx/issues/431), and Phase 2 PPO-clipped policy-gradient GRPO via [#363](https://github.com/dahai80/fusion-mlx/issues/363). `_run_rlsl` hosts a local RM reward server that GRPO calls back to score completions, closing the loop with no fake "completed" results. DPO, ORPO, GRPO (standalone) and reward-model training are also fully available.

### Experiment Tracking
- Metrics logging with TensorBoard support — **on by default**, streams `add_scalar` per metric/step to `output_dir/tensorboard` (set `enable_tensorboard=False` to disable); JSONL streaming always runs alongside
- Loss curve visualization (matplotlib)
- Experiment metadata and reproducibility tracking
- Git commit binding for full traceability

### BNUP Two-Stage Training Pipeline (issues #56–#59)
- **Data prep** ✅ — `ft bnup-prep --stage sft` validates raw BNUP math problems against the Socratic DSL v3 schema (via fusion-core `validate_dsl_str`) and packages valid samples as `<thinking>`+`<dsl>` ShareGPT JSONL. `ft bnup-prep --stage dpo` generates DPO preference pairs with configurable negative ratio.
- **Negative sample factory** ✅ — 5 static negative generators (broken JSON, direct-answer, wrong visual_type, missing tag, format drift) that each fail DSL validation, plus a 6th semantic generator `make_param_drift` (eval references a drifted parameter name — valid JSON/schema, fails only at eval execution, the E1 top failure class). Up to 5 negatives per sample, param_drift getting priority slots.
- **Model merge + export** ✅ — `ft merge --base <model> --adapter <path>` merges a trained LoRA/DoRA adapter into the base model via fusion-mlx `POST /v1/merge-adapter`, with optional 4-bit/8-bit post-merge quantization for Apple Silicon deployment.
- **Pipeline orchestration** ✅ — `ft bnup-train --stage pipeline` runs SFT → DPO → merge sequentially, auto-chaining DPO on the SFT adapter. Configs: `configs/bnup_sft_32b.yaml` (8-bit QDoRA, cutoff 4096), `configs/bnup_dpo_32b.yaml` (beta=0.1, sigmoid).
- **Checkpoint resume + engine-restart detection** ✅ (issue #65) — `ft sft --resume` / `ft bnup-train --stage sft|pipeline --resume` continues an interrupted run from the newest `NNNNNNN_adapters.safetensors` checkpoint (fusion-mlx #1182 `init_adapter_path`, weights loaded into the fresh LoRA layers, already-trained steps subtracted from `iters`). If fusion-mlx restarts mid-run, the trainer detects the lost/cancelled job, reports `interrupted` with the latest checkpoint, and never zombie-polls. A `job_state.json` liveness file tracks `job_id`/`adapter_dir`/`step`/`status` under the run output dir.

### TTS Voice Fine-Tune (issue #74)
- **Voice-pack training** ✅ — `ft voice-train` trains a Kokoro voice package from paired `(audio, transcript)` samples by delegating to fusion-mlx `POST /admin/api/fine-tune/voice/jobs` ([fusion-mlx#1336](https://github.com/dahai80/fusion-mlx/issues/1336)). The upstream backend is gradient-free SPSA over the voice-pack rows; the trainer submits the job, polls to terminal, and reports the upstream metrics (`improved`, `initial/final_spectral_distance`) plus the resulting `voice_ref`.
- **Dataset validation before submit** — accepts `metadata.csv` (`file|text`), `metadata.jsonl` (`{audio,text}`), or bare `<name>.wav` + `<name>.txt`; rejects missing/truncated audio and empty transcripts locally so a bad dataset never burns a job.
- **Serving** — the trained package lands in `~/.fusion-mlx/voices/<voice_id>/` and is served by fusion-mlx as `voice_ref` on `POST /v1/audio/speech` (unknown ref → 404, no silent fallback).
- **Hub registration** — `--publish` registers the package in fusion-model-hub as `model_type=audio` (`ModelHubClient.publish_voice_package`, 409-idempotent). Non-blocking: a hub outage never fails a training run.

> `--publish` requires a hub reachable at `--hub-url` / `FUSION_MODEL_HUB_URL`. Since fusion-gateway#316 the gateway on `:11432` proxies the hub's `/api/v1/models` (verified live: register 201, 409-idempotent, list through the gateway), so the default `http://localhost:11432` works with a gateway API key — no separate hub key needed. Pointing directly at the hub (`:11444` + `FUSION_MODEL_HUB_API_KEY`) still works.

---

## Hardware Reference

| Training Method | Model Size | Peak Memory | Recommended |
|---|---|---|---|
| Full FT | 7B | ~32 GB | M5 96GB ✓ |
| LoRA | 7B | ~16 GB | M5 96GB ✓ |
| QLoRA (4-bit) ✅ | 14B | ~28 GB | M5 96GB ✓ |
| QLoRA (4-bit) ✅ | 32B | ~48 GB | M5 96GB ✓ |
| RM Training ✅ | 14B | ~22 GB | M5 96GB ✓ |
| RLSL (GRPO, closed loop) ✅ | 14B | ~62 GB | M5 96GB ✓ |

Memory estimates assume 8-bit quantized-base LoRA (the MXFP8 path) on Apple Silicon. QLoRA is real ([#402](https://github.com/dahai80/fusion-mlx/issues/402)); RM Training is real ([#424](https://github.com/dahai80/fusion-mlx/issues/424)); RLSL closed loop (Phase 2 GRPO) is real ([#363](https://github.com/dahai80/fusion-mlx/issues/363), scored via [#431](https://github.com/dahai80/fusion-mlx/issues/431)). MXFP8 now routes to 8-bit QLoRA ([fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)/#425 CLOSED): `use_mxfp8=True` passes through and quantizes the frozen base to 8-bit + LoRA. With `use_mxfp8=False` (default) training runs at default precision, so actual peak memory may be higher than the table.

---

## Quick Start

### 1. Setup Environment

```bash
# Clone the repo
cd ~/fusion
git clone <repo-url> fusion-trainer
cd fusion-trainer

# One-click setup (with Chinese mirror)
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Activate virtual environment
source .venv/bin/activate
```

Verify the install:

```bash
ft --version     # ft 0.1.12
```

### 2. Prepare a Dataset

Dataset import/list/info is handled by the companion project **fusion-model-hub** — Fusion-Trainer has no `dataset` subcommand. Place a JSONL file (one JSON object per line) on disk and pass its path to `--dataset`, or use a hub dataset path (`hub:sft/code_alpaca` → `{FUSION_MODEL_HUB}/datasets/sft/code_alpaca.jsonl`).

```bash
# Local file (messages schema)
{"input": "Translate: hello", "output": "你好"}
{"input": "Summarize: ...", "output": "..."}

# Or a hub dataset (managed by fusion-model-hub)
fusion-trainer sft --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset hub:sft/code_alpaca
```

### 3. Run SFT Fine-Tuning

```bash
# Using a preset
./scripts/run_sft.sh --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_first_finetune

# --preset loads hyperparameters (method, lora_rank, lr, epochs, …) from
# SFT_PRESETS. Explicit CLI flags override the preset; fields the CLI does
# not expose (lora_alpha, dropout, target_modules, warmup, weight_decay,
# gradient_accumulation_steps) come from the preset. `--preset` works both
# on the shell wrapper and the CLI directly:
#   fusion-trainer sft --preset sft_7b_lora --model … --dataset …

# Or with a config file
fusion-trainer sft --config configs/sft_7b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl
```

`--method` selects the fine-tune type: `full`, `lora` (default), `dora` (DoRA via fusion-mlx `fine_tune_type=dora`, issue #402), or `qlora` (4-bit quantized-base LoRA). The former `use_dora` flag was a dead field and is removed — set `--method dora` instead.

Training runs through fusion-mlx's HTTP fine-tune API (`create_job → wait_for_job`), writing LoRA/DORA/full adapter weights to `~/.fusion-mlx/adapters/<model>/<run_name>/`. The fusion-router-light 1.5B preset (`sft_router_light_1_5b_lora`) has been validated end-to-end: a synthetic intent-classification dataset trained a LoRA adapter on `Llama-3.2-1B-Instruct-4bit` (1B used as the nearest available base to 1.5B) that correctly routes queries to `code`/`math`/`translate`/`chat`/`summary` (train_loss 0.011, val_loss 0.0026, ~3 GB peak, 10s on M5).

> **Netlayer compliance (issue #16):** MLX calls default to port `11434` (fusion-mlx direct). Since fusion-gateway v0.8.4 ([fusion-gateway#30](https://github.com/dahai80/fusion-gateway/issues/30), closed) the gateway on `:11432` proxies `/admin/api/fine-tune/*` 1:1 to fusion-mlx (including `/jobs/models`, fixed via [fusion-mlx#397](https://github.com/dahai80/fusion-mlx/issues/397)) — verified live: jobs/adapters/models all 200 through `:11432`. The default stays `11434` (one hop less, trainer's own admin key); set `FUSION_MLX_PORT=11432` to route fine-tune traffic through the gateway (the gateway auth chain validates the client key and injects the upstream credential). The hub (`hub_url`) stays on `:11432` — the gateway proxies `/api/v1/*` to fusion-model-hub.

### 4. Run Alignment (DPO)

```bash
./scripts/run_rlsl.sh --model qwen2.5-7b-4bit \
    --dataset /path/to/preference.jsonl \
    --method dpo \
    --run-name my_dpo_alignment
```

### 5. Run GRPO with Sandbox Reward

```bash
fusion-trainer rlsl --config configs/rlsl_14b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/grpo_prompts.jsonl \
    --method grpo \
    --sandbox code \
    --group-size 8 \
    --grpo-iters 1 \
    --run-name my_grpo_run
```

GRPO prompt dataset (one JSON object per line):

```json
{"prompt": "Solve: 12 * 7 = ?", "answer": "84"}
{"prompt": "Write a function that returns the factorial of n", "tests": "assert f(5) == 120"}
```

`--sandbox` defaults to `off` (no code exec). Set `code` for code-mode reward (execs the model's completion + dataset `tests` in a subprocess) or `math` for answer-normalization reward. **Code mode execs untrusted output** — it requires `FUSION_CODE_SANDBOX_TRUSTED=1` and is not a true OS sandbox; see `fusion_trainer/rlsl/sandbox.py`. `auto` still picks `code` when a `tests` field is present / `math` when an `answer` field is present, but is opt-in only. See [GRPO backend](#5-run-grpo-with-sandbox-reward).

`reward_code` runs the generated code under a real isolation layer on macOS via the built-in seatbelt sandbox (`sandbox-exec`): network denied, data reads confined to the interpreter prefix + stdlib + tmpfs workspace (metadata reads stay broad for path resolution — directory structure, not file contents), killed on timeout. Reads of `~/.ssh`, `/etc`, and sibling project source are denied. Override the backend with `FUSION_CODE_SANDBOX_BACKEND` (`auto` default | `seatbelt` | `rlimit`); on non-macOS it falls back to rlimits with a loud warning. The `FUSION_CODE_SANDBOX_TRUSTED=1` gate still applies regardless of backend. See `fusion_trainer/rlsl/isolation.py`.

### 6. Publish a Trained Adapter to fusion-model-hub (issue #49)

After a successful LoRA run, the produced adapter can be registered in [fusion-model-hub](https://github.com/dahai80/fusion-models-hub) so the gateway can index it (`GET /api/v1/models?model_type=lora`). This is **off by default** — opt in with `--publish` (or `publish_adapter: true` in YAML):

```bash
# Publish immediately after SFT completes
fusion-trainer sft --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_adapter \
    --publish

# Same for RLSL / DPO / ORPO / GRPO
fusion-trainer rlsl --model qwen2.5-7b-4bit \
    --dataset /path/to/pref.jsonl \
    --method dpo \
    --publish

# Retroactively publish a completed run that was trained without --publish
fusion-trainer publish <run_id>
```

Under the hood this calls `POST /api/v1/models` with `model_type="lora"` + `base_model_id` pointing at the run's base model (the `ModelHubClient` in `fusion_trainer/hub_client.py`). Publish **failures warn loudly but never block the training result** — a hub outage flips the published flag to false, logs the error, and exits 0 (fail-visible, never silent).

**Configuration** (`TrainerConfig`, env-var overridable):

| Field | Env var | Default |
|---|---|---|
| `hub_url` | `FUSION_MODEL_HUB_URL` | `http://localhost:11432` (gateway → model-hub) |
| `hub_api_key` | `FUSION_MODEL_HUB_API_KEY` | `""` (no auth) |
| `publish_adapter` | `FUSION_TRAINER_PUBLISH=1` | `False` |

The `FUSION_TRAINER_PUBLISH=1` env override forces publish on regardless of config. The **GUI path** now forwards publish fields natively ([fusion-agent-studio#277](https://github.com/dahai80/fusion-agent-studio/issues/277) + [fusion-studio#390](https://github.com/dahai80/fusion-studio/issues/390) CLOSED): fusion-agent-studio `TrainingService._build_trainer_config` passes `publish_adapter`/`hub_url`/`hub_api_key` and the Trainer panel exposes them, so GUI-triggered runs publish natively. The env override remains as a fallback. The CLI `--publish` path is unaffected.

#### Audit log & api-key gate (enterprise)

`RunManager` keeps an **append-only JSONL audit trail** at `{cache_path}/audit.jsonl` — one line per mutating operation (`start_sft`, `start_rlsl`, `stop_run`, `publish_run`, `delete_adapter`), each record `{timestamp, actor, action, run_id, detail, ok}`. Actor comes from `FUSION_TRAINER_ACTOR` (default `local`). The file is append-only; it is an accountability trail, not a forensic non-repudiation store (no signing — tamper-protection is out of scope for a local-first file).

An optional **api-key gate** (`FUSION_TRAINER_API_KEY` set) rejects mutating `RunManager` ops with `PermissionDeniedError` on a mismatched key (constant-time compare). When the env var is **unset**, local-trust mode allows all ops — appropriate for a single-user Mac. Set it for shared / multi-operator deployments. `info()` reports whether the gate is active and the audit-log path.

| Env var | Purpose | Default |
|---|---|---|
| `FUSION_TRAINER_ACTOR` | actor recorded in each audit entry | `local` |
| `FUSION_TRAINER_API_KEY` | gate mutating RunManager ops; unset = allow all | unset (local-trust) |

`start_sft`/`start_rlsl`/`stop_run`/`delete_adapter` accept an optional `api_key` arg; `publish_run` accepts `trainer_api_key` (kept distinct from the hub `api_key`, so a deployment can gate ops without sharing the hub secret).

#### Metrics export (enterprise)

`ExperimentLogger.export_prometheus(path, run_id=, extra_series=)` and the module-level `export_prometheus_snapshot(path, metrics, run_id=, extra_series=)` write a **Prometheus text-format** (`.prom`) snapshot — one gauge per scalar metric key:

```
# HELP fusion_trainer_train_loss run-1
# TYPE fusion_trainer_train_loss gauge
fusion_trainer_train_loss{run="run-1"} 0.1234
```

`RunManager` emits `{cache_path}/runs/<run_id>/metrics.prom` at run end, including `run_steps_total`, `run_total_steps`, and `run_status_code` (`0=running,1=completed,2=failed,3=cancelled`). Metric names are sanitized to the Prometheus `[a-zA-Z_:][a-zA-Z0-9_:]*` grammar; non-scalar values are skipped. This is a **file export** — a node-exporter `textfile_collector` or a sidecar ships it. A live scrape endpoint lives upstream in fusion-agent-studio (not in fusion-trainer, which is a pure computation layer with no HTTP server).

#### GRPO backend (`grpo_backend`)

`RLSLConfig.grpo_backend` selects the GRPO execution path:

| backend | behavior | endpoint |
|---|---|---|
| `rft` (default) | **reward-filtered SFT approximation** (RFT/RAFT-style): on-policy sample → sandbox score → top-k → SFT/LoRA job. No per-token policy gradient, no importance ratio — it converges toward higher-reward outputs but is NOT true GRPO. | `/admin/api/fine-tune/jobs` |
| `grpo` / `policy_gradient` | real policy-gradient GRPO, runs server-side in fusion-mlx; trainer hosts a local reward HTTP endpoint that fusion-mlx calls back (`reward_server.py` → `sandbox.reward_completion`) | `/api/fine-tune/grpo/jobs` |

`policy_gradient` is an alias of `grpo` (normalized in `TrainerConfig.__post_init__`, issue #75) — the honest name for when you want actual policy gradient. `grpo` requires fusion-mlx #363 Phase 2. Set in YAML (`rlsl.grpo_backend: grpo`) or pass via config. Per the project rule the trainer talks **only** to fusion-mlx's HTTP API — no direct `mlx-lm` imports.

#### Checkpoint selection via fusion-bench (`--eval-select`, issue #75)

Lowest val loss is a weak proxy for "best checkpoint". `ft sft --eval-select` closes the loop: after training, every numbered checkpoint (`NNNNNN_adapters.safetensors`) is benchmarked through fusion-bench's task API and the winner is picked by bench score:

```bash
ft sft --model Qwen3-0.6B-4bit --dataset data.jsonl --save-every 100 \
    --eval-select --bench-url http://localhost:11467 \
    --bench-api-key <jwt> --bench-tenant-id default
```

- Deterministic selection: bench score (higher wins, tie → later step) → val_loss (lower wins, tie → later step) → last checkpoint.
- The full table — checkpoint, step, val_loss, bench task id, bench score, chosen — persists atomically to `checkpoint_selection.json` next to the adapters and is attached to the run result.
- Fail-visible, never fatal: a bench outage (or no bench configured) marks every row `unavailable`, records a `fallback_reason`, and selects on val_loss. A selection is never silently wrong.
- Bench scores the true checkpoint weights: fusion-bench v0.4.11 ([fusion-bench#29](https://github.com/dahai80/fusion-bench/issues/29), closed) forwards the per-candidate `adapter_path` to fusion-mlx chat as the `adapters` field. Requires fusion-bench >= 0.4.11; against an older bench the score reflects the base model, so prefer the val_loss basis then.

---

## CLI Reference

| Command | Description |
|---|---|
| `fusion-trainer sft` | Run supervised fine-tuning (add `--publish` to publish the adapter to model-hub on completion, `--resume` to continue an interrupted run from its newest checkpoint, issue #65, `--eval-select` to benchmark every checkpoint via fusion-bench and pick the winner, issue #75, `--pack-sequences` to pack short samples into full rows, fusion-mlx#1400) |
| `fusion-trainer rlsl` | Run DPO/ORPO/RLSL/GRPO alignment (add `--publish` to publish) |
| `fusion-trainer publish <run_id>` | Publish a completed run's LoRA adapter to fusion-model-hub (issue #49) |
| `fusion-trainer bnup-prep --stage sft\|dpo` | Prepare BNUP training data (SFT DSL validation or DPO pair generation, issues #56/#57) |
| `fusion-trainer bnup-train --stage sft\|dpo\|pipeline` | Run BNUP two-stage training (SFT → DPO → merge, issue #59; `--resume` continues an interrupted SFT stage, issue #65) |
| `fusion-trainer merge --base <model> --adapter <path>` | Merge a trained adapter into the base model via fusion-mlx (issue #58) |
| `fusion-trainer voice-train --model <kokoro> --dataset <dir>` | Train a TTS voice pack (add `--publish` to register it with model-hub, issue #74) |
| `fusion-trainer info` | System info & hardware check |
| `fusion-trainer --version` / `-V` | Print version (`ft <version>`) |
| `fusion-trainer --help` | Show full help |

> Dataset import/list/info is handled by the companion project **fusion-model-hub**, not by Fusion-Trainer — there is no `dataset` subcommand here.

---

## Load / Stability Verification (enterprise #4)

Consecutive-run stability is a class of bug the mock-based suite cannot catch —
a leaked file handle, a singleton that accumulates state, or an event log that
appends across runs only surfaces on the Nth back-to-back run. Two artifacts
close that gap:

**`scripts/load_test.sh`** — an operator-facing shell soak: N consecutive SFT
runs (default 5) against a live fusion-mlx, with `~/claude-home/fusion-mlx/start.sh doctor`
health checks between each round. Exits non-zero on the first failed round so a
CI `&&` chain catches it. Requires `FUSION_MLX_ADMIN_TOKEN` and a loaded text
model; override with `ROUNDS`, `MODEL`, `PORT`.

```bash
FUSION_MLX_ADMIN_TOKEN=<admin-key> ./scripts/load_test.sh
ROUNDS=10 ./scripts/load_test.sh
```

**`tests/test_load_stability.py`** — a pytest harness (markers `integration` +
`load`, gated by `FUSION_TRAINER_LOAD=1`) that drives N consecutive SFT runs
through `RunManager` and asserts no state leak: each run reaches `running` then
`completed`, produces a non-empty `events.jsonl`, and the registry remembers
prior runs. Default 3 rounds (`FUSION_TRAINER_LOAD_ROUNDS`). Auto-skips without
a live server.

```bash
FUSION_TRAINER_LOAD=1 FUSION_MLX_ADMIN_TOKEN=<key> \
    pytest tests/test_load_stability.py -m load -v
```

## Nightly Real-Training CI (issue #75)

The per-push CI runs the mocked suite on ubuntu — MLX has no Linux wheel, so
every test needing a live fusion-mlx engine is env-gated
(`FUSION_TRAINER_REAL=1`) and silently skipped there.
`.github/workflows/nightly.yml` ungates them for real: on a macOS arm64
runner it boots a real fusion-mlx with `Qwen3-0.6B-4bit` and runs the
real-training suite (`tests/test_sft_trainer.py::TestRealFusionMLXIntegration`)
plus the contract suite (`tests/test_contract_integration.py`, marker
`integration`) against actual gradient loops. Runs on schedule (02:30
Beijing) and on demand via `workflow_dispatch`:

```bash
gh workflow run nightly -R dahai80/fusion-trainer
```

Upstream contract drift (field renames, job-lifecycle changes) is caught
within a day instead of at the next manual verification.

## API Stability (v1.0)

As of **1.0.0**, the public API surface is frozen under
[Semantic Versioning](https://semver.org/). The frozen surface — CLI
subcommands, `TrainerConfig` fields, `RunManager` methods,
`MLXFineTuneClient` methods, and public exception types — is documented in
[`docs/API_STABILITY.md`](docs/API_STABILITY.md). Breaking changes to that
surface require a major-version bump; additions are minor; fixes are patch.
Internal modules (`_`-prefixed, `presets.py`, `sandbox.py` reward internals,
`experiment/` logger internals) are not frozen and may change in any release.

---

## Product Integration

Fusion-Trainer is designed as the **model iteration factory** for the entire Fusion product ecosystem:

| Product | Training Use Case |
|---|---|
| fusion-science | Scientific multimodal model fine-tuning |
| fusion-security | Vulnerability detection model training |
| fusion-code-modernization | Legacy code migration model iteration |
| fusion-financial | Financial domain model alignment |
| fusion-health | Medical domain model fine-tuning |

**Feedback loop:** Each Fusion Agent produces interaction samples → samples are filtered and stored in the dataset hub → used for the next round of SFT/RLSL training → updated models are deployed back to the products.

---

## GUI Integration (fusion-studio)

Fusion-Trainer is drivable from the **fusion-studio** desktop client (SwiftUI) — no CLI needed for day-to-day training. The GUI talks to fusion-trainer **in-process through fusion-agent-studio** over the studio UDS (`/tmp/fusion-studio.sock`, JSON-RPC 2.0), never importing `mlx-lm`/`mlx-vlm` directly.

```
fusion-studio (SwiftUI TrainerView)
  └─ TrainerBridge → IPCClient.trainer* (UDS)
       └─ fusion-agent-studio TrainerDispatcher (trainer.* RPC, PR #167)
            └─ fusion-trainer RunManager (fusion_trainer/run_manager.py)
                 └─ MLXFineTuneClient → fusion-mlx HTTP (real gradient loops)
```

**RPC surface** (`trainer.*`, served by `agent_runtime/dispatchers/trainer.py`):

| Method | Purpose |
|---|---|
| `trainer.start_sft` / `trainer.start_rlsl` | Launch an async run from a `TrainerConfig` dict |
| `trainer.runs.list` / `trainer.runs.status_full` | Run registry + full config/status |
| `trainer.runs.progress` | Live progress events since a step (polled 2s in GUI) |
| `trainer.runs.stop` | Cancel a running job |
| `trainer.presets.list` | SFT/RLSL presets (`SFT_PRESETS`/`RLSL_PRESETS`) |
| `trainer.datasets.list` / `trainer.datasets.preview` | Hub dataset registry + sample preview |
| `trainer.adapters.list` / `trainer.adapters.delete` | Trained adapter registry + cleanup |
| `trainer.info_full` | Version + hardware + endpoint info |

**GUI panels** (fusion-studio `Modules/Trainer/TrainerView.swift`, 6 tabs): 训练运行 (run list + live metrics chart + stop), 启动训练 (SFT/RLSL config form), 预设, 数据集 (+preview), 适配器 (+delete), 环境信息. Reached via the icon rail → **Trainer** section.

**Upstream PRs:** fusion-agent-studio #167 (RPC widening) + fusion-studio #176 (GUI panel). See `architecture/fusion-trainer-gui.md` for the full high-fidelity mockup spec.

---

## Trainer Console (bridge, #71)

A self-contained four-section web console — datasets, training, merge & publish, base-vs-tuned evaluation — served by the **trainer-bridge**, a thin FastAPI facade around the fusion_trainer library (`bridge/`). The browser never sees the fusion-mlx admin key; the bridge owns it plus the `RunManager` lifecycle (runs are background threads in the bridge process, so a console refresh never kills a training).

```bash
export FUSION_TRAINER_BRIDGE_KEY=<token>     # bearer token browser -> bridge
./scripts/trainer-bridge.sh start            # serves http://127.0.0.1:8790/
./scripts/trainer-bridge.sh stop             # (engine key read from ~/.fusion-mlx/settings.json)
```

- **Endpoints:** `POST/GET/DELETE /datasets`, `GET /presets`, `POST /runs` + `GET /runs` + `POST /runs/{id}/stop|merge`, `POST /publish` + `GET /published`, `POST /eval`, `GET /health`, and `GET /` serving the console page (`bridge/trainer.html`). Loopback bind only; every data route requires the bearer token (401 fail-closed).
- **Publish** promotes a run's **merged standalone model** (written to `~/.fusion-mlx/models` at merge time, so published models auto-enter the fusion-mlx `/v1/models` catalog); `POST /publish {"run_id": ""}` reverts to the base model. Hot-mounting raw adapters remains deliberately unused — publish promotes a merged standalone model, which is simpler to serve and version; the original blocker (2-level adapter registry missing slash-id models) is fixed upstream ([fusion-mlx#1317](https://github.com/dahai80/fusion-mlx/issues/1317), PR #1327 — registry now walks 3 levels and lists slash-id adapters, verified live through `:11432`).
- **Preset registry:** `~/.fusion-trainer-bridge/presets.json` (seeded from `SFT_PRESETS` on first start, re-read per request — edit presets without cutting a release; corrupt/empty file falls back to built-ins with a warning).
- **Memory budget:** the bridge computes an explicit `memory_limit_gb` (85% safety line − resident − 6 GB headroom, floor 8) because fusion-mlx's default (60% of physical) assumes a dedicated training box and rejects jobs when large models are already serving.
- **Tiny-run reporting:** `steps_per_report` is capped to `iters` so smoke runs still emit progress events, and completed runs report `step == total_steps` instead of `0/N`.
- **Tests:** `bridge/test_server.py` (21 cases, FastAPI TestClient, mocked RunManager/MLX client): `python -m pytest bridge/test_server.py -q`
- **Shell module registration (#72):** on start the bridge registers `bridge/manifest.json` with the fusion-gateway shell registry and heartbeats every 15s, so the fusion-web-shell sidebar lists **训练控制台 / Trainer** and the gateway proxy `http://127.0.0.1:11432/shell/fusion-trainer/` is the single user-facing entry (stop the bridge → the sidebar entry greys as unreachable). Registration is K1-issued-key auth: the body carries no `api_key`, the gateway issues an `fm-` module key returned exactly once, and the bridge persists it to `~/.fusion-trainer-bridge/web_module_key` (0600) — heartbeats and proxied requests authenticate with it (inbound auth dual-accepts the static bridge key for direct loopback access). Env-gated via `FUSION_SHELL_GATEWAY_URL` + `FUSION_GATEWAY_ADMIN_KEY`; `trainer-bridge.sh` defaults the URL to the local gateway and reads the admin key from the gateway config when unset. `bridge/test_shell_registry.py` (14 cases, mocked HTTP) covers the loop.

---

## Releases & Change Management

Releases follow [Semantic Versioning](https://semver.org/) against the frozen
API surface in [`docs/API_STABILITY.md`](docs/API_STABILITY.md). The full
release process — branch strategy (`release/vX.Y.Z`), version-bump rules,
release-cut steps, review checklist, hotfix flow, and rollback policy — is
documented in [`docs/RELEASE_PROCESS.md`](docs/RELEASE_PROCESS.md).
`CHANGELOG.md` holds the per-version release notes; git tags `vX.Y.Z` mark the
immutable release points.

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

Built on top of [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) and [Apple MLX](https://github.com/ml-explore/mlx).