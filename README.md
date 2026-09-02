# Fusion-Trainer

**[English](README.md)** | **[中文](README_CN.md)**

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
- MXFP8 mixed precision ⏳ — **Upstream-blocked**: fusion-mlx `FineTuneConfig` accepts `mxfp8` but the training loop still rejects it (service.py raises `ValueError`, tracked in [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)). `use_mxfp8` defaults to `False`; an explicit `True` **fail-stops** (raises `MLXFineTuneError` at job-build) rather than silently downgrading. Training runs at default precision today.

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

Memory estimates assume MXFP8 mixed precision on Apple Silicon. QLoRA is now real ([#402](https://github.com/dahai80/fusion-mlx/issues/402)); RM Training is now real ([#424](https://github.com/dahai80/fusion-mlx/issues/424)); RLSL closed loop (Phase 2 GRPO) is now real ([#363](https://github.com/dahai80/fusion-mlx/issues/363), scored via [#431](https://github.com/dahai80/fusion-mlx/issues/431)). MXFP8 is still upstream-blocked (⏳, see [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)); `use_mxfp8` defaults `False` and explicit `True` fail-stops, so training runs at default precision today and actual peak memory may be higher than the table.

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

> **Netlayer compliance (issue #16):** MLX calls default to port `11434` (fusion-mlx direct). The fusion-gateway on `:11432` routes to fusion-mlx but currently proxies only `/v1/*` (model verification), not `/admin/api/fine-tune/*` (job creation) — tracked upstream in fusion-gateway#30. Override the default with `FUSION_MLX_PORT` (e.g. `11432` to route through the gateway once it proxies fine-tune paths). The hub (`hub_url`) stays on `:11432` — the gateway proxies `/api/v1/*` to fusion-model-hub.

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

The `FUSION_TRAINER_PUBLISH=1` env override exists for the **GUI path**: fusion-agent-studio `TrainingService._build_trainer_config` currently drops `publish_adapter`/`hub_url`/`hub_api_key` from the config it builds (tracked at [fusion-agent-studio#268](https://github.com/dahai80/fusion-agent-studio/issues/268)), so a GUI-triggered run never published even when the operator wanted it to. Setting `FUSION_TRAINER_PUBLISH=1` (plus `FUSION_MODEL_HUB_URL`/`FUSION_MODEL_HUB_API_KEY`) forces publish on for GUI runs until #268 lands. The CLI `--publish` path is unaffected.

#### GRPO backend (`grpo_backend`)

`RLSLConfig.grpo_backend` selects the GRPO execution path:

| backend | behavior | endpoint |
|---|---|---|
| `rft` (default) | reward-filtered SFT approximation: on-policy sample → sandbox score → top-k → SFT/LoRA job | `/admin/api/fine-tune/jobs` |
| `grpo` | real policy-gradient GRPO, runs server-side in fusion-mlx; trainer hosts a local reward HTTP endpoint that fusion-mlx calls back (`reward_server.py` → `sandbox.reward_completion`) | `/api/fine-tune/grpo/jobs` |

`grpo` requires fusion-mlx #363 Phase 2. Set in YAML (`rlsl.grpo_backend: grpo`) or pass via config. Per the project rule the trainer talks **only** to fusion-mlx's HTTP API — no direct `mlx-lm` imports.

---

## CLI Reference

| Command | Description |
|---|---|
| `fusion-trainer sft` | Run supervised fine-tuning (add `--publish` to publish the adapter to model-hub on completion) |
| `fusion-trainer rlsl` | Run DPO/ORPO/RLSL/GRPO alignment (add `--publish` to publish) |
| `fusion-trainer publish <run_id>` | Publish a completed run's LoRA adapter to fusion-model-hub (issue #49) |
| `fusion-trainer info` | System info & hardware check |
| `fusion-trainer --version` / `-V` | Print version (`ft <version>`) |
| `fusion-trainer --help` | Show full help |

> Dataset import/list/info is handled by the companion project **fusion-model-hub**, not by Fusion-Trainer — there is no `dataset` subcommand here.

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

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

Built on top of [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) and [Apple MLX](https://github.com/ml-explore/mlx).