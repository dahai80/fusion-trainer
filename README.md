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
- Full-parameter fine-tuning for 7B and smaller models
- LoRA fine-tuning (rank 8–32) for 7B–32B models
- QLoRA with 4-bit quantization for 32B models on M5 96GB
- Domain-specific presets (code, finance, science)
- MXFP8 mixed precision training (MLX-native)

### Reinforcement Learning from Supervised Learning (RLSL)
- **DPO** — Direct Preference Optimization (no reward model needed)
- **ORPO** — Odds Ratio Preference Optimization (single-stage SFT + alignment)
- **RLSL** — Full PPO-light pipeline with reward model training
- Designed for M5 96GB unified memory (14B–32B models)

### Experiment Tracking
- Metrics logging with TensorBoard support
- Loss curve visualization (matplotlib)
- Experiment metadata and reproducibility tracking
- Git commit binding for full traceability

---

## Hardware Reference

| Training Method | Model Size | Peak Memory | Recommended |
|---|---|---|---|
| Full FT | 7B | ~32 GB | M5 96GB ✓ |
| LoRA | 7B | ~16 GB | M5 96GB ✓ |
| QLoRA (4-bit) | 14B | ~28 GB | M5 96GB ✓ |
| QLoRA (4-bit) | 32B | ~48 GB | M5 96GB ✓ |
| RM Training | 14B | ~22 GB | M5 96GB ✓ |
| RLSL (PPO-light) | 14B | ~62 GB | M5 96GB ✓ |

All estimates assume MXFP8 mixed precision on Apple Silicon.

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

### 2. Import a Dataset

```bash
fusion-trainer dataset import /path/to/data.jsonl \
    --name my_dataset \
    --category sft \
    --format jsonl
```

### 3. Run SFT Fine-Tuning

```bash
# Using a preset
./scripts/run_sft.sh --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_first_finetune

# Or with a config file
fusion-trainer sft --config configs/sft_7b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl
```

### 4. Run Alignment (DPO)

```bash
./scripts/run_rlsl.sh --model qwen2.5-7b-4bit \
    --dataset /path/to/preference.jsonl \
    --method dpo \
    --run-name my_dpo_alignment
```

---

## CLI Reference

| Command | Description |
|---|---|
| `fusion-trainer sft` | Run supervised fine-tuning |
| `fusion-trainer rlsl` | Run DPO/ORPO/RLSL alignment |
| `fusion-trainer dataset import` | Import a dataset into the hub |
| `fusion-trainer dataset list` | List datasets in the hub |
| `fusion-trainer dataset info` | Show dataset details |
| `fusion-trainer info` | System info & hardware check |
| `fusion-trainer --help` | Show full help |

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

## License

Apache 2.0

## Acknowledgments

Built on top of [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) and [Apple MLX](https://github.com/ml-explore/mlx).