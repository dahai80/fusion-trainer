---
doc_id: fusion-trainer/get-started
product: fusion-trainer
class: platform
entry: service
doc_type: get-started
title: { zh: "fusion-trainer 快速上手", en: "fusion-trainer get started" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/get-started.md
---

# fusion-trainer 快速上手 / Get Started

前置条件 / Prerequisites:

- macOS on Apple Silicon, Python 3.12 (`requires-python >= 3.12`, pyproject.toml)
- fusion-mlx running on `localhost:11434` with a loaded text model
- (optional) fusion-gateway on `:11432` for hub publish

## 1. 安装 / Install

```bash
cd ~/fusion
git clone https://github.com/dahai80/fusion-trainer.git fusion-trainer
cd fusion-trainer
./scripts/setup_env.sh          # creates .venv, installs deps (mirror-aware)
source .venv/bin/activate
ft --version                    # ft 1.0.0rc23
ft info                         # hardware + endpoint check
```

## 2. 准备数据集 / Prepare a dataset

本仓库没有 `dataset` 子命令 —— 导入与浏览属于 fusion-model-hub。准备一个 JSONL 文件（每行一个 JSON 对象）:

This repo has no `dataset` subcommand — import/browse belongs to fusion-model-hub. Prepare a JSONL file (one JSON object per line):

```json
{"input": "Translate: hello", "output": "你好"}
```

Or reference a hub dataset: `hub:sft/code_alpaca` resolves to `{FUSION_MODEL_HUB}/datasets/sft/code_alpaca.jsonl`.

## 3. 运行 SFT / Run SFT

```bash
# Option A: preset + CLI flags
./scripts/run_sft.sh --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_first_finetune

# Option B: config file
ft sft --config configs/sft_7b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl
```

预设 / Presets: `sft_7b_lora`, `sft_7b_full`, `sft_14b_qlora`, `sft_32b_qlora`, `sft_code_7b_lora` (see `scripts/run_sft.sh` header and `configs/`). CLI flags override preset values. Adapters are written to `~/.fusion-mlx/adapters/<model>/<run_name>/`.

## 4. 对齐训练 / Alignment (DPO)

```bash
./scripts/run_rlsl.sh --model qwen2.5-7b-4bit \
    --dataset /path/to/preference.jsonl \
    --method dpo \
    --run-name my_dpo_alignment
```

## 5. 发布 adapter / Publish

```bash
ft sft --preset sft_7b_lora --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl --run-name my_adapter --publish
# or retroactively:
ft publish <run_id>
```

Publish 需要 hub 配置（默认关闭）/ Publish needs hub config (off by default): `FUSION_MODEL_HUB_URL` (default `http://localhost:11432`), optional `FUSION_MODEL_HUB_API_KEY`.

## 6. 训练控制台 / Trainer console (optional)

```bash
export FUSION_TRAINER_BRIDGE_KEY=<token>
./scripts/trainer-bridge.sh start   # http://127.0.0.1:8790/
./scripts/trainer-bridge.sh stop
```

## 下一步 / Next steps

- Full CLI reference: `ft --help` and `docs/api-reference.md`
- Config fields: `docs/config-reference.md`
- Alignment details (GRPO sandbox): `docs/architecture.md`
