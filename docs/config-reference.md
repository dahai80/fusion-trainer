---
doc_id: fusion-trainer/config-reference
product: fusion-trainer
class: platform
entry: service
doc_type: config-reference
title: { zh: "fusion-trainer 配置参考", en: "fusion-trainer config reference" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/config-reference.md
---

# fusion-trainer 配置参考 / Config Reference

配置三层叠加 / Three layers, later wins: 内置 preset (`SFT_PRESETS`/`RLSL_PRESETS`) -> YAML 文件 (`--config configs/*.yaml`) -> CLI 显式 flags。自 #80 起服务端点通过 `fusion_core.config` 单点解析，env 名不变。

Configuration layers, later wins: built-in presets (`SFT_PRESETS`/`RLSL_PRESETS`) -> YAML file (`--config configs/*.yaml`) -> explicit CLI flags. Since #80, service endpoints resolve single-point through `fusion_core.config` with unchanged env names.

## 示例文件 / Example files

`configs/default.yaml`（共享基础）, `configs/sft_7b.yaml`, `configs/sft_14b.yaml`, `configs/sft_32b.yaml`, `configs/rlsl_14b.yaml`, `configs/bnup_sft_32b.yaml`, `configs/bnup_dpo_32b.yaml`.

## YAML 字段 / YAML fields

### 共享基础 (configs/default.yaml)

| Field | Default | Description |
|---|---|---|
| `dataset.val_ratio` | `0.05` | Validation split ratio |
| `dataset.max_samples` | `0` | 0 = use all samples |
| `dataset.max_seq_length` | `4096` | Sequence length cap (tokens) |
| `dataset.remove_duplicates` | `true` | Dedup before training |
| `dataset.input_field` / `output_field` | `input` / `output` | JSONL field names |
| `dataset.hub_dataset` | `""` | e.g. `sft/code_alpaca` |
| `device` | `gpu` | MLX device |
| `seed` | `42` | Random seed |
| `model_hub_path` | `""` | Defaults to `$FUSION_MODEL_HUB` or `~/fusion-model-hub` (read-only reference) |
| `cache_path` | `""` | Defaults to `./runtime_cache` (run registry, events, audit) |

### `sft` block (configs/sft_7b.yaml)

| Field | Default | Description |
|---|---|---|
| `model` | `""` | fusion-mlx model alias, e.g. `qwen2.5-7b-4bit` |
| `model_type` | `lm` | `lm` or `vlm` (VLM routes to the fusion-mlx VLM endpoint) |
| `method` | `lora` | `full` / `lora` / `dora` / `qlora` |
| `lora_rank` | `16` | LoRA rank (8-32 supported range) |
| `lora_alpha` / `lora_dropout` / `lora_target_modules` | `32.0` / `0.05` / q,k,v,o_proj | Preset-only fields (not CLI-exposed) |
| `batch_size` / `gradient_accumulation_steps` | `1` / `4` | Effective batch = product |
| `learning_rate` / `lr_scheduler` / `warmup_steps` | `2.0e-5` / `cosine` / `100` | Optimizer schedule |
| `num_epochs` | `3` | Training epochs |
| `weight_decay` / `max_grad_norm` | `0.01` / `1.0` | Regularization |
| `use_mxfp8` | `false` | 8-bit QLoRA path (not fp8 compute); `true`+`full` fail-stops |
| `pack_sequences` | `false` | Pack short samples into `max_seq_length` rows; requires fusion-mlx >= 0.10.17 |
| `publish_adapter` | `false` | Register adapter to model-hub on completion |
| `save_every` / `log_every` | `500` / `10` | Checkpoint / logging interval |
| `run_name` / `output_dir` | preset value / `""` | Run identity and output location |

### 环境变量 / Environment variables

| Env var | Default | Purpose |
|---|---|---|
| `FUSION_MLX_PORT` | `11434` | fusion-mlx port; set `11432` to route fine-tune traffic through fusion-gateway |
| `FUSION_MLX_ADMIN_TOKEN` | resolved via `fusion_core.config` | Engine admin key; falls back to `~/.fusion-mlx/settings.json` `auth.api_key` |
| `FUSION_MODEL_HUB_URL` | `http://localhost:11432` | Hub endpoint (gateway -> model-hub) |
| `FUSION_MODEL_HUB_API_KEY` | `""` | Hub auth — direct-hub calls need a hub-issued `fmh-*` key (`fmh keys create`), not the gateway admin key |
| `FUSION_TRAINER_PUBLISH` | unset | `1` forces publish on regardless of config |
| `FUSION_TRAINER_ACTOR` | `local` | Actor recorded in audit entries |
| `FUSION_TRAINER_API_KEY` | unset | Gate mutating RunManager ops; unset = local-trust allow-all |
| `FUSION_CODE_SANDBOX_TRUSTED` | unset | Must be `1` to enable code-mode reward execution |
| `FUSION_CODE_SANDBOX_BACKEND` | `auto` | `seatbelt` (macOS) or `rlimit` fallback |
| `FUSION_TRAINER_BRIDGE_KEY` | unset | Bearer token for the trainer-bridge (:8790) |
| `FUSION_SHELL_GATEWAY_URL` | local gateway | Gateway endpoint for shell module registration |
| `FUSION_ADMIN_KEY` | via `fusion_core.config.admin_key` | Gateway admin key for shell registration |
| `FUSION_TRAINER_LOAD` | unset | `1` enables the load-stability pytest harness |
| `FUSION_TRAINER_REAL` | unset | `1` enables real-training (live fusion-mlx) tests |

预设调整 / Preset overrides without a release: edit `~/.fusion-trainer-bridge/presets.json` (bridge console); corrupt/empty file falls back to built-ins with a warning.
