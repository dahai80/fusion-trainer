---
doc_id: fusion-trainer/product-page
product: fusion-trainer
class: platform
entry: service
doc_type: product-page
title: { zh: "fusion-trainer 产品页", en: "fusion-trainer product page" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/product-page.md
---

# fusion-trainer 产品页 / Product Page

fusion-trainer 是 Apple Silicon (MLX) 上的监督 + 强化学习训练层。它不自带推理引擎：所有梯度更新都通过 fusion-mlx 的 HTTP fine-tune API (`localhost:11434`) 完成，本仓库只做数据组织、任务编排、奖励计算与实验追踪。

fusion-trainer is the supervised + reinforcement-learning training layer on Apple Silicon (MLX). It ships no inference engine of its own: every gradient update goes through the fusion-mlx HTTP fine-tune API (`localhost:11434`); this repo provides dataset organization, job orchestration, reward computation, and experiment tracking.

## 平台定位 / Platform positioning

```yaml
platform:
  serves:                       # 下游消费者 / who consumes this platform
    - fusion-science            # domain fine-tuning via CLI presets
    - fusion-security           # vulnerability-detection model training
    - fusion-code               # code model iteration (sft_code_7b_lora preset)
    - fusion-studio             # GUI training via trainer.* RPC
    - fusion-agent-studio       # TrainerDispatcher -> RunManager bridge
    - fusion-web-shell          # Trainer console module (bridge manifest, #72)
  builds_on:                    # 依赖的上游平台 / upstream platforms assumed
    - fusion-mlx                # HTTP fine-tune API :11434 (only MLX touchpoint)
    - fusion-model-hub          # dataset + adapter registry via gateway :11432
    - fusion-gateway            # hub API proxy + shell module routing :11432
    - fusion-bench              # checkpoint selection scoring :11467 (optional)
  hosts:                        # 本平台托管的运行物 / what this platform hosts
    - SFT / LoRA / DoRA / QLoRA training jobs
    - RLSL / GRPO / DPO / ORPO alignment jobs
    - GRPO sandbox reward execution (seatbelt isolation, code/math reward)
    - trainer-bridge console (FastAPI, 127.0.0.1:8790)
    - adapter merge + publish pipeline to fusion-model-hub
  integration_surface:          # 集成接口 / how consumers integrate
    - CLI: fusion-trainer (alias ft) subcommands
    - Python: RunManager / TrainerConfig / MLXFineTuneClient (frozen in docs/API_STABILITY.md)
    - HTTP: bridge REST API on 127.0.0.1:8790 (bearer-token)
    - RPC: trainer.* methods via fusion-agent-studio UDS dispatcher
    - Artifacts: adapters under ~/.fusion-mlx/adapters, metrics.prom, audit.jsonl
  extension_points:             # 扩展点 / where consumers extend
    - YAML config files (configs/*.yaml) merged over preset hyperparameters
    - SFT_PRESETS / RLSL_PRESETS registries (bridge presets.json is hot-editable)
    - dataset formats: JSONL (messages / preference pairs / GRPO prompts), hub: refs
    - env-var overrides for every service endpoint (FUSION_MLX_PORT, FUSION_MODEL_HUB_URL, ...)
```

## 能力清单 / Capabilities

### SFT 微调 / Supervised fine-tuning

消费者不再需要自己实现 LoRA 训练循环、checkpoint 轮转、训练数据校验与 loss 曲线记录 —— 提交一个 preset + 数据集路径即可拿到可发布的 adapter。约束：训练真实发生在 fusion-mlx 进程内，trainer 侧不持有梯度状态；4-bit QLoRA 依赖 fusion-mlx `FineTuneConfig` (`fine_tune_type=qlora`)，MXFP8 走 8-bit QLoRA 路径而非 fp8 计算 (MLX 无 fp8 dtype)。

Consumers no longer write a LoRA training loop, checkpoint rotation, dataset validation, or loss-curve bookkeeping — submit a preset plus a dataset path and receive a publishable adapter. Constraint: training actually runs inside the fusion-mlx process; the trainer holds no gradient state. 4-bit QLoRA depends on the fusion-mlx `FineTuneConfig` (`fine_tune_type=qlora`); MXFP8 maps to the 8-bit QLoRA path, not fp8 compute (MLX has no fp8 dtype).

### 对齐训练 / Alignment (RLSL / GRPO / DPO / ORPO)

消费者不再需要实现偏好对生成、GRPO 采样-打分-筛选循环或奖励服务器 —— `rlsl` 子命令按 `--method` 分派到对应 fusion-mlx 端点。约束：默认 `rft` 后端是 reward-filtered SFT 近似而非真实策略梯度；真实 GRPO 需要 fusion-mlx #363 Phase 2 并且回调本机奖励端点。code 奖励会执行模型输出，需显式 `FUSION_CODE_SANDBOX_TRUSTED=1`。

Consumers no longer implement preference-pair generation, the GRPO sample-score-filter loop, or a reward server — the `rlsl` subcommand dispatches by `--method` to the matching fusion-mlx endpoint. Constraint: the default `rft` backend is a reward-filtered SFT approximation, not true policy gradient; real GRPO requires fusion-mlx #363 Phase 2 and a local reward endpoint callback. Code reward executes model output and requires `FUSION_CODE_SANDBOX_TRUSTED=1`.

### 数据集管理 / Dataset handling

消费者不再需要写 JSONL/Parquet 去重、清洗、格式转换或 `hub:` 路径解析 —— `hub:sft/code_alpaca` 直接解析到 model-hub 数据集仓库。约束：数据集的导入/列表/信息由 fusion-model-hub 负责，本仓库没有 `dataset` 子命令；本地文件必须是一行一个 JSON 对象。

Consumers no longer write JSONL/Parquet dedup, cleaning, format conversion, or `hub:` path resolution — `hub:sft/code_alpaca` resolves into the model-hub dataset repo. Constraint: dataset import/list/info belongs to fusion-model-hub; this repo has no `dataset` subcommand. Local files must be one JSON object per line.

### 适配器合并与发布 / Merge & publish

消费者不再需要实现 adapter → 独立模型的合并调用或 model-hub 注册 —— `--publish` 或 `fusion-trainer publish <run_id>` 在训练完成后自动注册 (`POST /api/v1/models`, `model_type=lora`)。约束：发布默认关闭；发布失败只告警不阻塞训练结果 (exit 0)。

Consumers no longer implement the adapter-to-standalone-model merge call or model-hub registration — `--publish` or `fusion-trainer publish <run_id>` registers the adapter after training (`POST /api/v1/models`, `model_type=lora`). Constraint: publish is off by default; a publish failure warns loudly but never blocks the training result (exit 0).

### 实验追踪 / Experiment tracking

消费者不再需要自建 run 注册表、事件日志或 Prometheus 指标导出 —— 每个 run 落盘 `events.jsonl`、`metrics.prom`，所有变更操作追加 `audit.jsonl`。约束：metrics 是文件导出，需要 textfile collector 或 sidecar 上报，本平台无 HTTP scrape 端点。

Consumers no longer build a run registry, event log, or Prometheus metric export — every run writes `events.jsonl` and `metrics.prom`, and every mutating op appends to `audit.jsonl`. Constraint: metrics are file exports; a textfile collector or sidecar ships them — this platform has no HTTP scrape endpoint.

## 集成入口 / Integration entry points

| Entry | Consumer | Contract |
|---|---|---|
| `fusion-trainer` / `ft` CLI | humans, domain pipelines | `sft`, `rlsl`, `publish`, `merge`, `bnup-prep`, `bnup-train`, `voice-train`, `info` |
| `RunManager` (Python) | fusion-agent-studio, tests | `start_sft` / `start_rlsl` / `stop_run` / `publish_run` / `delete_adapter` |
| trainer-bridge HTTP :8790 | fusion-web-shell, browser console | `/runs`, `/datasets`, `/publish`, `/eval`, `/health` (bearer token) |
| `trainer.*` RPC | fusion-studio GUI | UDS via fusion-agent-studio `TrainerDispatcher` |

## 边界 / Non-goals

- 不做推理服务：推理与模型目录属于 fusion-mlx。 / No inference serving: inference and the model catalog belong to fusion-mlx.
- 不做数据集门户：导入/浏览属于 fusion-model-hub。 / No dataset portal: import/browse belongs to fusion-model-hub.
- 不做云端训练：全部计算在本机统一内存内。 / No cloud training: all computation uses local unified memory.
