# Fusion-Trainer

**[English](README.md)** | **[中文](README_CN.md)**

**基于 Apple MLX 的轻量化强化 + 监督学习训练工作台，专为 Mac M5 统一内存设计。**

Fusion-Trainer 是构建在 [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx)（推理层与模型管理层）之上的专业训练工作台。提供完整的本地训练流水线——数据集管理、SFT 微调、RLSL/DPO/ORPO 对齐、实验追踪——全程无需云端依赖。

---

## 架构

```
fusion-model-hub（存储层）
│
├── models/       ← 基座模型权重、LoRA 适配器
├── datasets/     ← 数据集仓库（sft, preference, reward, multimodal, domain）
└── snapshots/    ← 版本快照与元数据
        │
        ▼
Fusion-Trainer（计算层）
│
├── dataset/      ← 数据集加载、转换、版本管理
├── sft/          ← 监督微调（全量、LoRA、QLoRA）
├── rlsl/         ← RLSL / DPO / ORPO 对齐流水线
└── experiment/   ← 日志、指标、可视化、追踪
        │
        ▼
fusion-* 产品（消费层）
    fusion-science, fusion-security, fusion-financial, fusion-code, ...
```

---

## 功能特性

### 数据集管理中心
- 支持 JSONL、Parquet、JSON、CSV 格式导入
- 自动去重、清洗、格式转换
- 版本快照支持实验复现
- 按类别组织（sft、preference、reward、multimodal、domain）
- 与 `fusion-model-hub` 对接实现统一存储

### 监督微调（SFT）
- 7B 及以下模型全参数微调
- 7B–32B 模型 LoRA 微调（rank 8–32）
- QLoRA ⏳ — **上游阻塞**：fusion-mlx `FineTuneConfig` 无 `quantize_base`/`qlora`（见 [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)）。`_build_mlx_config` 在 `method=qlora` 时抛 `SFTUnsupportedError`，而非静默回退 lora —— 在上游落地前请用 `lora` 或 `full`。
- 领域专属预设配置（代码、金融、科学）
- MXFP8 混合精度 ⏳ — **上游阻塞**：fusion-mlx `FineTuneConfig` 无 `mxfp8` 字段（见 [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)）。当前以默认精度训练；`use_mxfp8=True` 会发出警告（峰值内存可能超过预设估算），而非静默丢弃该开关。

### 强化监督学习（RLSL）
- **GRPO** ✅ — 带沙盒奖励的组相对策略优化（数学 + 代码）。两种后端：`rft`（奖励过滤 SFT 近似，默认）与 `grpo`（经 fusion-mlx `/api/fine-tune/grpo/jobs` 的真实策略梯度，#363 Phase 2）。
- **DPO** ⏳ — 直接偏好优化。**上游阻塞**：需要 fusion-mlx DPO 端点（见 [fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)）。在此之前 `DPOTrainer.train` 抛 `RLSLUnsupportedError` —— 请改用 GRPO。
- **ORPO** ⏳ — 胜率比偏好优化。**上游阻塞**（[fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)）；`ORPOTrainer.train` 抛 `RLSLUnsupportedError`。
- **RLSL** ⏳ — 完整 PPO-light 流水线（含奖励模型训练）。**上游阻塞**（[fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)）；`RewardTrainer.train`/`predict` 抛 `RLSLUnsupportedError`。
- **RLSL 预设** —— `RLSL_PRESETS`（`fusion_trainer.rlsl.presets`）提供命名 GRPO 超参束，含 `memory_estimate_gb` 供 M5 96GB 规划：`rlsl_grpo_7b`（~22 GB，`rft` 后端）、`rlsl_grpo_14b`（~62 GB，`rft` 后端）、`rlsl_grpo_14b_true`（~62 GB，`grpo` 后端，需 fusion-mlx #363 Phase 2）。DPO/ORPO/RLSL 预设因上游阻塞（#399）有意省略。CLI `--preset` 当前仅 SFT 支持；RLSL 预设供编程式使用与作为文档化参考配置。
- 专为 M5 96GB 统一内存设计（14B–32B 模型）

> **为何 DPO/ORPO/RLSL 抛错而非训练：** 按项目规则，fusion-trainer **只**通过 HTTP 对接 fusion-mlx，绝不直接 import `mlx-lm`。这些方法需要模型内 logprob + 梯度更新，必须由 fusion-mlx 承载。在 [fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399) 落地前，它们显式失败而非静默返回假 "completed"。GRPO 当前已完整可用。

`reward_code` 在 macOS 上通过内置 seatbelt 沙箱（`sandbox-exec`）运行生成的代码：禁用网络、数据读取限定在解释器前缀 + 标准库 + tmpfs 工作区（元数据读取保持宽松以解析路径 —— 只泄露目录结构，不泄露文件内容）、超时即杀。读取 `~/.ssh`、`/etc` 及兄弟项目源码将被拒绝。用 `FUSION_CODE_SANDBOX_BACKEND` 覆盖后端（默认 `auto` | `seatbelt` | `rlimit`）；非 macOS 回退到 rlimit 并大声告警。无论何种后端，`FUSION_CODE_SANDBOX_TRUSTED=1` 显式开关仍然生效。详见 `fusion_trainer/rlsl/isolation.py`。

### 实验追踪
- 指标日志记录，支持 TensorBoard —— **默认开启**，每个指标/步通过 `add_scalar` 流式写入 `output_dir/tensorboard`（设 `enable_tensorboard=False` 关闭）；JSONL 流式始终并行写入
- Loss 曲线可视化（matplotlib）
- 实验元数据与可复现追踪
- Git commit 绑定，全程可追溯

---

## 硬件参考

| 训练方法 | 模型规模 | 峰值内存 | 推荐度 |
|---|---|---|---|
| 全量微调 | 7B | ~32 GB | M5 96GB ✓ |
| LoRA | 7B | ~16 GB | M5 96GB ✓ |
| QLoRA (4-bit) ⏳ | 14B | ~28 GB | M5 96GB ✓ |
| QLoRA (4-bit) ⏳ | 32B | ~48 GB | M5 96GB ✓ |
| 奖励模型训练 | 14B | ~22 GB | M5 96GB ✓ |
| RLSL (PPO-light) | 14B | ~62 GB | M5 96GB ✓ |

以上估算基于 MXFP8 混合精度（⏳ = 上游阻塞，见 [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)；当前以默认精度训练，实际峰值内存可能更高）。

---

## 快速开始

### 1. 部署环境

```bash
# 进入目录
cd ~/fusion/fusion-trainer

# 一键部署（国内镜像）
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 激活虚拟环境
source .venv/bin/activate
```

### 2. 准备数据集

数据集的导入/列出/查看由配套项目 **fusion-model-hub** 负责 —— Fusion-Trainer 没有 `dataset` 子命令。将 JSONL 文件（每行一个 JSON 对象）放在磁盘上并通过 `--dataset` 传入路径，或使用 hub 数据集路径（`hub:sft/code_alpaca` → `{FUSION_MODEL_HUB}/datasets/sft/code_alpaca.jsonl`）。

```bash
# 本地文件（messages schema）
{"input": "Translate: hello", "output": "你好"}
{"input": "Summarize: ...", "output": "..."}

# 或使用 hub 数据集（由 fusion-model-hub 管理）
fusion-trainer sft --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset hub:sft/code_alpaca
```

### 3. 运行 SFT 微调

```bash
# 使用预设
./scripts/run_sft.sh --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_first_finetune

# --preset 从 SFT_PRESETS 加载超参数（method、lora_rank、lr、epochs 等）。
# 显式 CLI 参数覆盖预设；CLI 未暴露的字段（lora_alpha、dropout、
# target_modules、warmup、weight_decay、gradient_accumulation_steps）取自预设。
# --preset 同时适用于 shell 脚本和 CLI：
#   fusion-trainer sft --preset sft_7b_lora --model … --dataset …

# 或使用配置文件
fusion-trainer sft --config configs/sft_7b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl
```

`--method` 选择微调类型：`full`（全量）、`lora`（默认）、`dora`（DoRA，经 fusion-mlx `fine_tune_type=dora`，issue #402）、`qlora`（4-bit 量化基座 LoRA）。旧的 `use_dora` 开关是无效死字段，已移除——请改用 `--method dora`。

### 4. 运行 DPO 对齐

```bash
./scripts/run_rlsl.sh --model qwen2.5-7b-4bit \
    --dataset /path/to/preference.jsonl \
    --method dpo \
    --run-name my_dpo_alignment
```

---

## CLI 命令参考

| 命令 | 说明 |
|---|---|
| `fusion-trainer sft` | 运行监督微调 |
| `fusion-trainer rlsl` | 运行 DPO/ORPO/RLSL/GRPO 对齐 |
| `fusion-trainer publish <run_id>` | 发布已完成 run 的 LoRA 适配器到 Hub |
| `fusion-trainer info` | 查看系统信息与硬件检查 |
| `fusion-trainer --help` | 查看完整帮助 |

> 数据集的导入/列出/查看请使用配套项目 **fusion-model-hub**（Fusion-Trainer 无 `dataset` 子命令）。

---

## 发布训练好的适配器到 fusion-model-hub（issue #49）

一次成功的 LoRA 运行后，产出的适配器可注册到 [fusion-model-hub](https://github.com/dahai80/fusion-models-hub)，使网关可索引它（`GET /api/v1/models?model_type=lora`）。默认**关闭**，用 `--publish`（或 YAML 中 `publish_adapter: true`）开启：

```bash
# SFT 完成后立即发布
fusion-trainer sft --preset sft_7b_lora \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/data.jsonl \
    --run-name my_adapter \
    --publish

# 补发一个未带 --publish 训练的已完成 run
fusion-trainer publish <run_id>
```

底层调用 `POST /api/v1/models`，`model_type="lora"` + `base_model_id` 指向 run 的基座模型（`fusion_trainer/hub_client.py` 中的 `ModelHubClient`）。发布**失败只警告、不阻断训练结果** —— Hub 不可用时把 published 置 false、记录错误、退出 0（fail-visible，绝不静默）。

**配置**（`TrainerConfig`，可用环境变量覆盖）：

| 字段 | 环境变量 | 默认值 |
|---|---|---|
| `hub_url` | `FUSION_MODEL_HUB_URL` | `http://localhost:11432`（网关 → model-hub） |
| `hub_api_key` | `FUSION_MODEL_HUB_API_KEY` | `""`（无鉴权） |
| `publish_adapter` | `FUSION_TRAINER_PUBLISH=1` | `False` |

`FUSION_TRAINER_PUBLISH=1` 环境变量覆盖专为 **GUI 路径**存在：fusion-agent-studio `TrainingService._build_trainer_config` 当前会丢弃它构建的配置里的 `publish_adapter`/`hub_url`/`hub_api_key`（上游 [fusion-agent-studio#268](https://github.com/dahai80/fusion-agent-studio/issues/268) 追踪），导致 GUI 触发的 run 即使操作者想发布也从不发布。在 #268 落地前，设置 `FUSION_TRAINER_PUBLISH=1`（配合 `FUSION_MODEL_HUB_URL`/`FUSION_MODEL_HUB_API_KEY`）可强制 GUI run 发布。CLI `--publish` 路径不受影响。

---

## 负载 / 稳定性验证（企业级 #4）

连续多轮运行的稳定性是一类 mock 测试套件无法捕获的缺陷 —— 文件句柄泄漏、单例累积状态、事件日志跨轮追加，这类问题只会在连续第 N 次运行时才暴露。两件产物补齐该缺口：

**`scripts/load_test.sh`** —— 面向运维的 shell 压测：对实时 fusion-mlx 连续 N 轮 SFT（默认 5 轮），每轮之间用 `~/claude-home/fusion-mlx/start.sh doctor` 做健康检查。首轮失败即非零退出，便于 CI `&&` 链捕获。需 `FUSION_MLX_ADMIN_TOKEN` + 已加载文本模型；可用 `ROUNDS`、`MODEL`、`PORT` 覆盖。

```bash
FUSION_MLX_ADMIN_TOKEN=<admin-key> ./scripts/load_test.sh
ROUNDS=10 ./scripts/load_test.sh
```

**`tests/test_load_stability.py`** —— pytest 压测（marker `integration`+`load`，由 `FUSION_TRAINER_LOAD=1` 开关）经 `RunManager` 连续驱动 N 轮 SFT，断言无状态泄漏：每轮到达 `running` 再 `completed`、产出非空 `events.jsonl`、注册表记得前序 run。默认 3 轮（`FUSION_TRAINER_LOAD_ROUNDS`）。无实时服务器时自动跳过。

```bash
FUSION_TRAINER_LOAD=1 FUSION_MLX_ADMIN_TOKEN=<key> \
    pytest tests/test_load_stability.py -m load -v
```

---

## 产品联动

Fusion-Trainer 作为整条 Fusion 产品矩阵的**模型迭代工厂**：

| 产品 | 训练用途 |
|---|---|
| fusion-science | 科研多模态模型微调 |
| fusion-security | 漏洞检测模型训练 |
| fusion-code-modernization | 老旧代码迁移模型迭代 |
| fusion-financial | 金融领域模型对齐 |
| fusion-health | 医疗领域模型微调 |

**闭环反馈：** 各 Fusion Agent 产生交互样本 → 筛选后存入数据集仓库 → 用于下一轮 SFT/RLSL 训练 → 更新模型部署回产品。

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 致谢

基于 [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) 和 [Apple MLX](https://github.com/ml-explore/mlx) 构建。