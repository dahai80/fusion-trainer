---
doc_id: fusion-trainer/readme-cn
product: fusion-trainer
class: platform
entry: service
doc_type: readme
title: { zh: "fusion-trainer 中文说明", en: "fusion-trainer readme (Chinese)" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - README_CN.md
---

# Fusion-Trainer

**[English](README.md)** | **[中文](README_CN.md)**

![version](https://img.shields.io/badge/version-1.0.0rc2-blue)
![status](https://img.shields.io/badge/API-stable%20(SemVer)-green)

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
- QLoRA ✅ — 4-bit 量化基座 LoRA，经 fusion-mlx `FineTuneConfig`（`fine_tune_type=qlora`、`quantize_base=True`、`quant_bits=4`），见 [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)。`_build_mlx_config` 将 `method=qlora` 映射到这些字段。
- 领域专属预设配置（代码、金融、科学）
- MXFP8 混合精度 ✅ — [fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)/#425 已关闭：`use_mxfp8=True` 透传给 fusion-mlx，路由到 8-bit QLoRA 路径（`quantize_base=True`、`quant_bits=8`、`fine_tune_type=qlora`）—— 8-bit 冻结基座 + LoRA 省显存，非 fp8 计算（MLX 0.32 无 fp8 dtype）。`mxfp8+full` 矛盾（full 解冻基座），仍 fail-visible。`use_mxfp8` 默认 `False`。
- **序列打包 ✅** — `--pack-sequences`（或 YAML `pack_sequences: true`）：短样本拼接为至多 `max_seq_length` token 的行，块对角注意力掩码（无跨样本注意力），逐 token loss 掩码仍遵守 `mask_prompt` 偏移。短样本语料吞吐大幅提升。需 fusion-mlx >= 0.10.17（[fusion-mlx#1400](https://github.com/dahai80/fusion-mlx/issues/1400) 已关闭，`pack_sequences` 字段）。默认关闭：打包改变 batch 组成，结果与不打包运行不可逐字节对比。
- **VLM 视觉语言微调 ✅** — [#55](https://github.com/dahai80/fusion-trainer/issues/55) 已关闭，消费上游 VLM 端点（[fusion-mlx#797](https://github.com/dahai80/fusion-mlx/issues/797)→[#800](https://github.com/dahai80/fusion-mlx/pull/800)，v0.8.80）。设置 `model_type="vlm"`（或使用 `sft_vlm_7b_qlora` 预设），`SFTTrainer` 即路由到 `POST /admin/api/fine-tune/vlm/jobs`，该端点在 fusion-mlx 内经 HTTP 驱动 `mlx_vlm.trainer` SFT/LoRA —— 同样的纯 HTTP 委托规则，fusion-trainer 不导入 `mlx-vlm`。VLM 路径**不**暂存 chat-messages jsonl：`mlx_vlm.lora.load_dataset` 直接消费 HF 数据集 id 或本地图片+jsonl 目录，因此通过 `dataset.path`（HF id 或本地目录）或 `dataset.hub_dataset` 传递数据集引用。**VLM QLoRA 注意：** `VLMFineTuneConfig` 无量化字段，故 VLM 运行中 `method="qlora"` 会降级为普通 LoRA 并发出明显告警 —— 不进行基座量化（mlx-vlm 的 QLoRA 语义与 mlx-lm 不同，上游未暴露，#797）。

### 强化监督学习（RLSL）
- **GRPO** ✅ — 带沙盒奖励的组相对策略优化（数学 + 代码）。两种后端：`rft`（奖励过滤 SFT 近似，默认）与 `grpo`（经 fusion-mlx `/api/fine-tune/grpo/jobs` 的真实策略梯度，#363 Phase 2）。两种后端均端到端可用（#751 已关闭，worker 重新绑定 `generation_stream`）。
- **DPO** ✅ — 直接偏好优化，经 fusion-mlx `/admin/api/fine-tune/dpo/jobs`（落地于 [fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)，DPO-list bug 修复于 [#420](https://github.com/dahai80/fusion-mlx/issues/420)）。`DPOTrainer.train` 构建 `DPOConfig`（iters、beta、来自 SFT 配置的 lora 参数），提交偏好对并轮询至完成；adapter 写入 `output_dir`。
- **ORPO** ✅ — 胜率比偏好优化，经 `/admin/api/fine-tune/orpo/jobs`（[fusion-mlx#399](https://github.com/dahai80/fusion-mlx/issues/399)）。与 DPO 同形，但用 `lambda_odrs`（映射自 `RLSLConfig.beta`）替代 DPO 温度；无参考模型。
- **奖励模型** ✅ — Bradley-Terry value-head 训练，经 fusion-mlx `/admin/api/fine-tune/reward/jobs`（落地于 [fusion-mlx#424](https://github.com/dahai80/fusion-mlx/issues/424)）。`RewardTrainer.train` 构建 `RewardConfig`（iters 来自 `reward_num_epochs`，lora 参数来自 SFT 配置），提交偏好对并轮询至完成；adapter 以 `reward_model=true` 写入。`RewardTrainer.predict`（打分）委托给 reward/score 端点（[fusion-mlx#431](https://github.com/dahai80/fusion-mlx/issues/431)），加载训练好的 RM adapter 并返回标量奖励。
- **RLSL** ✅ — 完整闭环流水线：Phase 1 训练奖励模型 ✅（#424）→ Phase 2 运行**真实策略梯度 GRPO** ✅（#363），其奖励信号经 reward/score 端点（#431）来自该 RM。`_run_rlsl` 托管一个本地 RM 奖励服务器（`rm_reward_server.py`），GRPO 回调它给补全打分；闭环为 `RM 训练 (#424) → reward/score (#431) → GRPO (#363, PPO 裁剪策略梯度)`。
- **RLSL 预设** —— `RLSL_PRESETS`（`fusion_trainer.rlsl.presets`）提供命名 GRPO 超参束，含 `memory_estimate_gb` 供 M5 96GB 规划：`rlsl_grpo_7b`（~22 GB，`rft` 后端）、`rlsl_grpo_14b`（~62 GB，`rft` 后端）、`rlsl_grpo_14b_true`（~62 GB，`grpo` 后端，需 fusion-mlx #363 Phase 2）。DPO/ORPO 预设有意省略（超参映射自 SFT 配置）。CLI `--preset` 当前仅 SFT 支持；RLSL 预设供编程式使用与作为文档化参考配置。
- 专为 M5 96GB 统一内存设计（14B–32B 模型）

> **RLSL 闭环：** 按项目规则，fusion-trainer **只**通过 HTTP 对接 fusion-mlx，绝不直接 import `mlx-lm`。完整闭环现已真实落地 —— Phase 1 奖励模型训练经 [fusion-mlx#424](https://github.com/dahai80/fusion-mlx/issues/424)，reward/score 端点经 [#431](https://github.com/dahai80/fusion-mlx/issues/431)，Phase 2 PPO 裁剪策略梯度 GRPO 经 [#363](https://github.com/dahai80/fusion-mlx/issues/363)。`_run_rlsl` 托管一个本地 RM 奖励服务器供 GRPO 回调打分，闭环无任何假 "completed" 结果。DPO、ORPO、GRPO（独立）与奖励模型训练也全部可用。

`reward_code` 在 macOS 上通过内置 seatbelt 沙箱（`sandbox-exec`）运行生成的代码：禁用网络、数据读取限定在解释器前缀 + 标准库 + tmpfs 工作区（元数据读取保持宽松以解析路径 —— 只泄露目录结构，不泄露文件内容）、超时即杀。读取 `~/.ssh`、`/etc` 及兄弟项目源码将被拒绝。用 `FUSION_CODE_SANDBOX_BACKEND` 覆盖后端（默认 `auto` | `seatbelt` | `rlimit`）；非 macOS 回退到 rlimit 并大声告警。无论何种后端，`FUSION_CODE_SANDBOX_TRUSTED=1` 显式开关仍然生效。详见 `fusion_trainer/rlsl/isolation.py`。

### 实验追踪
- 指标日志记录，支持 TensorBoard —— **默认开启**，每个指标/步通过 `add_scalar` 流式写入 `output_dir/tensorboard`（设 `enable_tensorboard=False` 关闭）；JSONL 流式始终并行写入
- Loss 曲线可视化（matplotlib）
- 实验元数据与可复现追踪
- Git commit 绑定，全程可追溯

### TTS 音色微调（issue #74）
- **音色包训练** ✅ —— `ft voice-train` 基于成对的 `(音频, 文本)` 样本训练 Kokoro 音色包，全部委托 fusion-mlx `POST /admin/api/fine-tune/voice/jobs`（[fusion-mlx#1336](https://github.com/dahai80/fusion-mlx/issues/1336)）。上游为无梯度 SPSA 优化，trainer 只负责提交任务、轮询到终态，并返回上游指标（`improved`、`initial/final_spectral_distance`）与 `voice_ref`。
- **提交前数据集校验** —— 支持 `metadata.csv`（`file|text`）、`metadata.jsonl`（`{audio,text}`）、或裸 `<name>.wav` + `<name>.txt`；本地直接拒绝缺失/截断音频与空转写，避免坏数据集白白消耗任务。
- **推理** —— 训练产物落在 `~/.fusion-mlx/voices/<voice_id>/`，由 fusion-mlx 在 `POST /v1/audio/speech` 上以 `voice_ref` 提供（未知 ref 返回 404，无静默回退）。
- **Hub 注册** —— `--publish` 以 `model_type=audio` 注册到 fusion-model-hub（`ModelHubClient.publish_voice_package`，409 幂等）。非阻塞：Hub 故障不影响训练结果。

> `--publish` 需要可访问的 Hub（`--hub-url` / `FUSION_MODEL_HUB_URL`）。自 fusion-gateway#316 起，gateway（`:11432`）已代理 Hub 的 `/api/v1/models`（已实测：注册 201、409 幂等、列表可经 gateway 查询），默认 `http://localhost:11432` + gateway API key 即可，无需单独的 Hub key。直连 Hub（`:11444` + `FUSION_MODEL_HUB_API_KEY`）也仍然可用。

### 评测飞轮：数据摄取 + 曲线（issue #76）
- **轨迹摄取** ✅ —— `ft datasets from-eval --export <trajectories.jsonl> --output <sft.jsonl> [--dpo-output <dpo.jsonl>]` 消费 fusion-eval 轨迹导出（[fusion-eval#6](https://github.com/dahai80/fusion-eval/issues/6)）：通过的轨迹转为 `{"messages": [...]}` SFT 样本（多轮保留，现有 `load_messages` 路径直接可训）；配合 `--all` 导出时，同一任务同时有通过+失败轨迹的样本被挖掘为 `{"prompt","chosen","rejected"}` DPO 偏好对（无通过对照的失败只大声计数，绝不伪造对）。**DPO 多臂门控（#78）：** 单臂导出（每任务仅一条轨迹 —— 所有 `fusion-eval run --arm <单个>` 的导出都是这个形状）结构上不可能产出偏好对，此时 `--dpo-output` 会带解释大声报错；需用多臂评测（`fusion-eval run --arm <a> --arm <b>`）重跑，或传 `--dpo-anyway` 强制写出空对文件。清单记录 `arms` 与 `dpo_structurally_empty`，下游可据此看出对文件为空的原因。
- **来源可追溯** ✅ —— 每个 from-eval 数据集在旁边原子写入清单 `<output>.from_eval.json`（含导出 sha256）。`ft sft` 自动识别并在运行结果里记录 `source_export_sha256` / `source_export_path` —— 每个由评测数据喂养的训练运行都可追溯到其评测来源。
- **轮次注册表 + 曲线** ✅ —— `ft flywheel record --label <名称> --summary <fusion-eval summary.json> --checkpoint <adapter 目录或底座模型 id> [--export <jsonl>] [--dataset <路径>]` 追加一轮记录（通过率、分类型通过率、导出哈希、评测 run id）到 `{cache_path}/flywheel.json`（原子写）。`ft flywheel status` 打印通过率-轮次曲线（含增量）、分类型通过率表与趋势线 —— 即验收证据：每轮均可由导出哈希追溯。
- **轮次串联** ✅ —— `ft sft --init-adapter-path <adapter 目录>` 从上一轮的适配器目录初始化新 LoRA（最终权重或最新编号 checkpoint，走 #65 resume 路径暂存；步数计数从 0 开始）。用于飞轮训练轮次串联：第 N+1 轮在第 N 轮权重之上、以第 N 轮评测数据继续训练。
- **真实轮次证据** ✅ —— Qwen3.8-27B-8bit、30 任务集：baseline 53.33% → fw-round1 73.33%（+20pp）→ fw-round2 63.33%（−10pp，testwrite 类型回归 3/3→0/3，全部 900s 超时 —— 链式过拟合信号）；`long-context` 与 `multi-turn-repair` 两个基线零分类型转非零并保持。完整曲线：`ft flywheel status`。
- **新鲜数据纪律 + 评测门控晋级** ✅（issue #82）—— `ft flywheel record` 追踪每个训练轮消费的导出 sha；在已被消费的导出上再记一轮会大声报错（exit 1），除非传 `--allow-reconsume`（条目标记 `reconsumed`，绝不静默）。`--baseline` 标记种子轮（其导出从未被训练过）。`--gate` 开启评测门控：通过率相对现任（此前未被废弃轮次中的最优）回退的轮记为 `discarded` —— 其 checkpoint 不晋级 —— `ft flywheel status` 渲染仅含晋级轮的趋势线（废弃轮排除）及废弃轮汇总。`ft flywheel mark --label X --status promoted|discarded` 为门控存在前的历史轮回填状态。**实弹门控第 4 轮（2026-10-31）：** 全新数据（第 3 轮评测导出 `504298d3…`）从现任最优 fw-round1 适配器链式训练，得 43.33%（13/30，run `20261031-r4-eval2`）—— 相对现任 −30pp 回退，门控判定 **discarded**，checkpoint 未晋级。**据此重划飞轮声明：** 飞轮的实测价值在于第一轮高信号失败数据训练（+20pp，fw-round1）；重复 SFT 串联每轮必退化（第 2/3/4 轮全部废弃 —— 累积过度生产，数据新鲜度或更优起点都无法修复）。无新增失败数据源或换用其他对齐方法（如多臂导出上的 DPO）时，不要在首个晋级轮之后继续串联。
- **多臂 DPO 轮** ✅（issue #83）—— #82 重划后的"换一种对齐方法"，已实弹执行：合并既有 4 份 30 任务评测导出（120 条轨迹）为一份多臂 JSONL，挖出 22 对偏好样本（`chosen` 血统 19/22 来自现任 fw-round1 —— 真·同策略偏好数据；30 条无配对失败大声计数；6 个从未通过任务排除），在现任 fw-round1 之上链式 DPO 训练（`ft rlsl --method dpo --init-adapter-path`，beta 0.1、lr 5e-6、seq 1024），`ft flywheel record --gate` 门控记录。**让 DPO 在 27B-8bit 上跑起来共需三处上游修复**（fusion-mlx #1461 链式适配器双重挂载、#1463 混合线性注意力反传缺 `model.train()`、#1472 逐对参考模型 load/evict 的 Metal 分配churn —— 最后一项连同本仓 `RLSLConfig.keep_ref_resident` 透传落地），外加 `RLSLConfig.gradient_checkpointing` 透传（上游 #1460）：DPO 损失图同时持有 chosen+rejected 两条前向（激活内存 ≈ SFT 2×），逐对参考加载每对两次全量 27B 装载直至引擎 memory_enforcer 判 165.9GB > 120GB 天花板 fatal 退出。**第 5 轮结果（2026-11-01，run `20261101-r5-eval`，钉定 30 任务集）：17/30 = 56.67% —— 门控判定 DISCARDED（相对现任 fw-round1 73.33% 回退 −16.67pp），checkpoint 未晋级。** 诚实负结果：在现任自身血统上挖出的 22 对偏好样本做 DPO 未能巩固其收益 —— `testwrite` 回退 2/3→0/3（代理未产出可运行测试；隐藏变异全部存活），而 `configfix`/`depfix`/`bugfix` 保持 3/3，`feature` 依旧 0/3（持续性规格缺口）。失败模式为规格缺口/实现落空（缺模块、函数返回 None、零行编辑），仅 1 例 900s 超时（`t27-dag-filter-size`）。曲线：baseline 53.33% → fw-round1 73.33%（现任）→ fw-round2 63.33% → fw-round3 46.67% → fw-round4 43.33%（废弃）→ fw-round5 56.67%（废弃）。多臂 DPO 排位高于所有被废弃的 SFT 串联轮，且比分类型强度保持得更好，但 4 份导出挖出的 22 对偏好信号太薄，赢不了现任 —— 在更大规模多臂导出语料出现前，飞轮仍是"一轮高信号"工具。

---

## 硬件参考

| 训练方法 | 模型规模 | 峰值内存 | 推荐度 |
|---|---|---|---|
| 全量微调 | 7B | ~32 GB | M5 96GB ✓ |
| LoRA | 7B | ~16 GB | M5 96GB ✓ |
| QLoRA (4-bit) ✅ | 14B | ~28 GB | M5 96GB ✓ |
| QLoRA (4-bit) ✅ | 32B | ~48 GB | M5 96GB ✓ |
| 奖励模型训练 | 14B | ~22 GB | M5 96GB ✓ |
| RLSL (PPO-light) | 14B | ~62 GB | M5 96GB ✓ |

以上估算基于 8-bit 量化基座 LoRA（MXFP8 路径）。QLoRA 已落地（[#402](https://github.com/dahai80/fusion-mlx/issues/402)）；RM 训练已落地（[#424](https://github.com/dahai80/fusion-mlx/issues/424)）；RLSL 闭环（Phase 2 GRPO）已落地（[#363](https://github.com/dahai80/fusion-mlx/issues/363)，经 [#431](https://github.com/dahai80/fusion-mlx/issues/431) 打分）。MXFP8 现路由到 8-bit QLoRA（[fusion-mlx#402](https://github.com/dahai80/fusion-mlx/issues/402)/#425 已关闭）：`use_mxfp8=True` 透传并将冻结基座量化到 8-bit + LoRA。`use_mxfp8=False`（默认）时以默认精度训练，实际峰值内存可能高于表中数值。

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
| `fusion-trainer sft` | 运行监督微调（`--publish` 完成后发布适配器到 Hub；`--resume` 从最新 checkpoint 续训中断的 run，#65；`--eval-select` 训练后经 fusion-bench 逐 checkpoint 评测并择优，#75；`--pack-sequences` 短样本序列打包，fusion-mlx#1400） |
| `fusion-trainer rlsl` | 运行 DPO/ORPO/RLSL/GRPO 对齐 |
| `fusion-trainer publish <run_id>` | 发布已完成 run 的 LoRA 适配器到 Hub |
| `fusion-trainer bnup-prep --stage sft\|dpo` | BNUP 数据准备（SFT 校验 / DPO 负样本生成，#56/#57） |
| `fusion-trainer bnup-train --stage sft\|dpo\|pipeline` | BNUP 两阶段训练（SFT → DPO → 合并，#59；`--resume` 续训中断的 SFT 阶段，#65） |
| `fusion-trainer merge --base <模型> --adapter <路径>` | 合并适配器到基座模型（#58） |
| `fusion-trainer voice-train --model <kokoro> --dataset <目录>` | 训练 TTS 音色包（`--publish` 完成后注册到 Hub，#74） |
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

自 #80 起，服务默认值统一经 `fusion_core.config`（v0.7.0+，钉子从 v0.5.0 上调）单点解析，环境变量名不变。`MLXFineTuneClient` 引擎令牌经 `resolve_api_key(env_var="FUSION_MLX_ADMIN_TOKEN")` 解析 —— 先环境变量，再回退 `~/.fusion-mlx/settings.json` 的 `auth.api_key`（与 `scripts/trainer-bridge.sh` 一直在做的回退一致，现在进了进程内）—— 已有 fusion-mlx settings 文件的机器上，CLI 作业无需导出令牌即可通过鉴权。

`FUSION_TRAINER_PUBLISH=1` 环境变量覆盖可无条件强制发布。**GUI 路径**现已原生转发发布字段（[fusion-agent-studio#277](https://github.com/dahai80/fusion-agent-studio/issues/277) + [fusion-studio#390](https://github.com/dahai80/fusion-studio/issues/390) 已关闭）：fusion-agent-studio `TrainingService._build_trainer_config` 透传 `publish_adapter`/`hub_url`/`hub_api_key`，训练面板也暴露了这些字段，GUI 触发的 run 可原生发布。环境变量覆盖保留为兜底。CLI `--publish` 路径不受影响。

#### 审计日志与 api-key 网关（企业级）

`RunManager` 在 `{cache_path}/audit.jsonl` 维护 **只追加 JSONL 审计日志** —— 每个写操作（`start_sft`、`start_rlsl`、`stop_run`、`publish_run`、`delete_adapter`）记录一行，格式 `{timestamp, actor, action, run_id, detail, ok}`。actor 来自 `FUSION_TRAINER_ACTOR`（默认 `local`）。文件只追加，不重写历史；它是问责追踪，不是防篡改的不可否认日志（本地优先文件不做签名，超出范围）。

可选 **api-key 网关**（设置 `FUSION_TRAINER_API_KEY`）会在密钥不匹配时以 `PermissionDeniedError` 拒绝写操作（常数时间比较）。当该环境变量 **未设置** 时为本地信任模式，允许全部操作 —— 适合单用户 Mac。共享/多操作者部署请设置它。`info()` 会报告网关是否启用及审计日志路径。

| 环境变量 | 用途 | 默认值 |
|---|---|---|
| `FUSION_TRAINER_ACTOR` | 每条审计记录的 actor | `local` |
| `FUSION_TRAINER_API_KEY` | 网关写操作；未设置则全部放行 | 未设置（本地信任） |

`start_sft`/`start_rlsl`/`stop_run`/`delete_adapter` 接受可选 `api_key` 参数；`publish_run` 接受 `trainer_api_key`（与 hub 的 `api_key` 区分开，使部署可在不泄露 hub 密钥的情况下网关操作）。

#### GRPO 后端诚实声明（`grpo_backend`，issue #75）

`RLSLConfig.grpo_backend` 选择 GRPO 执行路径：

| 后端 | 行为 | 端点 |
|---|---|---|
| `rft`（默认） | **奖励过滤的 SFT 近似**（RFT/RAFT 风格）：在线采样 → sandbox 打分 → top-k → SFT/LoRA 作业。没有逐 token 策略梯度、没有重要性比率 —— 它朝更高奖励输出收敛，但**不是真正的 GRPO**。 | `/admin/api/fine-tune/jobs` |
| `grpo` / `policy_gradient` | 真正的策略梯度 GRPO，在 fusion-mlx 服务端运行；trainer 本地起一个奖励 HTTP 回调端点供 fusion-mlx 调用（`reward_server.py` → `sandbox.reward_completion`） | `/api/fine-tune/grpo/jobs` |

`policy_gradient` 是 `grpo` 的别名（在 `TrainerConfig.__post_init__` 归一化，issue #75）—— 想要真策略梯度时的诚实命名。`grpo` 需要 fusion-mlx #363 Phase 2。YAML 中设置（`rlsl.grpo_backend: grpo`）。按项目规则，trainer **只**通过 fusion-mlx HTTP API 通信 —— 不直接 import `mlx-lm`。

#### 基于 fusion-bench 的 checkpoint 择优（`--eval-select`，issue #75）

最低 val loss 是"最优 checkpoint"的弱代理。`ft sft --eval-select` 补全训练闭环：训练结束后，每个编号 checkpoint（`NNNNNN_adapters.safetensors`）经 fusion-bench 任务 API 跑一次评测，按 bench 分数择优：

```bash
ft sft --model Qwen3-0.6B-4bit --dataset data.jsonl --save-every 100 \
    --eval-select --bench-url http://localhost:11467 \
    --bench-api-key <jwt> --bench-tenant-id default
```

- 确定性选择：bench 分数（高者胜，平分取更晚 step）→ val_loss（低者胜，平分取更晚 step）→ 最后一个 checkpoint。
- 完整选择表 —— checkpoint、step、val_loss、bench 任务 id、bench 分数、chosen —— 原子写入适配器旁的 `checkpoint_selection.json`，并附到 run 结果。
- Fail-visible、绝不致命：bench 不可用（或未配置）时每行标记 `unavailable`、记录 `fallback_reason`、按 val_loss 择优。选择永远不会静默出错。
- bench 打分针对真实 checkpoint 权重：fusion-bench v0.4.11（[fusion-bench#29](https://github.com/dahai80/fusion-bench/issues/29)，已关闭）把每个候选的 `adapter_path` 作为 `adapters` 字段转发给 fusion-mlx chat。需 fusion-bench >= 0.4.11；对更旧版本，分数反映基座模型，此时应以 val_loss 为准。

---

#### 指标导出（企业级）

`ExperimentLogger.export_prometheus(path, run_id=, extra_series=)` 与模块级 `export_prometheus_snapshot(path, metrics, run_id=, extra_series=)` 写入 **Prometheus 文本格式**（`.prom`）快照 —— 每个标量指标键一个 gauge：

```
# HELP fusion_trainer_train_loss run-1
# TYPE fusion_trainer_train_loss gauge
fusion_trainer_train_loss{run="run-1"} 0.1234
```

`RunManager` 在 run 结束时写 `{cache_path}/runs/<run_id>/metrics.prom`，包含 `run_steps_total`、`run_total_steps`、`run_status_code`（`0=running,1=completed,2=failed,3=cancelled`）。指标名按 Prometheus `[a-zA-Z_:][a-zA-Z0-9_:]*` 语法净化；非标量值跳过。这是**文件导出** —— 由 node-exporter `textfile_collector` 或 sidecar 投递。实时抓取端点上游在 fusion-agent-studio（不在 fusion-trainer，它是纯计算层，无 HTTP 服务器）。

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

## 每日真实训练 CI（issue #75）

每次 push 的 CI 在 ubuntu 上跑 mock 套件 —— MLX 没有 Linux wheel，因此所有需要实时 fusion-mlx 引擎的测试都由环境变量开关（`FUSION_TRAINER_REAL=1`）控制并在那里静默跳过。`.github/workflows/nightly.yml` 将其真正放开：在 macOS arm64 runner 上启动一个加载 `Qwen3-0.6B-4bit` 的真实 fusion-mlx，对真实梯度循环跑训练套件（`tests/test_sft_trainer.py::TestRealFusionMLXIntegration`）与契约套件（`tests/test_contract_integration.py`，marker `integration`）。按计划触发（北京时间 02:30），也可 `workflow_dispatch` 手动触发：

```bash
gh workflow run nightly -R dahai80/fusion-trainer
```

上游契约漂移（字段改名、作业生命周期变更）一天内即可捕获，而不是等下一次人工验证。

---

## API 稳定性（v1.0）

自 **1.0.0** 起，公开 API 表面遵循 [语义化版本](https://semver.org/lang/zh-CN/) 冻结。冻结表面 —— CLI 子命令、`TrainerConfig` 字段、`RunManager` 方法、`MLXFineTuneClient` 方法及公开异常类型 —— 详见 [`docs/API_STABILITY.md`](docs/API_STABILITY.md)。对该表面的破坏性变更需主版本号升级；新增为次版本；修复为修订号。内部模块（`_` 前缀、`presets.py`、`sandbox.py` 奖励内部、`experiment/` 日志内部）未冻结，任意版本均可变更。

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

### 训练控制台（trainer-bridge，#71/#72）

自带四节式 Web 控制台 —— 数据集 / 训练 / 合并发布 / 基座对比评测 —— 由 trainer-bridge（`bridge/`，FastAPI 薄门面）提供。浏览器永远接触不到 fusion-mlx 管理密钥；训练运行为 bridge 进程内的后台线程，刷新页面不影响训练。

```bash
export FUSION_TRAINER_BRIDGE_KEY=<token>     # 浏览器直连 bridge 的令牌
./scripts/trainer-bridge.sh start            # 监听 http://127.0.0.1:8790/
./scripts/trainer-bridge.sh stop
```

**Shell 模块注册（#72）：** bridge 启动时向 fusion-gateway shell 注册表登记 `bridge/manifest.json` 并每 15 秒心跳，fusion-web-shell 侧边栏由此出现 **训练控制台** 一级入口，网关代理 `http://127.0.0.1:11432/shell/fusion-trainer/` 成为唯一用户入口（bridge 停止 → 侧边栏条目置灰 unreachable）。注册走 K1 网关签发密钥：注册体不带 `api_key`，网关签发 `fm-` 模块密钥（仅返回一次），bridge 落盘到 `~/.fusion-trainer-bridge/web_module_key`（0600），心跳与代理请求均以它鉴权（入站同时接受静态 bridge 令牌供环回直连）。由 `FUSION_SHELL_GATEWAY_URL` + `FUSION_ADMIN_KEY` 环境变量开启；`trainer-bridge.sh` 默认指向本机网关并自动从网关配置读取管理密钥。

---

## 发布与变更管理

发布遵循[语义化版本](https://semver.org/lang/zh-CN/)，冻结 API 表面见 [`docs/API_STABILITY.md`](docs/API_STABILITY.md)。完整发布流程 —— 分支策略（`release/vX.Y.Z`）、版本号规则、切版步骤、评审清单、热修复流程、回滚策略 —— 详见 [`docs/RELEASE_PROCESS.md`](docs/RELEASE_PROCESS.md)。`CHANGELOG.md` 存放各版本发布说明；git 标签 `vX.Y.Z` 为不可变发布点。

---

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 许可证。

## 致谢

基于 [fusion-mlx](https://github.com/fusion-mlx/fusion-mlx) 和 [Apple MLX](https://github.com/ml-explore/mlx) 构建。