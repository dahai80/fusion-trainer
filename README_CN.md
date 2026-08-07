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
- 32B 模型 4-bit QLoRA 微调（M5 96GB）
- 领域专属预设配置（代码、金融、科学）
- MXFP8 混合精度训练（MLX 原生）

### 强化监督学习（RLSL）
- **DPO** — 直接偏好优化（无需奖励模型）
- **ORPO** — 胜率比偏好优化（单阶段 SFT+对齐）
- **RLSL** — 完整 PPO-light 流水线（含奖励模型训练）
- 专为 M5 96GB 统一内存设计（14B–32B 模型）

### 实验追踪
- 指标日志记录，支持 TensorBoard
- Loss 曲线可视化（matplotlib）
- 实验元数据与可复现追踪
- Git commit 绑定，全程可追溯

---

## 硬件参考

| 训练方法 | 模型规模 | 峰值内存 | 推荐度 |
|---|---|---|---|
| 全量微调 | 7B | ~32 GB | M5 96GB ✓ |
| LoRA | 7B | ~16 GB | M5 96GB ✓ |
| QLoRA (4-bit) | 14B | ~28 GB | M5 96GB ✓ |
| QLoRA (4-bit) | 32B | ~48 GB | M5 96GB ✓ |
| 奖励模型训练 | 14B | ~22 GB | M5 96GB ✓ |
| RLSL (PPO-light) | 14B | ~62 GB | M5 96GB ✓ |

以上估算基于 MXFP8 混合精度。

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

### 2. 导入数据集

```bash
fusion-trainer dataset import /path/to/data.jsonl \
    --name my_dataset \
    --category sft \
    --format jsonl
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
| `fusion-trainer rlsl` | 运行 DPO/ORPO/RLSL 对齐 |
| `fusion-trainer dataset import` | 导入数据集到 Hub |
| `fusion-trainer dataset list` | 列出 Hub 中的数据集 |
| `fusion-trainer dataset info` | 查看数据集详情 |
| `fusion-trainer info` | 查看系统信息与硬件检查 |
| `fusion-trainer --help` | 查看完整帮助 |

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