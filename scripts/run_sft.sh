#!/usr/bin/env bash
# ============================================================================
# Fusion-Trainer: SFT 监督学习启动脚本
# ============================================================================
# 用法:
#   ./scripts/run_sft.sh --model qwen2.5-7b-4bit --dataset /path/to/data.jsonl
#   ./scripts/run_sft.sh --config configs/sft_7b.yaml
#   ./scripts/run_sft.sh --preset sft_7b_lora
#
# 预设:
#   sft_7b_lora      7B LoRA 微调
#   sft_7b_full      7B 全参数微调
#   sft_14b_qlora    14B QLoRA 微调
#   sft_32b_qlora    32B QLoRA 微调
#   sft_code_7b_lora 7B 代码模型 LoRA 微调（长上下文）
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 激活虚拟环境（如果存在）
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

# ---- 解析参数 ----
CONFIG=""
MODEL=""
DATASET=""
PRESET=""
OUTPUT=""
METHOD=""
LR=""
EPOCHS=""
BATCH_SIZE=""
RUN_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)    CONFIG="$2"; shift 2 ;;
        --model|-m)     MODEL="$2"; shift 2 ;;
        --dataset|-d)   DATASET="$2"; shift 2 ;;
        --preset|-p)    PRESET="$2"; shift 2 ;;
        --output|-o)    OUTPUT="$2"; shift 2 ;;
        --method|-M)    METHOD="$2"; shift 2 ;;
        --lr|-l)        LR="$2"; shift 2 ;;
        --epochs|-e)    EPOCHS="$2"; shift 2 ;;
        --batch-size|-b) BATCH_SIZE="$2"; shift 2 ;;
        --run-name|-n)  RUN_NAME="$2"; shift 2 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- 构建命令 ----
CMD="fusion-trainer sft"

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi
if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi
if [ -n "$PRESET" ]; then
    CMD="$CMD --preset $PRESET"
fi
if [ -n "$OUTPUT" ]; then
    CMD="$CMD --output $OUTPUT"
fi
if [ -n "$METHOD" ]; then
    CMD="$CMD --method $METHOD"
fi
if [ -n "$LR" ]; then
    CMD="$CMD --learning-rate $LR"
fi
if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi
if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi
if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

# 附加参数
for arg in "${EXTRA_ARGS[@]}"; do
    CMD="$CMD $arg"
done

# ---- 显示配置 ----
echo "╔══ Fusion-Trainer SFT 启动 ═══════════════════════════════╗"
echo "║  命令: $CMD"
echo "╚═══════════════════════════════════════════════════════════╝"

# ---- 执行 ----
echo ""
eval "$CMD"
echo ""
echo "✓ SFT 训练完成"