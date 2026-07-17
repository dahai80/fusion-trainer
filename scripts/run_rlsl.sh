#!/usr/bin/env bash
# ============================================================================
# Fusion-Trainer: RLSL 强化监督学习流水线启动脚本
# ============================================================================
# 完整链路: 导入偏好数据集 → RM训练 → DPO/ORPO/RLSL对齐 → 导出权重
#
# 用法:
#   ./scripts/run_rlsl.sh --model qwen2.5-7b-4bit \\
#                         --dataset /path/to/preference.jsonl \\
#                         --method dpo
#
# 方法选项:
#   dpo   直接偏好优化 (推荐，无需RM)
#   orpo  单阶段SFT+偏好对齐
#   rlsl  完整RLSL流水线 (需奖励模型)
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
METHOD="dpo"
REWARD_MODEL=""
RUN_NAME=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config|-c)     CONFIG="$2"; shift 2 ;;
        --model|-m)      MODEL="$2"; shift 2 ;;
        --dataset|-d)    DATASET="$2"; shift 2 ;;
        --method|-M)     METHOD="$2"; shift 2 ;;
        --reward-model|-r) REWARD_MODEL="$2"; shift 2 ;;
        --run-name|-n)   RUN_NAME="$2"; shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ---- 参数校验 ----
if [ "$METHOD" = "rlsl" ] && [ -z "$REWARD_MODEL" ]; then
    echo "❌ RLSL 方法需要指定奖励模型 (--reward-model)"
    echo "   用法: ./scripts/run_rlsl.sh --method rlsl --reward-model <path> [其他参数]"
    exit 1
fi

# ---- 构建命令 ----
CMD="fusion-trainer rlsl"

if [ -n "$CONFIG" ]; then
    CMD="$CMD --config $CONFIG"
fi
if [ -n "$MODEL" ]; then
    CMD="$CMD --model $MODEL"
fi
if [ -n "$DATASET" ]; then
    CMD="$CMD --dataset $DATASET"
fi
if [ -n "$METHOD" ]; then
    CMD="$CMD --method $METHOD"
fi
if [ -n "$REWARD_MODEL" ]; then
    CMD="$CMD --reward-model $REWARD_MODEL"
fi
if [ -n "$RUN_NAME" ]; then
    CMD="$CMD --run-name $RUN_NAME"
fi

# 附加参数
for arg in "${EXTRA_ARGS[@]}"; do
    CMD="$CMD $arg"
done

# ---- 显示配置 ----
echo "╔══ Fusion-Trainer RLSL 流水线 ═══════════════════════════╗"
echo "║  方法:   $METHOD"
echo "║  模型:   ${MODEL:-<从配置读取>}"
echo "║  数据集: ${DATASET:-<从配置读取>}"
echo "║  命令:   $CMD"
echo "╚═══════════════════════════════════════════════════════════╝"

# ---- 执行 ----
echo ""
eval "$CMD"
echo ""
echo "✓ RLSL 对齐完成"