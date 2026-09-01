#!/usr/bin/env bash
# ============================================================================
# Fusion-Trainer: 一键环境部署脚本 (国内镜像版)
# ============================================================================
# 用法:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh              # 创建 venv + 安装依赖
#   ./scripts/setup_env.sh --dev        # 包含 dev 依赖
#   ./scripts/setup_env.sh --no-venv    # 使用系统 Python（不推荐）
#
# 默认使用清华镜像源加速国内下载。
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "╔══ Fusion-Trainer 环境部署 ═══════════════════════════════╗"

# ---- Python 版本检查 ----
# P0-7 (round-3 audit): grep -oP is PCRE (GNU-only); macOS BSD grep does not
# support -P and fails outright. Use sed instead (portable POSIX).
PYTHON="python3"
PYTHON_VERSION=$($PYTHON --version 2>&1 | sed -E 's/Python ([0-9]+\.[0-9]+).*/\1/')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    echo "❌ 需要 Python >= 3.11，当前版本: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python $PYTHON_VERSION"

# ---- 检查 fusion-mlx 依赖 ----
# P0-7 (round-3 audit): no hardcoded developer path. fusion-mlx is optional
# at install time (trainer talks to it over HTTP at runtime), so a missing
# path is a soft warning, never a silent broken install. Resolve from env,
# fall back to a sibling directory convention, warn if neither exists.
FUSION_MLX_PATH="${FUSION_MLX_PATH:-$(dirname "$SCRIPT_DIR")/fusion-mlx}"
if [ ! -d "$FUSION_MLX_PATH" ]; then
    echo "⚠️  fusion-mlx 未在 $FUSION_MLX_PATH 找到"
    echo "   fusion-mlx 在安装阶段可选（运行时通过 HTTP 对接）。"
    echo "   如需本地安装 fusion-mlx：设置 FUSION_MLX_PATH 环境变量指向其目录，"
    echo "   或手动安装: cd /path/to/fusion-mlx && pip install -e ."
    echo "   跳过 fusion-mlx 本地安装，继续安装 Fusion-Trainer。"
    FUSION_MLX_PATH=""
else
    echo "✓ fusion-mlx 路径: $FUSION_MLX_PATH"
fi

# ---- 虚拟环境 ----
USE_VENV=true
for arg in "$@"; do
    [ "$arg" = "--no-venv" ] && USE_VENV=false
done

if $USE_VENV; then
    VENV_DIR="$SCRIPT_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        echo "→ 创建虚拟环境: $VENV_DIR"
        $PYTHON -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    PYTHON="$VENV_DIR/bin/python"
    PIP="$VENV_DIR/bin/pip"
    echo "✓ 虚拟环境: $VENV_DIR"
else
    PIP="pip3"
    echo "⚠️  使用系统 Python，不创建虚拟环境"
fi

# ---- 升级 pip ----
$PIP install --upgrade pip -q

# ---- 安装 fusion-mlx (本地路径) ----
if [ -d "$FUSION_MLX_PATH" ]; then
    echo "→ 安装 fusion-mlx (本地)..."
    $PIP install -e "$FUSION_MLX_PATH" -q
fi

# ---- 安装 Fusion-Trainer ----
echo "→ 安装 Fusion-Trainer 依赖..."
DEV_FLAG=""
for arg in "$@"; do
    [ "$arg" = "--dev" ] && DEV_FLAG="[dev]"
done

$PIP install -e ".${DEV_FLAG}" -q

echo "✓ Fusion-Trainer 安装完成"
echo ""
echo "可用命令:"
echo "  fusion-trainer info          # 查看系统信息"
echo "  fusion-trainer --help        # 查看帮助"
echo "  fusion-trainer sft --help    # SFT 训练帮助"
echo "  fusion-trainer rlsl --help   # RLSL 对齐帮助"
echo ""

# ---- 验证安装 ----
echo "→ 验证安装..."
$PYTHON -c "
from fusion_trainer import __version__
from fusion_mlx import __version__ as fm_version
print(f'  Fusion-Trainer v{__version__}')
print(f'  fusion-mlx     v{fm_version}')
try:
    import mlx.core as mx
    print(f'  MLX device: {mx.default_device()}')
    print(f'  Metal: {mx.metal.is_available()}')
except ImportError:
    print('  MLX: not available')
"
echo "✓ 验证完成"