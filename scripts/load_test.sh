#!/usr/bin/env bash
# ============================================================================
# Fusion-Trainer: load / stability verification (enterprise #4)
# ============================================================================
# Runs N consecutive training rounds against a live fusion-mlx, doctor-checks
# the server between each, and reports any run that did not reach "completed".
# Exits non-zero on the first failed round so a CI && chain catches it.
#
# Prereqs:
#   - fusion-mlx running (~/claude-home/fusion-mlx/start.sh start)
#   - FUSION_MLX_ADMIN_TOKEN set (settings.json auth.api_key)
#   - a small model loaded (default: mlx-community-Llama-3.2-1B-Instruct-4bit)
#   - a tiny SFT dataset (the script makes a 4-line one in a temp file)
#
# Usage:
#   FUSION_MLX_ADMIN_TOKEN=fg-admin-key ./scripts/load_test.sh
#   ROUNDS=10 MODEL=mlx-community-Llama-3.2-1B-Instruct-4bit ./scripts/load_test.sh
#
# Environment:
#   ROUNDS   number of consecutive rounds (default 5)
#   MODEL    base model id (default mlx-community-Llama-3.2-1B-Instruct-4bit)
#   PORT     fusion-mlx port (default 11434, or $FUSION_MLX_PORT)
# ============================================================================
set -euo pipefail

ROUNDS="${ROUNDS:-5}"
MODEL="${MODEL:-mlx-community-Llama-3.2-1B-Instruct-4bit}"
PORT="${FUSION_MLX_PORT:-11434}"
MLX_START="${MLX_START:-$HOME/claude-home/fusion-mlx/start.sh}"

if [ -z "${FUSION_MLX_ADMIN_TOKEN:-}" ]; then
    echo "ERROR: FUSION_MLX_ADMIN_TOKEN not set (settings.json auth.api_key)" >&2
    exit 2
fi

log() { printf '[load_test %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Tiny 4-line SFT dataset — input/output schema (what the loader expects).
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT
DATA="$TMPDIR_WORK/sft_tiny.jsonl"
cat > "$DATA" <<'JSONL'
{"input":"What is 2+2?","output":"4"}
{"input":"What is 3+3?","output":"6"}
{"input":"What is 4+4?","output":"8"}
{"input":"What is 5+5?","output":"10"}
JSONL
log "wrote tiny dataset -> $DATA"

doctor() {
    if [ -x "$MLX_START" ]; then
        log "doctor: $MLX_START doctor"
        "$MLX_START" doctor || log "WARN: doctor reported issues (continuing)"
    fi
}

log "starting load test: ROUNDS=$ROUNDS MODEL=$MODEL PORT=$PORT"
doctor

passed=0
failed=0
for i in $(seq 1 "$ROUNDS"); do
    log "=== round $i/$ROUNDS ==="
    RUN_NAME="load_${i}"
    if fusion-trainer sft \
        --model "$MODEL" \
        --dataset "$DATA" \
        --run-name "$RUN_NAME" \
        --mlx-port "$PORT" \
        --iters 1 \
        --batch-size 1 \
        --lora-rank 4 >/dev/null 2>&1; then
        log "round $i: SFT exit 0"
        passed=$((passed + 1))
    else
        log "round $i: SFT FAILED (exit $?)"
        failed=$((failed + 1))
        exit 1
    fi
    doctor
done

log "DONE: passed=$passed failed=$failed rounds=$ROUNDS"
[ "$failed" -eq 0 ]
