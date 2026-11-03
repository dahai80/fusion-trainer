#!/usr/bin/env bash
# trainer-bridge lifecycle: start | stop | status | log [-f]
#
# The bridge is the Trainer console's backend (bridge/trainer.html served
# at /). It owns the fusion-mlx admin key and the RunManager lifecycle —
# runs are background threads in the bridge process, so the bridge must
# stay up for the duration of a training.
#
# Required env: FUSION_TRAINER_BRIDGE_KEY  (bearer token browser -> bridge)
# Optional env: FUSION_MLX_ADMIN_TOKEN, FUSION_TRAINER_MLX_PORT (11434),
#               FUSION_TRAINER_BRIDGE_PORT (8790), FUSION_MLX_MODEL,
#               FUSION_SHELL_GATEWAY_URL + FUSION_ADMIN_KEY (shell
#               registration, issue #72 — admin key auto-read from the
#               gateway config when unset)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FUSION_TRAINER_BRIDGE_VENV:-$(dirname "$ROOT")/.venv}"
PY="$VENV/bin/python"
PORT="${FUSION_TRAINER_BRIDGE_PORT:-8790}"
RUN_DIR="$ROOT/runtime_cache/bridge"
PID_FILE="$RUN_DIR/bridge.pid"
LOG_FILE="$RUN_DIR/bridge.log"

mkdir -p "$RUN_DIR"

alive() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "${1:-status}" in
    start)
        if alive; then
            echo "bridge already running (pid $(cat "$PID_FILE"), port $PORT)"
            exit 0
        fi
        if [ -z "${FUSION_TRAINER_BRIDGE_KEY:-}" ]; then
            # Fallback: persisted key file (0600), same read-only pattern
            # as the settings.json reads below. Generate with:
            #   python3 -c "import secrets;print(secrets.token_urlsafe(32))" \
            #     > ~/.fusion-trainer-bridge/bridge_key
            FUSION_TRAINER_BRIDGE_KEY="$(cat "$HOME/.fusion-trainer-bridge/bridge_key" 2>/dev/null || true)"
        fi
        if [ -z "${FUSION_TRAINER_BRIDGE_KEY:-}" ]; then
            echo "ERROR: FUSION_TRAINER_BRIDGE_KEY is required (env or ~/.fusion-trainer-bridge/bridge_key)" >&2
            exit 1
        fi
        export FUSION_TRAINER_BRIDGE_KEY
        if [ ! -x "$PY" ]; then
            echo "ERROR: venv python not found: $PY" >&2
            exit 1
        fi
        # Engine key: same source the f-ecommerce bridge used —
        # ~/.fusion-mlx/settings.json auth.api_key (never hardcode it here).
        if [ -z "${FUSION_MLX_ADMIN_TOKEN:-}" ]; then
            FUSION_MLX_ADMIN_TOKEN="$("$PY" -c \
                "import json;print(json.load(open('$HOME/.fusion-mlx/settings.json'))['auth']['api_key'])" \
                2>/dev/null || true)"
        fi
        if [ -z "$FUSION_MLX_ADMIN_TOKEN" ]; then
            echo "ERROR: FUSION_MLX_ADMIN_TOKEN unset and settings.json has no api_key" >&2
            exit 1
        fi
        export FUSION_MLX_ADMIN_TOKEN
        # Shell module registration (issue #72): default to the local
        # gateway; admin key from env or the gateway's own config (read-only,
        # same pattern as the fusion-mlx settings.json read above). Without
        # both, the console stays a standalone page — warn, never fail.
        export FUSION_SHELL_GATEWAY_URL="${FUSION_SHELL_GATEWAY_URL:-http://127.0.0.1:11432}"
        # #80: fusion_core.config.admin_key reads FUSION_ADMIN_KEY; fall
        # back to the gateway's own config (read-only, same pattern as the
        # fusion-mlx settings.json read above).
        if [ -z "${FUSION_ADMIN_KEY:-}" ]; then
            GW_CONFIG="${FUSION_GATEWAY_CONFIG:-$HOME/fusion/fusion-gateway/config.yaml}"
            FUSION_ADMIN_KEY="$("$PY" -c \
                "import yaml;print(yaml.safe_load(open('$GW_CONFIG'))['web_shell']['admin_key'])" \
                2>/dev/null || true)"
        fi
        if [ -n "${FUSION_ADMIN_KEY:-}" ]; then
            export FUSION_ADMIN_KEY
            echo "shell registration: gateway=${FUSION_SHELL_GATEWAY_URL}"
        else
            echo "WARN: FUSION_ADMIN_KEY unset — console will NOT appear in the shell sidebar" >&2
        fi
        cd "$ROOT"
        nohup "$PY" -m bridge.server >>"$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        sleep 1
        if alive; then
            echo "bridge started (pid $(cat "$PID_FILE"), port $PORT, log $LOG_FILE)"
        else
            echo "bridge failed to start — see $LOG_FILE" >&2
            tail -5 "$LOG_FILE" >&2
            exit 1
        fi
        ;;
    stop)
        if alive; then
            kill "$(cat "$PID_FILE")"
            rm -f "$PID_FILE"
            echo "bridge stopped"
        else
            rm -f "$PID_FILE"
            echo "bridge not running"
        fi
        ;;
    status)
        if alive; then
            echo "bridge running (pid $(cat "$PID_FILE"), port $PORT)"
            curl -fsS "http://127.0.0.1:$PORT/health" | "$VENV/bin/python" -m json.tool \
                2>/dev/null || true
        else
            echo "bridge not running"
            exit 1
        fi
        ;;
    log)
        tail "${2:--n}20" "$LOG_FILE"
        ;;
    *)
        echo "usage: $0 {start|stop|status|log [-f|-n N]}" >&2
        exit 2
        ;;
esac
