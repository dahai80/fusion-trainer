# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

fusion-trainer is a **pure HTTP delegation orchestrator** for MLX fine-tuning on Apple Silicon. It owns no training loop — all training/eval/merge work is delegated to the fusion-mlx engine (`/admin/api/fine-tune/*` on `:11434`) over HTTP. The `ft` CLI (`fusion_trainer/cli.py`, also `fusion-trainer`) is the primary entry point.

- Pure Python, no Rust/Swift. Depends on the in-tree shared library `fusion-core` (pinned `>=0.9.0`; CI installs the git tag explicitly — do not add a git-URL pin, uv clobbers the workspace editable install, see issue #81).
- Part of the fusion monorepo (`/Users/dahai/fusion`); it shares the root `.venv/`.

## Environment

```bash
cd /Users/dahai/fusion
source .venv/bin/activate
cd fusion-trainer
```

## Common Commands

```bash
pytest tests/                 # Full suite (asyncio_mode=auto, testpaths=tests)
pytest tests/test_mlx_client.py::test_name -v   # Single test
ruff check . && ruff format . # Lint + format
ft --help                     # CLI surface (subcommands: datasets, train, eval, rlsl, merge, flywheel, ...)
```

CI (`.github/workflows/ci.yml`) runs ruff + pytest. `nightly.yml` runs a real small-model training round on a macOS runner.

## Architecture

- `fusion_trainer/cli.py` — `ft` CLI; thin wrappers that build config + call engine via client.
- `fusion_trainer/mlx_client.py` — `MLXFineTuneClient`, all engine HTTP. Headers: `X-Fusion-Route: admin` always; `X-Fusion-Source: model-hub` by default (model-management endpoints are gated on it; env `FUSION_MLX_SOURCE` overrides, explicit `source=` arg wins). Ports: default `11434` (direct engine; the gateway does not proxy `/admin/api/fine-tune/*`), `FUSION_MLX_PORT` override.
- `bridge/server.py` — FastAPI bridge on `:8790`; owns RunManager + the admin key; `trainer.*` RPC surface used by fusion-agent-studio/fusion-studio. `bridge/dashboard.py` — `/dashboard` + `/api/dashboard`.
- `fusion_trainer/dataset/` — dataset builders, including `rollout_ingest.py` (tap JSONL → SFT/GRPO sets, outcome sidecar join by `rollout_id`).
- `fusion_trainer/flywheel.py` + `flywheel_watch.py` + `eval_loop.py` — flywheel loop: eval-gated rounds, curve registry, `ft flywheel watch` (poll `results/<run>/summary.json`, sha256 consumption state).

## Key Invariants / Gotchas

- **httpx timeout**: `request(timeout=None)` *disables* the timeout. To keep the pooled client default pass `httpx.USE_CLIENT_DEFAULT`. `merge_adapter` uses `MERGE_CALL_TIMEOUT = 1800.0`; env override `FUSION_TRAINER_MERGE_TIMEOUT` (explicit arg beats env).
- `_request` is the single egress path — test fakes must implement `request(self, method, url, json=None, params=None, timeout=None)`, not `post()`.
- Engine job `to_dict()` has no `events` array (SSE-only); `_JobEventCursor` synthesizes train_loss events from the `progress` snapshot.
- Fresh-data discipline (issue #82): flywheel rounds record the dataset sha; `--gate` discards a round whose eval pass rate regresses vs the incumbent.
- Local egress: macOS system proxy can 502 loopback httpx requests (`NO_PROXY=localhost,127.0.0.1` if seen).
- Gateway proxy-plane API key comes from `~/.fusion-gateway/config.yaml` `auth.api_keys` (admin entry); the `FUSION_ADMIN_KEY`/admin-panel credential is different and 401s on `/v1/*`.
- `ft` paths resolve relative to the process CWD, not the project dir.

## Conventions

- English-only code comments, log messages, and GitHub artifacts (hard constraint).
- No docstrings; 4-space multiples indentation; every code path carries logging.
- Release: squash-merge PR → master CI green → bump `fusion_trainer/_version.py` + `CHANGELOG.md` → tag `v<version>` + GitHub Release. See `docs/RELEASE_PROCESS.md`.
- `runtime_cache/` is gitignored scratch space — keep final artifacts + logs only.
