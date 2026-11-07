---
doc_id: fusion-trainer/architecture
product: fusion-trainer
class: platform
entry: service
doc_type: architecture
title: { zh: "fusion-trainer 架构", en: "fusion-trainer architecture" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - docs/architecture.md
---

# fusion-trainer 架构 / Architecture

## 分层 / Layering

```
Consumers:  fusion-studio GUI | fusion-web-shell console | domain CLIs | tests
                | UDS trainer.* RPC      | HTTP :8790         | CLI (ft)
                v                        v                    v
        fusion-agent-studio        trainer-bridge (FastAPI)   fusion_trainer.cli
        TrainerDispatcher          bridge/                    (typer, 66K LOC)
                |                        |                        |
                +-----------+------------+-----------+------------+
                            v
                     fusion_trainer.RunManager        (run_manager.py)
                     run registry / events / audit / publish
                            |
                     MLXFineTuneClient        (mlx_client.py)
                            |  HTTP only, no in-process MLX
                            v
                     fusion-mlx :11434  (gradient loops, adapters)
                            |
              +-------------+------------------+
              v                               v
   fusion-model-hub (via :11432)       fusion-bench :11467
   datasets + published adapters       checkpoint scoring (optional)
```

核心规则 / Core rule: trainer 对 MLX 只走 HTTP（`/admin/api/fine-tune/*`、`/api/fine-tune/grpo/jobs`、VLM 端点），从不 import `mlx-lm`/`mlx-vlm`。所有真实梯度计算在 fusion-mlx 进程内。

The trainer talks to MLX exclusively over HTTP (`/admin/api/fine-tune/*`, `/api/fine-tune/grpo/jobs`, the VLM endpoints) and never imports `mlx-lm`/`mlx-vlm`. All real gradient computation happens inside the fusion-mlx process.

## 模块地图 / Module map (`fusion_trainer/`)

| Module | Role |
|---|---|
| `cli.py` | Typer CLI: `sft`, `rlsl`, `publish`, `bnup-prep`, `bnup-train`, `merge`, `voice-train`, `info` |
| `run_manager.py` | Run lifecycle: registry (`{cache_path}/runs/<run_id>/`), `events.jsonl`, `metrics.prom`, `audit.jsonl`, api-key gate |
| `mlx_client.py` | `MLXFineTuneClient`: job submit (`create_job` -> `wait_for_job`) against fusion-mlx |
| `config.py` | `TrainerConfig` / `RLSLConfig`: YAML + env + CLI merge, `__post_init__` normalization (`policy_gradient` -> `grpo`) |
| `hub_client.py` | `ModelHubClient`: `POST /api/v1/models` (`model_type=lora`) for publish |
| `sft/` | SFT training driver: preset expansion, `_build_mlx_config`, dataset staging, VLM routing |
| `rlsl/` | Alignment pipeline: GRPO (rft / policy-gradient), DPO, ORPO; `sandbox.py` reward scoring; `isolation.py` seatbelt |
| `reward_server.py` | Local HTTP reward endpoint fusion-mlx calls back during true-GRPO runs |
| `dataset/` | Loading, transforms, versioning; `hub:` resolution |
| `experiment/` | Logging, metrics export (Prometheus text format), visualization |
| `merge.py` | Adapter merge into base model via fusion-mlx |
| `flywheel.py`, `eval_loop.py`, `audit.py`, `voice.py` | Feedback-loop sampling, evaluation loop, audit helpers, TTS voice training |

## 关键数据流 / Key flows

### SFT run

1. CLI/preset/config merge -> `TrainerConfig`; `RunManager.start_sft` allocates `run_id`, writes events.
2. `MLXFineTuneClient.create_job` posts to fusion-mlx (`FineTuneConfig`; `qlora` maps to `fine_tune_type=qlora, quantize_base=True, quant_bits=4`).
3. `wait_for_job` polls; checkpoints land in `~/.fusion-mlx/adapters/<model>/<run_name>/`; run end writes `metrics.prom` and appends `audit.jsonl`.
4. If `--eval-select`: each numbered checkpoint is scored via fusion-bench (`adapter_path` forwarding, bench >= 0.4.11); selection table persists to `checkpoint_selection.json` (bench score -> val_loss -> last, ties to later step). Bench outage marks rows `unavailable` and falls back to val_loss — never silent.
5. If `--publish`: `ModelHubClient` registers the adapter; failure warns, exit 0.

### GRPO

`rft` backend (default): on-policy sample -> sandbox score -> top-k -> SFT/LoRA job via `/admin/api/fine-tune/jobs`. `grpo` backend: fusion-mlx runs policy gradient server-side via `/api/fine-tune/grpo/jobs` and calls the local `reward_server.py`. Code-mode reward executes model output under the macOS seatbelt sandbox (`sandbox-exec`: network denied, reads confined, timeout kill) and requires `FUSION_CODE_SANDBOX_TRUSTED=1`; non-macOS falls back to rlimits with a loud warning.

### Publish & merge

`ft merge` calls fusion-mlx to write a standalone model to `~/.fusion-mlx/models` (auto-enters `/v1/models` catalog). The bridge `POST /publish` promotes that merged model; `{"run_id": ""}` reverts to base.

## 进程与端口 / Processes & ports

| Process | Port | Notes |
|---|---|---|
| fusion-trainer CLI runs | n/a | Short-lived; long runs are background threads/jobs in fusion-mlx or the bridge |
| trainer-bridge | 127.0.0.1:8790 | Registered in `architecture/port-registry.yaml`; shell-module heartbeat every 15s; process title `fusion-trainer bridge` |
| fusion-mlx (upstream) | 11434 | Default engine target |
| fusion-gateway (upstream) | 11432 | Hub API proxy + shell routing |
| fusion-bench (upstream) | 11467 | Only used with `--eval-select` |

## 运行物 / Artifacts

| Path | Content |
|---|---|
| `~/.fusion-mlx/adapters/<model>/<run_name>/` | LoRA/DoRA/full adapters, `checkpoint_selection.json` |
| `~/.fusion-mlx/models/` | Merged standalone models (via merge/publish) |
| `{cache_path}/runs/<run_id>/` | `events.jsonl`, `metrics.prom` (default cache: `./runtime_cache`) |
| `{cache_path}/audit.jsonl` | Append-only audit: `{timestamp, actor, action, run_id, detail, ok}` |
| `~/.fusion-trainer-bridge/` | Bridge presets registry, `web_module_key` (0600) |

## 测试与 CI / Testing & CI

- Per-push CI (ubuntu, `.github/workflows/ci.yml`): mocked suite; MLX has no Linux wheel, so live-engine tests are gated by `FUSION_TRAINER_REAL=1` and skip there.
- Nightly (`.github/workflows/nightly.yml`): macOS arm64 runner boots real fusion-mlx with `Qwen3-0.6B-4bit`, runs `tests/test_sft_trainer.py::TestRealFusionMLXIntegration` + `tests/test_contract_integration.py`.
- Soak: `scripts/load_test.sh` (N consecutive SFT rounds with health checks) and `tests/test_load_stability.py` (gated by `FUSION_TRAINER_LOAD=1`).
