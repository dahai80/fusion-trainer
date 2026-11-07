---
doc_id: fusion-trainer/api-reference
product: fusion-trainer
class: platform
entry: service
doc_type: api-reference
title: { zh: "fusion-trainer API 参考", en: "fusion-trainer API reference" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - docs/api-reference.md
---

# fusion-trainer API 参考 / API Reference

三个集成面 / Three integration surfaces: CLI, Python (`RunManager` 等，冻结面见 `docs/API_STABILITY.md`), trainer-bridge HTTP (:8790)。GUI 走 fusion-agent-studio 的 `trainer.*` RPC，不属于本仓库实现。

Three integration surfaces: the CLI, the Python API (`RunManager` etc., frozen surface in `docs/API_STABILITY.md`), and the trainer-bridge HTTP API (:8790). GUIs go through the fusion-agent-studio `trainer.*` RPC dispatcher, which is implemented upstream.

## CLI

入口 / Entry points: `fusion-trainer` and `ft` (pyproject.toml `[project.scripts]`).

| Command | Description |
|---|---|
| `ft sft` | Run supervised fine-tuning. Flags: `--config`, `--preset`, `--model`, `--dataset`, `--method full\|lora\|dora\|qlora`, `--run-name`, `--publish`, `--resume`, `--pack-sequences`, `--save-every` |
| `ft rlsl` | Run DPO/ORPO/RLSL/GRPO alignment. Flags: `--method`, `--sandbox off\|code\|math`, `--group-size`, `--grpo-iters`, `--publish` |
| `ft publish <run_id>` | Publish a completed run's adapter to fusion-model-hub |
| `ft bnup-prep --stage sft\|dpo` | Prepare BNUP training data (SFT DSL validation / DPO pair generation) |
| `ft bnup-train --stage sft\|dpo\|pipeline` | Run BNUP two-stage training (SFT -> DPO -> merge); `--resume` supported |
| `ft merge --base <model> --adapter <path>` | Merge an adapter into the base model via fusion-mlx |
| `ft voice-train --model <kokoro> --dataset <dir>` | Train a TTS voice pack (`--publish` to register) |
| `ft info` | System info & hardware check |
| `ft --version` / `-V` | Print version |

Checkpoint selection flags on `ft sft`: `--eval-select`, `--bench-url` (e.g. `http://localhost:11467`), `--bench-api-key`, `--bench-tenant-id`, `--save-every`.

数据集格式 / Dataset formats:

- SFT: `{"input": ..., "output": ...}` (field names configurable, `dataset.input_field`/`output_field`)
- GRPO prompts: `{"prompt": ..., "answer": ...}` (math reward) or `{"prompt": ..., "tests": ...}` (code reward)
- DPO/ORPO: preference pairs JSONL
- `hub:<category>/<name>` resolves to the fusion-model-hub dataset repo

## Python API (frozen surface)

Per `docs/API_STABILITY.md`, as of 1.0.0 these are SemVer-frozen:

- `TrainerConfig` — dataclass consumed by `RunManager`; fields map to YAML config + env overrides
- `RunManager` — `start_sft`, `start_rlsl`, `stop_run`, `publish_run`, `delete_adapter`, `info`; mutating ops accept an optional `api_key` (`publish_run` uses `trainer_api_key`); `info()` reports gate status + audit path
- `MLXFineTuneClient` — HTTP client to fusion-mlx fine-tune endpoints (`create_job` / `wait_for_job`)
- Public exception types (e.g. `PermissionDeniedError`)

未冻结 / Not frozen: `_`-prefixed modules, `presets.py`, sandbox reward internals, `experiment/` logger internals.

## trainer-bridge HTTP (127.0.0.1:8790)

Loopback bind only. Every data route requires the bearer token (`FUSION_TRAINER_BRIDGE_KEY`); missing/wrong key returns 401 (fail-closed).

| Method + Path | Purpose |
|---|---|
| `GET /` | Console page (`bridge/trainer.html`) |
| `POST /datasets`, `GET /datasets`, `DELETE /datasets` | Dataset management |
| `GET /presets` | SFT/RLSL presets (registry `~/.fusion-trainer-bridge/presets.json`, hot-reloaded) |
| `POST /runs`, `GET /runs` | Start a run / list runs |
| `POST /runs/{id}/stop`, `POST /runs/{id}/merge` | Cancel a run / merge its adapter |
| `POST /publish`, `GET /published` | Publish merged model / list published |
| `POST /eval` | Base-vs-tuned evaluation |
| `GET /health` | Health check |

Publish 语义 / Publish semantics: `POST /publish` promotes the run's merged standalone model (written to `~/.fusion-mlx/models` at merge time, so it enters the fusion-mlx `/v1/models` catalog); `{"run_id": ""}` reverts to the base model.

## trainer.* RPC (served by fusion-agent-studio)

Dispatched to `RunManager` over the studio UDS (`/tmp/fusion-studio.sock`, JSON-RPC 2.0):

`trainer.start_sft`, `trainer.start_rlsl`, `trainer.runs.list`, `trainer.runs.status_full`, `trainer.runs.progress`, `trainer.runs.stop`, `trainer.presets.list`, `trainer.datasets.list`, `trainer.datasets.preview`, `trainer.adapters.list`, `trainer.adapters.delete`, `trainer.info_full`.

## Error shape

REST 路由错误返回 HTTP 状态码 + JSON 错误体；Python 层以异常类型表达（如 key 不匹配的 `PermissionDeniedError`）。审计与发布失败策略：publish 失败告警但不阻塞（exit 0）；audit 写入 `{cache_path}/audit.jsonl`。

REST errors return an HTTP status code plus a JSON error body; the Python layer expresses failures as exception types (e.g. `PermissionDeniedError` on key mismatch). Failure policies: publish failures warn but never block (exit 0); audit records append to `{cache_path}/audit.jsonl`.
