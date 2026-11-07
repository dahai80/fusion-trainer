---
doc_id: fusion-trainer/api-stability
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "fusion-trainer API 稳定性契约", en: "fusion-trainer API stability contract" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - docs/API_STABILITY.md
---

# API Stability Contract — fusion-trainer 1.0

**Frozen at:** v1.0.0
**SemVer policy:** [Semantic Versioning 2.0.0](https://semver.org/)

This document defines the **public API surface** of fusion-trainer frozen at
1.0. Anything listed here is covered by the SemVer stability promise: breaking
changes require a major-version bump. Anything NOT listed here is internal and
may change in any release without notice.

---

## 1. CLI Subcommands

The `fusion-trainer` (alias `ft`) CLI exposes these subcommands. Their names,
positional arguments, and long-option spellings are frozen. Adding a new option
is a minor change; removing or renaming one is a major change.

| Subcommand | Purpose |
|---|---|
| `sft` | Run supervised fine-tuning (full / lora / dora / qlora) |
| `rlsl` | Run alignment (dpo / orpo / grpo / rlsl) |
| `publish <run_id>` | Retroactively publish a completed run's LoRA adapter to fusion-model-hub |
| `info` | Print system info + hardware check |
| `--version` | Print `fusion_trainer.__version__` |

There is **no `dataset` subcommand** — dataset import/list/info lives in the
companion project fusion-model-hub. This absence is part of the contract.

---

## 2. `TrainerConfig` (dataclass)

`fusion_trainer.config.TrainerConfig` fields are frozen:

| Field | Type | Default |
|---|---|---|
| `dataset` | `DatasetConfig` | — |
| `sft` | `SFTConfig` | — |
| `rlsl` | `RLSLConfig` | — |
| `device` | `str` | `"mps"` |
| `seed` | `int` | `42` |
| `model_hub_path` | `str` | `$FUSION_MODEL_HUB` or `~/fusion-model-hub` |
| `cache_path` | `str` | `$FUSION_TRAINER_CACHE` or `./runtime_cache` |
| `mlx_port` | `int` | `$FUSION_MLX_PORT` or `11434` |
| `hub_url` | `str` | `$FUSION_MODEL_HUB_URL` or `http://localhost:11432` |
| `hub_api_key` | `str` | `$FUSION_MODEL_HUB_API_KEY` or `""` |
| `publish_adapter` | `bool` | `False` |

Adding a new field with a default is a minor change. Renaming, removing, or
changing the type of an existing field is a major change.

---

## 3. `RunManager` methods

`fusion_trainer.run_manager.RunManager` — the async run + live-progress +
registry surface consumed by the GUI (fusion-agent-studio `TrainingService`).
These method names and their keyword-argument names are frozen:

| Method | Signature (frozen kwargs) |
|---|---|
| `start_sft` | `(config, api_key=None)` |
| `start_rlsl` | `(config, api_key=None)` |
| `run_status` | `(run_id)` |
| `run_progress` | `(run_id, since_step=-1)` |
| `stop_run` | `(run_id, wait=False, timeout=30.0, api_key=None)` |
| `list_runs` | `(limit=200)` |
| `list_presets` | `()` |
| `list_datasets` | `()` |
| `list_adapters` | `()` |
| `preview_dataset` | `(path, max_samples=5)` |
| `publish_run` | `(run_id, hub_url="", hub_api_key="", trainer_api_key=None)` |
| `delete_adapter` | `(adapter_name, model_id, api_key=None)` |
| `info` | `()` |

The singleton entry `get_run_manager()` is frozen. Adding a new method is a
minor change; removing/renaming one is a major change.

---

## 4. `MLXFineTuneClient` methods

`fusion_trainer.mlx_client.MLXFineTuneClient` — the single HTTP boundary to
fusion-mlx. Constructor kwargs frozen: `base_url`, `port`, `api_key`, `timeout`.
Method names frozen:

- Job lifecycle: `create_job`, `get_job`, `cancel_job`, `wait_for_job`,
  `list_jobs`
- DPO: `create_dpo_job`, `get_dpo_job`, `cancel_dpo_job`, `wait_for_dpo_job`,
  `list_dpo_jobs`
- ORPO: `create_orpo_job` (shares the dpo wait/list/cancel shape)
- GRPO: `create_grpo_job`, `get_grpo_job`, `cancel_grpo_job`,
  `wait_for_grpo_job`, `list_grpo_jobs`
- Reward model: `create_reward_job`, `get_reward_job`, `cancel_reward_job`,
  `wait_for_reward_job`, `list_reward_jobs`, `score_reward`
- Adapters: `list_adapters`, `serve_adapter`
- Serving: `chat_completions`
- Introspection: `health`, `list_finetunable_models`
- Data: `stage_dataset`
- Lifecycle: `close`

`MLXFineTuneError` is the frozen exception type raised on HTTP/transport
failures.

---

## 5. Exception types

These public exception types are frozen (name + inheritance):

- `fusion_trainer.mlx_client.MLXFineTuneError`
- `fusion_trainer.sft.SFTUnsupportedError`
- `fusion_trainer.rlsl.RLSLUnsupportedError`
- `fusion_trainer.hub_client.HubPublishError`

---

## 6. SemVer contract

- **Major (2.0.0):** a breaking change to any frozen surface above — removed or
  renamed CLI subcommand/option, renamed `TrainerConfig` field, renamed
  `RunManager`/`MLXFineTuneClient` method, changed constructor signature.
- **Minor (1.1.0):** new subcommand, new `TrainerConfig` field with default,
  new `RunManager`/`MLXFineTuneClient` method, new exception type. Backward-
  compatible.
- **Patch (1.0.1):** bug fixes, doc changes, test additions, internal refactors
  that leave every frozen surface identical.

Internal modules (`_`-prefixed, `presets.py` contents, `sandbox.py` reward
internals, `experiment/` logger internals) are NOT frozen and may change in
any release.

---

## 7. Upstream-bound surface (NOT under this contract)

These are blocked on fusion-mlx and explicitly excluded from the 1.0 freeze —
their behavior may change when the upstream lands, without a fusion-trainer
major bump:

- MXFP8 mixed precision (fusion-mlx#402) — `use_mxfp8` defaults `False`, an
  explicit `True` fail-stops today.
- OS-level sandbox isolation (fusion-mlx#743) — current seatbelt/rlimit
  isolation is a defense-in-depth layer, not a full container.
- Prometheus scrape endpoint — fusion-trainer exports a `.prom` file only; the
  scrape endpoint lives upstream in fusion-agent-studio.
