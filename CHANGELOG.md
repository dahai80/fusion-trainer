# Changelog

All notable changes to fusion-trainer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are tracked by the `__version__` in `fusion_trainer/_version.py` and
by git tags of the form `v<version>`.

## [0.1.12] — 2026-09-01

Round-3 product-readiness audit (`audit/fusion-trainer-audit-result-product-0901.md`)
remediation: closes all P0–P3 checklist items.

### Security
- **P3-6 / S6**: Added `validate_service_url` scheme+host allow-list applied
  at `MLXFineTuneClient` and `ModelHubClient` construction. Only `http`/`https`
  schemes accepted; loopback/wildcard IPs (`127.0.0.0/8`, `::1`, `0.0.0.0`)
  must be written as the literal host `localhost`. A tainted `--config` YAML
  can no longer redirect `hub_url`/`base_url` to an attacker server and
  exfiltrate the Bearer key on publish. Rejects raise `URLGuardError`.
- **P2-13**: GRPO sandbox `RewardServer` prompt-miss is now fail-visible.
  `FUSION_REWARD_PROMPT_MISS=fail` → 503, `=zero` → reward 0.0, default
  `warn` keeps legacy behavior.
- **P2-4**: `RewardServer` gained a latency circuit breaker
  (`FUSION_REWARD_LATENCY_BUDGET_S`, `FUSION_REWARD_LATENCY_TRIP_AFTER`) that
  trips 503 after consecutive slow scores, preventing a hung scoring path
  from stalling a multi-hour GRPO job.

### Changed
- **P3-5 / O18**: `run_status.metrics` now derives `it_sec`, `elapsed_s`, and
  `progress_pct` locally (deterministic, no model routing) and passes through
  upstream throughput fields, so a status poll answers "how fast / how far"
  without re-parsing `events.jsonl`.
- **P3-5 / O19**: `stop_run(run_id, wait=False, timeout=30.0)` — `wait=True`
  joins the run worker thread (with timeout) and returns the final terminal
  status instead of fire-and-forget `cancelling`. Default stays
  fire-and-forget so GUI cancel flows don't block the RPC thread.
- **P3-4 / F6**: SFT `full` method no longer sends `lora_rank`/`lora_alpha`/
  `lora_dropout`/`lora_layers` to the upstream job config (only LoRA-family
  methods emit them).
- **P3-4 / F7**: CLI `rlsl` gained `--reward-lr`, `--reward-batch-size`,
  `--reward-epochs` flags, wired end-to-end to reward-model training config.
- **P2-6**: `RunManager.list_adapters` returns `{"items": [...], "error": ...}`
  instead of a bare list, so the GUI can distinguish "no adapters" from
  "fusion-mlx unreachable".
- **P2-3**: `preview_dataset` uses the cached sample counter and is
  format-aware (parquet/csv/json/jsonl) with path-traversal guarding.
- **P2-2**: Dataset loader enforces per-line (`FUSION_MAX_LINE_BYTES`) and
  per-JSON-file (`FUSION_MAX_JSON_FILE_BYTES`) byte caps as depth-of-defense
  against attacker-controlled oversized dataset files.
- **P3-1**: `_RunSink` lock annotated `threading.RLock`; `_persist_meta`
  writes `run_meta.json` via tmp+`fsync`+`os.replace` (atomic, crash-safe).
- **P3-2**: RLSL engine and its GRPO sub-trainer share a single
  `ExperimentLogger`, so one run produces one `events.jsonl` + TensorBoard
  log instead of double handles on the same file.
- **P3-3**: Documented the daemon-level shared-`MLXFineTuneClient` decision
  and the `wait_for_*_job` append-only event-increment upstream contract.
- **P2-9**: `experiment.__all__` trimmed to `["ExperimentLogger"]` only;
  `ExperimentTracker`/`TrainingVisualizer` remain importable but are no
  longer promoted as the public API (not wired into trainers).

### Tests
- Added `TestValidateServiceUrl` covering scheme/loopback/empty rejection.
- `test_list_adapters_proxies` + `test_list_adapters_down_returns_error`
  assert the new `{items, error}` shape.

## [0.1.11] — 2026-09-01

Closed all 13 defects from the round-2 adversarial audit
(`audit/fusion-trainer-audit-0901.md`): reward-server token auth + path
validation, sandbox code-mode default-off + env gate, per-prompt top-k, mxfp8
default-off + fail-stop, HTTP connection reuse + retry/backoff, RunManager
singleton, atomic meta writes, publish 409 idempotency, and a contract
integration test. Trained LoRA adapters publish to fusion-model-hub on
completion (issue #49, PR #51). Only MXFP8 mixed precision remains
upstream-blocked (fusion-mlx#402) — `use_mxfp8` defaults to `False`; an
explicit `True` fail-stops.

## [0.1.10] — 2026-08

RunManager live-progress API + GUI integration spec (PR #48). fusion-trainer
drivable in-process from fusion-studio via fusion-agent-studio `trainer.*`
RPC over UDS.

## [0.1.9] — 2026-08

GRPO endpoint path `/admin` prefix fix (#42/#45).
