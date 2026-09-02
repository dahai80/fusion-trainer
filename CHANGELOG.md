# Changelog

All notable changes to fusion-trainer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are tracked by the `__version__` in `fusion_trainer/_version.py` and
by git tags of the form `v<version>`.

## [Unreleased]

Enterprise-readiness soft-issue remediation (soft #2–#5) plus a real-model
GUI end-to-end integration test that caught three production defects the
mock-based suite could not.

### Fixed
- **P0 — `get_run_manager()` self-deadlock**: the singleton entry acquired
  `_SINGLETON_LOCK` (a plain `threading.Lock`) then called `RunManager()`,
  whose `__init__` re-acquired the same lock to emit the A3 singleton-bypass
  warning. A plain `Lock` self-deadlocks on that re-acquire, so the supported
  `get_run_manager()` path hung forever (while a direct `RunManager()` worked).
  Changed `_SINGLETON_LOCK` to `threading.RLock` so the same thread reenters.
  The CLI/GUI singleton path was silently broken since the A3 warning landed.
- **`run_status` dropped `events_path`**: the GUI reconnect flow replays run
  history from `events.jsonl` on disk, but `run_status` never returned the
  path, so a reconnect could not locate the event stream. Added `events_path`
  to the returned dict.
- **`publish_run` adapter existence check read the wrong field**: fusion-mlx
  `GET /admin/api/fine-tune/adapters` returns each adapter as
  `{model_id, adapter_name, adapter_path, ...}`, but `publish_run` matched on
  `a.get("name")` (always `""`), so the L4 existence check always reported the
  adapter missing and refused to publish — a trained adapter could never be
  retroactively published via `publish_run`. Now reads `adapter_name` (with a
  `name` fallback for older servers).

### Security
- **soft #2 — sandbox code-mode defense-in-depth**: `reward_completion` code
  mode now sets `RLIMIT_CPU` (reliably kills infinite loops via SIGXCPU) and
  `RLIMIT_FDATA`/`RLIMIT_NPROC` before exec, in addition to the existing
  restricted temp dir. `RLIMIT_AS` is unreliable on Apple Silicon (mapped
  memory doesn't count) and is not relied on. OS-level isolation (container/
  seccomp) remains tracked at [fusion-mlx#743](https://github.com/dahai80/fusion-mlx/issues/743);
  the CLI `--sandbox` default stays `off` and `reward_code` still requires
  `FUSION_CODE_SANDBOX_TRUSTED=1`.
- **soft #2 / #53 — seatbelt isolation backend**: `reward_code` now runs
  model-generated code under a real MAC isolation layer on macOS via
  `sandbox-exec` (seatbelt): network denied, filesystem restricted to the
  tmpfs workspace + read-only interpreter prefix, killed on timeout. Selected
  automatically on Darwin (`FUSION_CODE_SANDBOX_BACKEND=auto`, default);
  `=rlimit` forces the existing rlimit-only path; `=seatbelt` on a platform
  without the binary raises `SandboxIsolationError` (fail-visible, never a
  silent downgrade). The rlimit prelude still runs *inside* the seatbelt child
  as an inner CPU cap. `FUSION_CODE_SANDBOX_TRUSTED=1` gate unchanged. Real
  container isolation for the inference-server reward path stays upstream at
  [fusion-mlx#743](https://github.com/dahai80/fusion-mlx/issues/743).
- **#53 — seatbelt data-read narrowed**: the seatbelt profile previously allowed
  `(allow file-read* (subpath "/"))` — read-only access to every path — because
  enumerated per-dir allow-lists SIGABRT'd during dyld-init (cryptex/firmlink
  mounts cross subpath boundaries). Replaced with `(import "system.sb")` (pulls
  in `dyld-support.sb`, solving the dyld-init crash) plus
  `(allow file-read-metadata (subpath "/"))` (broad lstat for realpath ancestor
  walks — leaks directory structure, not contents) and
  `(allow file-read-data ...)` narrowed to the interpreter prefix + stdlib +
  lib-dynload + tmpdir. Reads of `~/.ssh`, `/etc/hosts`, and sibling project
  source are now denied. Threat model shifts from buggy-not-malicious to
  malicious-capable for data-exfiltration attempts; timing side-channels remain
  out of scope ([fusion-mlx#743](https://github.com/dahai80/fusion-mlx/issues/743)).

### Changed
- **soft #3 — GUI publish env-fallback**: the GUI path
  (fusion-agent-studio `TrainingService._build_trainer_config`) drops
  `publish_adapter`/`hub_url`/`hub_api_key` from the config it builds, so a
  GUI-triggered run never published even when the operator wanted it to
  (upstream [fusion-agent-studio#268](https://github.com/dahai80/fusion-agent-studio/issues/268)).
  `publish_adapter_if_enabled` now honors `FUSION_TRAINER_PUBLISH=1` to force
  publish on, and `FUSION_MODEL_HUB_URL`/`FUSION_MODEL_HUB_API_KEY` supply the
  endpoint, so GUI runs can publish via env until #268 lands.

### Tests
- **soft #4 — real-model GUI end-to-end integration test**:
  `test_gui_run_manager_sft_live_e2e` drives `RunManager.start_sft` through a
  real fusion-mlx SFT LoRA job (2-iter on the smallest loaded text model) and
  asserts the live-progress contract the GUI renders against: status reaches
  `running` then `completed`, `adapter_name` is populated, `events.jsonl`
  exists + non-empty, and the run appears in `list_runs`. Gated by
  `FUSION_TRAINER_REAL=1`; auto-skips otherwise. This test is what surfaced
  the three `Fixed` defects above — the mock-based RunManager suite patched
  `httpx` and asserted trainer-side assumptions, never the real upstream
  shapes. Also fixed the contract suite's server-reachability probe
  (admin-token header + 20s timeout for a large-cache server) and the
  `base_model_id` fixture (filter `text_generation` capability, prefer the
  smallest model so the smoke run is fast and never picks an image/video
  model like FLUX/Wan).

### Chores
- **soft #5 — `runtime_cache/` products gitignored**: adapter checkpoints,
  staged datasets, and run artifacts written under `runtime_cache/` are now
  excluded from version control.

## [1.0.0] — 2026-09-02

**First stable release.** The public API surface is now frozen under
[Semantic Versioning](https://semver.org/). See
[`docs/API_STABILITY.md`](docs/API_STABILITY.md) for the full freeze list —
CLI subcommands, `TrainerConfig` fields, `RunManager` methods,
`MLXFineTuneClient` methods, and public exception types.

- **Version bump** `0.1.12` → `1.0.0` (`fusion_trainer/_version.py`).
- **API stability contract** added at `docs/API_STABILITY.md`: defines which
  surfaces are frozen (SemVer-protected) vs internal (may change anytime), and
  the SemVer rules for major/minor/patch against that surface.
- **Upstream-bound surface** explicitly excluded from the freeze: MXFP8
  (fusion-mlx#402), OS-level sandbox isolation (fusion-mlx#743), Prometheus
  scrape endpoint (fusion-agent-studio).

## [0.1.12] — 2026-09-01

Round-3 product-readiness audit (`audit/fusion-trainer-audit-result-product-0901.md`)
remediation: closes all P0–P3 checklist items, plus a fail-visible defect found
during the mandatory real-machine smoke test.

### Fail-visible (smoke-test finding)
- **GRPO status reporting**: `GRPOTrainer.train` (both `rft` and `grpo`
  backends) and `RLSLEngine.run` unconditionally printed `✓ GRPO complete!` /
  `✓ Alignment complete!` even when the job `status` was `failed`. A failed
  true-GRPO job now prints `✗ GRPO failed: status=failed error=<msg>` and the
  engine prints `✗ Alignment failed`; the CLI already exited non-zero on a
  non-`completed` status, now the human-readable output matches. The
  `grpo`-backend return dict also carries the upstream `error` string.

### Upstream-blocked
- **GRPO true backend (regression of fusion-mlx #430)**: real policy-gradient
  GRPO jobs (`grpo_backend="grpo"`, `/admin/api/fine-tune/grpo/jobs`) fail
  within ~0.3s with `There is no Stream(gpu, N) in current thread`. Re-verified
  on fusion-mlx 0.8.63 (dce6ff15): the `worker_stream` fix in
  `grpo_service.py:250` is present but insufficient — `mlx_lm.generate` holds a
  module-level thread-local `generation_stream` (created at import time in the
  loader thread, not the `asyncio.to_thread` worker), so `generate_step`'s
  `async_eval` still has no stream in the worker thread. Tracked at
  [fusion-mlx#751](https://github.com/dahai80/fusion-mlx/issues/751)
  (comment 5492669837). The `rft` backend (reward-filtered SFT via
  `/admin/api/fine-tune/jobs`) is unaffected and runs end-to-end.

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
