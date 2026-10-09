# Changelog

All notable changes to fusion-trainer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are tracked by the `__version__` in `fusion_trainer/_version.py` and
by git tags of the form `v<version>`.

## [1.0.0rc5] - 2026-10-10

BNUP two-stage training pipeline — data prep (#56), DPO negative factory
(#57), model merge/export (#58), and BNUP SFT+DPO config + pipeline
orchestration (#59).

### Added
- **BNUP data preparation (#56):** `dataset/bnup_prep.py`
  (`validate_dsl_json` via fusion-core, `build_sft_sample`, `process_raw_sft`)
  + `dataset/bnup_dpo_prep.py` (`create_dpo_pairs`, `normalize_for_trainer`).
  CLI: `ft bnup-prep --stage sft|dpo`.
- **DPO negative sample factory (#57):** `dataset/bnup_negatives.py` — 5
  generators (`make_broken_json`, `make_direct_answer`,
  `make_wrong_visual_type`, `make_missing_tag`, `make_format_drift`) +
  `generate_negatives` (configurable ratio, max 5 distinct per sample).
- **Model merge + export (#58):** `merge.py` (`merge_adapter`,
  `export_merged_model`, `validate_merged_model`, `merge_and_export`) +
  `MLXFineTuneClient.merge_adapter` (POST /v1/merge-adapter). CLI: `ft merge`.
- **BNUP two-stage config (#59):** `configs/bnup_sft_32b.yaml` (8-bit QLoRA,
  cutoff 4096) + `configs/bnup_dpo_32b.yaml` (beta=0.1, sigmoid). Presets
  `bnup_sft_32b` / `bnup_dpo_32b`. CLI: `ft bnup-train --stage sft|dpo|pipeline`.
- **fusion-core dependency:** `validate_dsl_str` used for DSL schema validation
  (no duplicated schema). `jsonschema` added as explicit dep.
- **Contract test:** `test_contract_merge_adapter_route` verifies the
  merge-adapter route exists on live fusion-mlx.

### Upstream
- Filed [fusion-mlx#1149](https://github.com/dahai80/fusion-mlx/issues/1149):
  DPO training does not support loading a pre-trained adapter
  (`init_adapter_path`). Until it lands, the BNUP pipeline runs DPO as a
  fresh adapter (not chained on SFT output) and merges the DPO adapter with
  the base. The DPO config documents this gap.

## [1.0.0rc4] - 2026-09-06

Patch release — code-formatting only, no behavior change.

### Changed
- Applied `ruff format` to `experiment/logger.py`, `run_manager.py`, and the
  `test_audit_run_manager`/`test_isolation`/`test_load_stability` tests
  (line-wrapping only). Re-formatted the seatbelt-isolation plan doc.

## [1.0.0rc3] - 2026-09-06

Vision-language (VLM) fine-tune wiring — closes [#55](https://github.com/dahai80/fusion-trainer/issues/55),
consuming the upstream fusion-mlx VLM endpoint ([#797](https://github.com/dahai80/fusion-mlx/issues/797) → fixed in [#800](https://github.com/dahai80/fusion-mlx/pull/800), fusion-mlx v0.8.80).

### Added
- **VLM fine-tune pipeline (#55)**: `SFTTrainer` now routes `model_type=="vlm"`
  to the new fusion-mlx VLM endpoint (`POST /admin/api/fine-tune/vlm/jobs`),
  which drives `mlx_vlm.trainer` SFT/LoRA over HTTP. This is the same pure-
  HTTP-delegation pattern as text SFT — fusion-trainer never imports
  `mlx-vlm`; the real gradient loop runs inside fusion-mlx. The existing
  `SFTConfig.model_type` field (a placeholder since it was introduced) and
  the `sft_vlm_7b_qlora` preset are now actually wired.
  - `MLXFineTuneClient` gained `create_vlm_job` / `get_vlm_job` /
    `cancel_vlm_job` / `list_vlm_jobs` / `wait_for_vlm_job` mirroring the
    text-SFT lifecycle on the VLM path (same `events[]` append-only +
    terminal-status contract, so `wait_for_job`'s retry/cancel logic reuses).
  - `SFTTrainer._build_vlm_config` maps `SFTConfig` → the upstream
    `VLMFineTuneConfig` field set (`lora_rank`/`lora_alpha`/`lora_dropout`/
    `lora_layers`/`full_finetune`/`train_vision`/`optimizer`/`learning_rate`/
    `batch_size`/`iters`/`val_batches`/`steps_per_*`/`max_seq_length`/
    `grad_checkpoint`/`grad_accumulation_steps`/`seed`/`num_processes`).
    Only the declared fields are sent — the text-SFT-only keys
    (`lr_schedule`, `mask_prompt`, `quant_bits`, `quantize_base`, `mxfp8`,
    `lora_target_modules`) are NOT emitted (would 400 upstream:
    `VLMFineTuneConfig(**body)` raises `TypeError` on unknown keys).
  - `SFTTrainer._submit_vlm_job` does NOT call `stage_dataset` — mlx-vlm's
    `load_dataset` consumes a HF dataset id or a local dir of images + jsonl
    directly, not the staged chat-messages jsonl. The dataset reference is
    resolved from `dataset.path` (HF id or local dir) or `dataset.hub_dataset`
    (hub convention, same path-traversal guard as `DatasetLoader`).
  - Result dict carries `model_type="vlm"` so the GUI / publish path can
    distinguish a VLM adapter.

### Changed
- VLM QLoRA caveat: `VLMFineTuneConfig` has no quantize field, so a VLM run
  with `method="qlora"` degrades to plain LoRA (`full_finetune=False`) with a
  loud `logger.warning` (fail-visible) — no base quantization occurs. mlx-vlm
  QLoRA semantics differ from mlx-lm and are not exposed upstream (#797).

## [Unreleased]

Enterprise-readiness soft-issue remediation (soft #2–#5) plus a real-model
GUI end-to-end integration test that caught three production defects the
mock-based suite could not.

### Added
- **enterprise #3 — Prometheus metrics export**: `ExperimentLogger.export_prometheus`
  and module-level `export_prometheus_snapshot` write a Prometheus
  text-format (`.prom`) snapshot — one gauge per scalar metric key
  (`fusion_trainer_<metric>{run="<id>"} <value>`), plus `run_steps_total` /
  `run_total_steps` / `run_status_code` supplied by `RunManager`. Metric
  names are sanitized to the Prom `[a-zA-Z_:][a-zA-Z0-9_:]*` grammar; non-
  scalar values are skipped. `RunManager._export_run_metrics` emits
  `{cache_path}/runs/<run_id>/metrics.prom` at run end (deterministic from
  record state, Rule 5; fail-visible, never raises into the run path). A
  scrape endpoint lives upstream (fusion-agent-studio); this is a file
  export for a node-exporter textfile collector or sidecar.
- **enterprise #4 — load / stability verification**: `tests/test_load_stability.py`
  (markers `integration`+`load`, gated by `FUSION_TRAINER_LOAD=1`) drives N
  consecutive SFT runs through `RunManager` against a real fusion-mlx and
  asserts no state leak between rounds — each run reaches `running` then
  `completed`, produces a non-empty `events.jsonl`, and the registry
  remembers prior runs. Default 3 rounds, overridable via
  `FUSION_TRAINER_LOAD_ROUNDS`. Auto-skips without a live server. The
  companion `scripts/load_test.sh` runs an operator-facing shell soak (N
  consecutive SFT runs with `~/claude-home/fusion-mlx/start.sh doctor`
  health checks between, default 5, exits non-zero on the first failed
  round so a CI `&&` chain catches it). Mock-based suites cannot surface
  the class of bugs that only appear on the Nth consecutive run (leaked
  file handles, accumulating singleton state, cross-run event-log
  corruption); this closes that gap.
- **enterprise #6 — change management / release process**:
  `docs/RELEASE_PROCESS.md` defines the branch strategy (`release/vX.Y.Z`
  release branches off `master`), the version-bump rules (one source of
  truth in `_version.py`), the release-cut steps, the release-PR review
  checklist, the hotfix flow, and the rollback policy (never rewrite/delete
  published tags; yank + cut a patch instead). Pairs with
  `docs/API_STABILITY.md` (the SemVer frozen-surface contract).

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
- **enterprise #2 — RunManager audit log + api-key gate**: `RunManager` now
  keeps an append-only JSONL audit trail at `{cache_path}/audit.jsonl` — one
  line per mutating op (`start_sft`, `start_rlsl`, `stop_run`, `publish_run`,
  `delete_adapter`): `{timestamp, actor, action, run_id, detail, ok}`. Actor
  comes from `FUSION_TRAINER_ACTOR` (default `local`). An optional api-key gate
  (`FUSION_TRAINER_API_KEY` set) rejects mutating ops with
  `PermissionDeniedError` on a mismatched key (constant-time compare); when the
  env is unset, local-trust mode allows all ops (appropriate for single-user
  Mac). `info()` reports whether the gate is active and the audit-log path.
  `start_sft`/`start_rlsl`/`stop_run`/`delete_adapter` take an optional
  `api_key`; `publish_run` takes `trainer_api_key` (distinct from the hub
  `api_key` so a deployment can gate ops without sharing the hub secret).

### Changed
- **soft #3 — GUI publish now native**: the GUI path
  (fusion-agent-studio `TrainingService._build_trainer_config`) previously
  dropped `publish_adapter`/`hub_url`/`hub_api_key` from the config it built,
  so a GUI-triggered run never published even when the operator wanted it to.
  Upstream [fusion-agent-studio#277](https://github.com/dahai80/fusion-agent-studio/issues/277)
  now forwards these fields, and
  [fusion-studio#390](https://github.com/dahai80/fusion-studio/issues/390)
  exposes them in the Trainer panel — GUI-triggered publish fires natively
  without the env bypass. The `FUSION_TRAINER_PUBLISH=1` /
  `FUSION_MODEL_HUB_URL` / `FUSION_MODEL_HUB_API_KEY` env fallback remains as
  an override.

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

## [1.0.0rc2] — 2026-09-05

**Second release candidate.** All six former upstream blockers are now CLOSED;
this RC removes the stale client-side guards that referenced them and consumes
the newly-landed upstream features. No public API surface changes — the
[API stability freeze](docs/API_STABILITY.md) from rc1 holds.

- **Version bump** `1.0.0rc1` → `1.0.0rc2` (`fusion_trainer/_version.py`).

### Fixed
- **MXFP8 passthrough (fusion-mlx #402/#425 CLOSED)**: removed the stale
  `MLXFineTuneError` fail-stop that refused all `use_mxfp8=True`. Upstream
  `service.py` now routes `mxfp8=True` to the 8-bit QLoRA path
  (`quantize_base=True`, `quant_bits=8`, `fine_tune_type=qlora`) — 8-bit frozen
  base + LoRA for memory saving, NOT fp8 compute (MLX 0.32 has no fp8 dtype).
  fusion-trainer now passes `mxfp8=True` through to the job config verbatim.
  `mxfp8+full` remains a contradiction (full unfreezes the base, nothing to
  quantize) and still fails visibly — now client-side at job-build with a clear
  message before submission (defense-in-depth; upstream also raises). The
  `run_manager.info()` mxfp8 warning drops from `warn` to `info` and reflects
  the 8-bit QLoRA routing.
- **GRPO true backend (fusion-mlx #751 CLOSED)**: the `There is no Stream(gpu,
  N) in current thread` regression is fixed upstream — the worker rebinds
  `_mlx_gen.generation_stream = mx.new_thread_local_stream(...)`. Verified
  in-tree on fusion-mlx 0.8.79; the `/admin/api/fine-tune/grpo/jobs` endpoint
  is live (admin-auth gated). Both `rft` and `grpo` backends run end-to-end.

### Changed
- **Hub idempotency uses exact `name=` filter (fusion-models-hub #51 CLOSED)**:
  `ModelHubClient.list_models` gains a `name=` param (exact, tenant-scoped —
  the same lookup the 409 collision check uses server-side).
  `_find_existing_lora` now queries with `name=` instead of the fuzzy
  `keyword=`, which could not distinguish "my adapter" from "another model
  sharing a token". `keyword=` remains a fallback for older hub versions.
- **GUI publish now native (fusion-studio #390 + fusion-agent-studio #277
  CLOSED)**: the GUI path forwards `publish_adapter`/`hub_url`/`hub_api_key`
  natively, so GUI-triggered runs publish without the env-var bypass. The
  `FUSION_TRAINER_PUBLISH=1` / `FUSION_MODEL_HUB_URL` /
  `FUSION_MODEL_HUB_API_KEY` env fallback remains as an override.

### Docs
- Updated CLAUDE.md (current state, publish flow, decisions 7 & 13), README.md
  and README_CN.md (MXFP8 ✅, DPO/ORPO/RLSL ✅, GUI publish native, version
  badge rc2) to reflect all six resolved upstream blockers.
- README_CN DPO/ORPO/RLSL/QLoRA sections synced to the English README's
  resolved state (were stale at "上游阻塞 ⏳").
- fusion-trainer #55 (GUI-VL fine-tune) triaged: VL fine-tune is still an
  upstream gap (no `mlx-vlm` training endpoint in fusion-mlx). Upstream
  feature request filed as [fusion-mlx#797](https://github.com/dahai80/fusion-mlx/issues/797);
  fusion-trainer #55 stays open as the landing tracker.

## [1.0.0rc1] — 2026-09-02

**First release candidate.** The public API surface is now frozen under
[Semantic Versioning](https://semver.org/). See
[`docs/API_STABILITY.md`](docs/API_STABILITY.md) for the full freeze list —
CLI subcommands, `TrainerConfig` fields, `RunManager` methods,
`MLXFineTuneClient` methods, and public exception types. This is a release
candidate, not a stable release: the training loops, security gates, audit
trail, metrics export, and publish-to-hub flow are all real and verified, but
three upstream gaps block the final `1.0.0` (listed below). `1.0.0` is cut
only when those land or are explicitly descoped.

- **Version bump** `0.1.12` → `1.0.0rc1` (`fusion_trainer/_version.py`).
- **API stability contract** added at `docs/API_STABILITY.md`: defines which
  surfaces are frozen (SemVer-protected) vs internal (may change anytime), and
  the SemVer rules for major/minor/patch against that surface.
- **Release process** documented at `docs/RELEASE_PROCESS.md`: branch strategy,
  version-bump rules, release-cut steps, review checklist, hotfix flow,
  rollback policy.
- **Upstream-bound surface** explicitly excluded from the freeze, and formerly
  blocking the `1.0.0` stable cut — **all now resolved** as of `1.0.0rc2`:
  - **MXFP8 mixed precision** — fusion-mlx#402/#425 CLOSED. `use_mxfp8=True`
    now passes through and routes to 8-bit QLoRA (8-bit frozen base + LoRA,
    not fp8 compute; MLX 0.32 has no fp8 dtype). `mxfp8+full` still fails
    visibly. Resolved in `1.0.0rc2`.
  - **GRPO true backend** — fusion-mlx#751 CLOSED. The worker rebinds
    `generation_stream` to a thread-local stream; real policy-gradient GRPO
    runs end-to-end. Resolved upstream, verified in-tree on 0.8.79.
  - **OS-level sandbox isolation** — fusion-mlx#743 CLOSED. fusion-mlx shipped
    server-side `code_sandbox.py`; fusion-trainer's seatbelt backend
    (`isolation.py`) remains as client-side defense-in-depth.
  - **GUI publish-to-hub** — fusion-studio#390 + fusion-agent-studio#277
    CLOSED. The Trainer panel exposes `publish_adapter`/`hub_url`/
    `hub_api_key` and the agent forwards them, so GUI-triggered runs publish
    natively (env-var bypass still works as an override). CLI `--publish`
    unaffected.
  - **Prometheus scrape endpoint** — lives upstream in fusion-agent-studio;
    this release exports `.prom` textfiles only.

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

### Upstream-resolved
- **GRPO true backend (was fusion-mlx #751)**: real policy-gradient GRPO jobs
  (`grpo_backend="grpo"`, `/admin/api/fine-tune/grpo/jobs`) are unblocked.
  Root cause was `There is no Stream(gpu, N) in current thread` — `mlx_lm.generate`
  holds a module-level thread-local `generation_stream` captured at import time
  in the loader thread, not the `asyncio.to_thread` worker. Fixed in
  `grpo_service.py` (fusion-mlx #751): the worker rebinds
  `_mlx_gen.generation_stream = mx.new_thread_local_stream(mx.default_device())`
  and runs `with mx.stream(worker_stream):`. Verified in-tree on fusion-mlx
  0.8.79; the `/admin/api/fine-tune/grpo/jobs` endpoint is live (admin-auth
  gated). Both `rft` (reward-filtered SFT) and `grpo` (real policy gradient)
  backends now run end-to-end.

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
completion (issue #49, PR #51). (At this release MXFP8 mixed precision was
upstream-blocked — fusion-mlx#402 — and `use_mxfp8` fail-stopped on `True`;
#402/#425 later closed in `1.0.0rc2`, routing `use_mxfp8=True` to 8-bit QLoRA.)

## [0.1.10] — 2026-08

RunManager live-progress API + GUI integration spec (PR #48). fusion-trainer
drivable in-process from fusion-studio via fusion-agent-studio `trainer.*`
RPC over UDS.

## [0.1.9] — 2026-08

GRPO endpoint path `/admin` prefix fix (#42/#45).
