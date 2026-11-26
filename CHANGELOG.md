---
doc_id: fusion-trainer/changelog
product: fusion-trainer
class: platform
entry: service
doc_type: changelog
title: { zh: "fusion-trainer 变更日志", en: "fusion-trainer changelog" }
audience: engineer
status: draft
version: 1.0.0rc37
updated: 2026-11-25
owner: "@dahai"
source_of_truth:
  - CHANGELOG.md
---

# Changelog

## [1.0.0rc37] - 2026-11-26

### Added

- **Failure-trajectory channel (#111):** `ft datasets from-rollouts --failures-output` writes scored-negative rollouts (reward < min-score, unobserved excluded) as full records with outcome, feeding the resynthesis stage; an empty failure set writes an empty file (honest), never an error. The manifest carries a `weakness` profile (per-signal failure counts + bounded sample rollout_ids) for targeted resynthesis.
- **Efficiency pre-step (#111, CARE-PO):** offline redundant consecutive tool-call rate (same tool name + normalized arguments, reset at user turns) computed on every ingest and recorded in the manifest as `efficiency` — the BEFORE number for any future efficiency reward; the budget anchor must be fixed before training, never policy-relative.

Remaining #111 items (environment snapshots, validator audit gate, cross-app synthesis, expert injection, efficiency reward proper) stay open on the issue — cross-repo or training-time scope.

## [1.0.0rc36] - 2026-11-26

### Fixed

- **Merged-model publish 422 + wrong idempotency partition (#113):** `publish_merged_model` sent `model_type: "mlx"`, but the hub `ModelType` enum has no `mlx` value (`mlx` is a `ModelFormat`, and the `POST /api/v1/models` `ModelCreate` schema takes no format field) — merged-model publish 422'd out of the box and the 409 idempotency lookup (`_find_existing_model(name, "mlx")`) searched the wrong type partition. A merged model now registers as `model_type=llm` in both the payload and the lookup; adapter (`lora`) and voice (`audio`) paths unchanged.

## [1.0.0rc35] - 2026-11-26

### Changed

- **Hub API key domain docs (#109, #110):** `FUSION_MODEL_HUB_API_KEY` must be a hub-issued `fmh-*` key when pointing directly at the hub (`:11444`) — the gateway admin key is a different trust domain and 401s by design. Documented the mint path (`fmh keys create`, models-hub >= 1.2.23), the public zero-key bootstrap endpoint (plaintext shown once), and rotation in README/README_CN `--publish` note, `ft publish --hub-api-key` help, `config.py` field doc, `docs/config-reference.md`, and `docs/troubleshooting.md`. Docs + help text only, no behavior change.

## [1.0.0rc34] - 2026-11-26

### Changed

- **Audit #106 closeout:** adds `CLAUDE.md` (repo guidance for agents: HTTP delegation architecture, commands, gotchas, release process) with required docs-contract frontmatter (validator 0 errors), and removes the `.gitignore` rule excluding it (rule predated the repo going public; file carries no secrets). PR #108.
- **English source cleanup (#106/#107):** remaining Chinese comments and log strings in `fusion_trainer/`, `bridge/`, `ci/` translated to English — comment-only diff, no behavior change (PR #107, f3d2fb6).

## [1.0.0rc33] - 2026-11-26

### Fixed

- **merge_adapter 30s timeout (#104):** `_request` gains a per-call `timeout` kwarg (httpx `USE_CLIENT_DEFAULT` sentinel keeps the pooled default when unset — `request(timeout=None)` would disable timeouts entirely). `merge_adapter` now sends `MERGE_CALL_TIMEOUT = 1800s`, overridable via `FUSION_TRAINER_MERGE_TIMEOUT` (invalid/`<=0` falls back loudly; explicit arg beats env). Live-verified: real merge through the gateway `:11432` completes in one shot where it previously timed out at 30s.
- **Dead timeout param in `_chat_completions_raw`:** accepted `timeout` (default 120s) but never forwarded it to `_request` — the post-merge validation probe rode the 30s pooled default and false-timed-out against a cold-loading model. Now forwarded.

## [1.0.0rc32] - 2026-11-26

### Added

- **Outcome sidecar join (#97):** reader-side join by `rollout_id` between the gateway rollout tap (never writes outcomes, gateway#33 contract) and app sidecars like office-api's `rollout-outcomes.jsonl`. `load_outcome_sidecar` + `join_outcomes` in `rollout_ingest` (inline outcomes always win, malformed lines skipped loudly), `ft datasets from-rollouts --outcomes <sidecar>`, dashboard `rollout_section` applies the same join (sidecar records carry `outcome` and no `source` — never counted as separate trajectories). Live-verified on real records: SFT positive set produced where ingest previously errored `no positive rollouts`; dashboard total counts tap-only with joined signals. `docs/rollout-tap.md` outcome-backfill row names the join owner.

### Fixed

- **X-Fusion-Source declaration (#100):** `MLXFineTuneClient` declares `X-Fusion-Source: model-hub` by default (`FUSION_MLX_SOURCE` env override, explicit arg wins) — model-management routes through the gateway (`/v1/merge-adapter`, `/v1/convert/jobs*`, proxied since fusion-gateway#721) 403 without the declaration. `merge_adapter` normalizes dash-form catalog ids to slash repo ids (hf rejects `--` in repo_id).
- **Bridge sink defaults (#98, PR #101):** `trainer-bridge.sh` defaults `FUSION_TRAINER_ROLLOUT_SINKS` to the gateway rollout-tap + office outcome files so the dashboard trajectory panel survives supervisor-managed restarts.

## [1.0.0rc31] - 2026-11-25

### Added

- **Flywheel auto-record (#95):** `ft flywheel watch --dir <fusion-eval-results> [--gate] [--once] [--checkpoint <adapter>]` polls for `results/<run>/summary.json` and records a gated flywheel round per unconsumed summary — closes the manual-recording leg of the publish -> deploy -> re-record loop. Consumption state (path + sha256) in `{cache}/flywheel_watch.json`: re-runs never double-record, rewritten summaries record a new round, malformed summaries stay unconsumed and retry loudly. `--checkpoint` passes the loop-invariant adapter path through to `record_round`. 7 tests + walkthrough `docs/guides/flywheel-loop.md`.
- **Loop invariant live-verified (#95):** round checkpoint == published adapter == fusion-mlx served model id (`sft_1795605347026_00eee724-20261125-191812`), confirmed by chat-completion probe and dashboard model pill; gate verdict (including discards) visible on `/dashboard`.

### Fixed

- **Zero train_loss events in bridge run records:** the engine job dict (`to_dict()`) carries no `events` array (SSE stream only), so `wait_for_job`'s events-diff never fired. `_JobEventCursor` now synthesizes per-step events from the `progress` snapshot (`step`/`train_loss`) when `events` is absent and prefers the real array when present. Bridge run `events.jsonl` now streams per-step loss (verified live: 3/3 steps, loss 2.056).

## [1.0.0rc30] - 2026-11-25

### Added

- **Training-loop dashboard (#93):** `GET /dashboard` page + `GET /api/dashboard` payload on the trainer-bridge, per prototype 06 — flywheel round progression chart with gate-discard warning, incumbent eval pass rate, per-task-type breakdown, recent runs table, rollout-sink data mix (`FUSION_TRAINER_ROLLOUT_SINKS`), next-step card. Live data only: every section returns explicit `{available, reason}` and missing data renders an honest empty state. Reuses the trainer-console theme (dark mode + shell theme follow) and bearer auth. 14 tests.

## [1.0.0rc29] - 2026-11-25

Pluggable reward functions for GRPO (#92, in-repo portion of #90): office
validators + rubric judge beyond the built-in math/code sandboxes.

### Added
- **Reward plugin registry** (`rlsl/reward_plugins.py`): sample-level
  `"reward": {"plugin": "...", "args": {...}}` spec dispatched from
  `sandbox.reward_completion` ahead of the auto math/code pick.
- **Office validators** (deterministic, stdlib zip+XML, no new deps):
  `office_xlsx` (required sheets, `formula_policy: require|strict`),
  `office_docx` (required sections), `office_pptx` (min slides, required
  titles). Result file from `args.path_regex` or the last path-like token
  in the completion; missing file / corrupt zip -> 0.0.
- **Rubric LLM judge**: deterministic rubric prompt over MLXClient HTTP
  (`args.model` or `FUSION_RUBRIC_JUDGE_MODEL`), JSON total clamped to
  [0,1]; failure -> 0.0 + loud warning.
- **Dispatch safety**: unknown plugin / plugin exception -> 0.0
  fail-visible, scores clamped to [0,1].

### Notes
- Office validators are deterministic by construction (issue #90
  acceptance); the rubric judge is model-scored by design and documented
  as non-deterministic.
- Upstream gap: fusion-office ships TS web apps, no Python office-api —
  validators here are trainer-side; issue filed on fusion-office.
- Live GRPO acceptance (non-degenerate reward distribution on real office
  trajectories) remains open on #90.
- Tests: +19 (`tests/test_reward_plugins.py`); suite 790 passed, 15
  skipped.

## [1.0.0rc28] - 2026-11-25

Tool-calling trajectory support (#91, closes #88): agent trajectories with
function calls (e.g. office-agent runs captured via the gateway rollout
tap) are now trainable end to end.

### Added
- **Message schema (#88):** assistant turns may carry
  `tool_calls=[{id, type, function:{name, arguments}}]`; `role: "tool"`
  turns carry `tool_call_id`; `content` may be a list of typed parts.
  `formats.copy_message` preserves the tool fields on every round-trip
  (they were silently amputated to `{"role","content"}` before).
- **`serialize_chatml`:** deterministic flatten for model templates
  without tool grammar — tool calls become sorted-key
  `TOOL_CALL {"arguments": ..., "name": ...}` lines, tool results become
  `TOOL_RESULT[call_id]` lines.
- **Sample-level `tools`:** forwarded verbatim to
  `tokenizer.apply_chat_template(messages, tools=...)` by fusion-mlx
  (mlx_lm.tuner `ChatDataset`); templates with tool grammar
  (Llama-3.x, Qwen2.5+) render it.
- **Ingest entry points:** `ft datasets from-rollouts` and `from-eval`
  keep tool fields; DPO pair mining renders tool-call turns as
  `TOOL_CALL` lines so chosen/rejected text is never emptied by
  tool-only turns.
- **Docs:** `docs/tool-trajectories.md` (EN/CN) — schema, sample shape,
  training path, ingest commands.

### Behavior
- Invalid tool structure (orphan `tool` turn without `tool_call_id`,
  tool call missing `function.name`/`arguments`) is dropped with a loud
  warning — never silently mutated. Text-only datasets are unaffected.
- No upstream gap: mlx_lm.tuner `ChatDataset` already passes whole
  message dicts and sample-level `tools` to `apply_chat_template`.

### Tests
- New `tests/test_tool_trajectories.py` (14 cases): validation matrix,
  serialize determinism, loader round-trip/trim/drop, ingest
  preservation, DPO mining. Suite: 771 passed, 15 skipped.

## [1.0.0rc27] - 2026-11-25

CI-repair patch release: the memory admission gate shipped in rc26 made the
nightly real-training suite structurally unpassable on the 7GB runner, and
#87 introduced a ruff lint failure on the master push CI.

### Fixed
- **Memory-guard CI opt-out (#89):** `memory_guard.precheck` honors
  `FUSION_TRAINER_MEMORY_GUARD=off` (logged loudly) — the 4GB
  runtime-overhead constant puts the estimate floor at ~4.4GB while the
  7GB macOS runner leaves ~2.7GB available with the engine loaded, so the
  real RLSL closed-loop test failed the gate regardless of model size.
  `nightly.yml` sets the opt-out for the real-training job only; the gate
  stays active for operator machines.
- **ruff SIM105 (#89):** `grpo.py` rollout token counter replaced
  `try/except pass` with `contextlib.suppress` (master lint job red since
  #87).

### Tests
- New `test_precheck_env_off_bypasses_gate` (bypass + non-off values keep
  the gate active). Suite: 756 passed, 15 skipped.

## [1.0.0rc26] - 2026-11-12

Capability-plan implementation release: RL resume, job queue/mutex, memory
admission gate, dataset lineage + cowork ingestion, reward-hacking monitors,
bench-as-GRPO-verifier reward backend, cross-run comparison, distributed
orchestration surface, rollout throughput metric. Branch
`plan/evolution-2026-11` (plan doc:
`docs/superpowers/plans/2026-11-12-capability-plan.md`).

### Added (capability plan 2026-11)
- **T1 — RL run resume** (#plan 1.1): `ft rlsl --resume` resolves the newest
  numbered checkpoint under `~/.fusion-mlx/adapters/<model>/<run_name>/`,
  stages it to `{output_dir}/resume_stage/adapters.safetensors`, and sets
  `RLSLConfig.init_adapter_path` (DPO/ORPO; GRPO warns loudly and starts
  fresh — no upstream resume field). Engine-restart detection: a lost job
  (404) or upstream-cancelled job (409) during alignment yields an
  `interrupted` result with `resume_checkpoint`/`resume_step`/`resume_hint`
  instead of a zombie failure. Tests: `tests/test_rlsl_resume.py` (8 cases).
- **T2 — cross-process run lock + priority queue** (#plan 1.2): `RunLock`
  (fcntl flock on `{cache_path}/run.lock`, kernel-released on process death)
  serializes hosts (agent-studio in-process / bridge / cowork all converge
  on one lock); in-process pending queue (priority + FIFO) with `queued`
  status + `queue_position` in status/info. Tests: lock contention, queued
  stop, info queue surface.
- **§1.3 — unified-memory admission gate**: `memory_guard.estimate_training_gb`
  (deterministic: params × quant factor + batch×seq×params activation,
  checkpointing 0.30×) + `precheck` (two constraints: psutil available ≥
  estimate AND estimate ≤ budget) wired into SFT + RLSL before any job
  submission — fail-visible with de-escalation suggestions
  (`--grad-checkpoint` / `--max-seq-length` / 8-bit / free resident models),
  never auto-mutates config. Tests: `tests/test_memory_guard.py` (10 cases).
- **T3 — cross-run comparison** (#plan 2.1): `RunManager.compare_runs` +
  bridge `GET /runs/compare?run_ids=a,b` — downsampled metric series
  (train_loss/val_loss/mean_reward) from events.jsonl, compact per-run
  config summary recorded at start, field-level diff; trainer.html section 5
  renders an SVG curve overlay + config diff table (mismatches highlighted).
  Tests: `tests/test_run_manager.py` (5 cases) + `bridge/test_server.py`
  (3 cases).
- **T4 — dataset lineage** (#plan 2.2 + flywheel entry 3): unified lineage
  records (`source`/`source_path`/`source_sha256`/`transforms`/`output`) on
  the from-eval/from-rollouts/from-cowork sidecar manifests;
  `ft datasets info <path>` prints the full chain; `ft datasets from-cowork`
  ingests cowork DAG flow exports (eval-trajectory shape, retry siblings
  share task_id) reusing the eval DPO miner.
- **T7 — reward-hacking monitoring** (#plan 2.3): `hacking_monitor.py` —
  three deterministic detectors over the reward stream (Rule 5, no model
  routing): reward drift (last-window mean shift vs FIRST-window std,
  z ≥ 3 + abs drift ≥ 0.15), length inflation (last/first window mean ratio
  ≥ 1.5), format-token gaming (`<answer>`-family share drift ≥ 0.5 with
  last share ≥ 0.9). `HackingMonitor.observe()` fed by RewardServer batches
  and the RFT scoring loop; snapshot lands in the GRPO result
  (`hacking_metrics`) and in metrics.prom via `prom_series()`. Offline twin
  `compute_hacking_metrics(events)` reduces any events.jsonl. Tests:
  `tests/test_hacking_monitor.py` (10 cases, injected drift triggers each
  alarm; flat series zero false positives).
- **Bench-as-GRPO-verifier** (#plan 3.1): `sandbox_mode="bench"` —
  `RewardServer` scores each completion through fusion-bench's task API
  (per-(task, completion) cache; sample must carry `bench_task`), any bench
  failure degrades the whole batch to the sandbox reward with a loud error +
  `bench_degraded=True` in the result (fail-visible, same semantics as
  eval_loop #75). `RewardServer(mode="bench")` without a client refuses to
  start. Tests: `tests/test_bench_reward.py` (7 cases, real HTTP callback).
- **T5 — distributed orchestration surface** (#plan 2.4, orchestration only):
  `distributed.py` — node manifest validation (`load_nodes`), deterministic
  round-robin shard plan with per-shard sha256 (`plan_shards` →
  `shard_plan.json`), aggregated progress with worst-status-wins
  (`aggregate_progress`). Training execution stays upstream (fusion-mlx
  mx.distributed job API — filed fusion-mlx#1566). Tests:
  `tests/test_distributed.py` (12 cases).
- **T9 — rollout throughput metric** (#plan 3.2): GRPO RFT loop logs
  `rollout_samples_per_second` + `rollout_seconds` per iteration into
  events.jsonl/metrics.prom — quantifies fusion-mlx prefix-cache /
  continuous-batching gains on the sampling path.

### Upstream issues filed (U1-U6)
- fusion-mlx#1564 — RL job numbered checkpoints + init_adapter_path resume
- fusion-mlx#1565 — memory_limit_gb generalized to all fine-tune endpoints
- fusion-models-hub#100 — dataset version/lineage registration API
- fusion-mlx#1566 — distributed fine-tune job API
- fusion-bench#44 — programmatic per-item scoring endpoint
- fusion-mlx#1567 — VLMFineTuneConfig quantize fields

### Fixed
- `bridge/test_shell_registry.py` clean_env fixture now deletes
  FUSION_CLIENT_KEY/FUSION_MLX_API_KEY/MLX_API_KEY — a developer shell with
  FUSION_CLIENT_KEY exported leaked into `accepted_bridge_keys()`
  expectations via fusion_core.config.client_key().

### Tests
752 tests + 47 bridge tests green, ruff clean, docs validator 0 ERROR.

## [1.0.0rc25] - 2026-11-10

Patch release: rollout-sink ingestion (#86) + a test-suite fix for the
macOS system-proxy environment.

### Added (#86)
- **Rollout tap design doc** — `docs/rollout-tap.md`: adopts the
  microsoft/agent-lightning v1.0 proxy-side rollout-collection pattern onto
  the fusion topology. Seams split by repo ownership: tap middleware + sink
  belong to fusion-gateway (tracked upstream, fusion-gateway#608); the
  trainer-side consumption ships here. Covers the rollout record schema v1,
  the opt-in matrix (tenant/key/routes/PII gate, redact by default), the
  deterministic reward-signal mapping (task_success/user_accepted 1.0,
  user_edited 0.5, user_rejected/error 0.0, unobserved excluded), the
  reward-hacking countermeasures, and the #86 acceptance mapping.
- **`ft datasets from-rollouts`** (`dataset/rollout_ingest.py`) — consumes a
  gateway rollout-tap sink JSONL into trainer datasets: positive rollouts
  (reward >= `--min-score`, default 1.0) become `{"messages": [...]}` SFT
  samples via the existing `ft sft` path; every prompted record becomes a
  GRPO prompt (`{"prompt", "reward", "metadata"}`, prompt falls back to the
  first user turn) for `ft rlsl --method grpo`. Bad lines skipped loudly,
  zero-record sinks raise fail-visible, prompts without outcomes can never
  enter the positive SFT set. Sidecar manifest `<output>.from_rollouts.json`
  (atomic) carries the sink sha256 + per-signal counts.
- **Rollout-fed run traceability** — `SFTTrainer._find_eval_manifest` now
  falls through to the from-rollouts sidecar, so `ft sft` records
  `source_export_sha256`/`source_export_path` for rollout-fed runs too
  (same mechanism as eval-fed runs, #76 requirement 3).
- Tests: `tests/test_rollout_ingest.py` (19 cases, fully offline).

### Fixed
- **Loopback tests vs macOS system proxy** — `tests/test_grpo.py` +
  `tests/test_load_stability.py` real-HTTP call sites now pass
  `trust_env=False`: with a system proxy enabled (e.g. 127.0.0.1:7897),
  httpx picks it up via `getproxies()`/`_scproxy` and 502s loopback calls to
  the in-process reward servers. Environment change, not a code regression
  (CI unaffected — no system proxy on runners).


All notable changes to fusion-trainer are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are tracked by the `__version__` in `fusion_trainer/_version.py` and
by git tags of the form `v<version>`.

## [1.0.0rc24] - 2026-11-07

Docs release: adopted the fusion docs-system v1.1 contract (platform x
service, PR #84) — added ci/validate_docs.py + docs workflow gating,
Tier 0 docs (product-page / get-started / architecture / api-reference /
troubleshooting), contract frontmatter on first-party markdown, and
frontmatter for the archived seatbelt-sandbox plan/spec (status:
deprecated, superseded_by the grpo-sandbox-reward guide). ruff UP
modernization of the validator itself. No frozen-surface change.

## [1.0.0rc23] - 2026-11-01

Patch release: multi-arm DPO flywheel round (#83) — the DPO lever from #82's
re-scope, plus the DPO memory-safety passthroughs it took to run DPO on a
27B-8bit base at all.

### Added (#83)
- **Multi-arm DPO mining → training → eval gate** — merged the 4 existing
  30-task eval exports (structurally multi-arm: 120 trajectories) into one
  JSONL, mined 22 `chosen`/`rejected` pairs (chosen lineage 19/22 from the
  incumbent fw-round1 adapter — true on-policy preference data; 30 unpaired
  failures counted loudly, never fabricated; 6 never-passed tasks excluded),
  DPO-trained chained on the incumbent fw-round1 (73.33%), eval-gated via
  `ft flywheel record --gate`. **Round-5 result (run `20261101-r5-eval`):
  17/30 = 56.67% — gate verdict DISCARDED (−16.67pp vs incumbent), checkpoint
  not promoted.** Honest negative: 22 pairs from 4 exports is too thin a
  preference signal to beat the incumbent — `testwrite` regressed 2/3→0/3
  (agent wrote no runnable tests; hidden mutations all survived) while
  `configfix`/`depfix`/`bugfix` held 3/3 and `feature` stayed 0/3 (persistent
  spec-gap). Failure modes are spec-gap/implementation-miss, one 900s timeout.
  Multi-arm DPO scores above every discarded SFT chain round (43.33%/46.67%)
  and preserves more of round-1's per-type strength, but the flywheel remains
  a one-high-signal-round tool until a larger multi-arm export corpus exists.
- **`RLSLConfig.gradient_checkpointing`** (DPO/ORPO passthrough) — the DPO
  loss graph holds the chosen AND rejected forwards simultaneously (~2x an
  SFT step's activation memory); 27B-8bit + seq 1024 OOM'd the engine's
  memory_enforcer without recompute (live-verified). Maps to fusion-mlx
  `DPOConfig.gradient_checkpointing` (#1460). Key sent only when True.
- **`RLSLConfig.keep_ref_resident`** (DPO passthrough) — fusion-mlx's per-pair
  reference load-and-evict performs TWO full base loads per pair (chosen +
  rejected); on a 27B-8bit base the repeated same-size Metal allocations
  churned phys_footprint to 165.9GB over the 120GB ceiling and the enforcer
  fatal-exited the engine ~90s after job start (fusion-mlx #1472, fixed
  upstream v0.10.36 — the engine now also auto-enables a resident reference
  for param_bytes >= 16GB). Key sent only when True.

### Fixed (#83)
- **`ft rlsl` dropped YAML alignment hyperparams** — `num_epochs`, `beta`,
  `batch_size`, `learning_rate`, `log_every`, `save_every`, `output_dir` from
  the YAML `rlsl:` section (and presets) never reached `RLSLConfig`; the
  dataclass defaults ran silently (live-observed: YAML `num_epochs: 2`
  trained 1 epoch — iters 22, not 44; `beta`/`lr`/`batch` matched only by
  coincidence). Resolution order now mirrors bnup-train: preset > YAML >
  dataclass default.
- **`--dpo-output` single-arm gate message** (#78 residual) — the remediation
  text referenced a nonexistent `--arm` CLI form; now says to merge several
  single-arm exports into one JSONL (or wait for fusion-eval#17 multi-arm).

### Upstream (fusion-mlx)
- #1461 DPO `init_adapter_path` double-apply ("Can't convert layer of type
  LoRALinear to LoRA") — fixed, v0.10.34.
- #1463 in-process trainers never called `model.train()` — hybrid
  linear-attention models (qwen3_5/qwen3_next, incl. Qwen3.8-27B-8bit) routed
  the backward through a no-vjp CustomKernel; first backward raised
  "[Primitive::vjp] Not implemented for CustomKernel" — fixed, v0.10.35.
- #1472 DPO per-pair ref load-and-evict Metal churn (this repo's round-5
  live evidence) — `DPOConfig.keep_ref_resident` + auto-enable, v0.10.36.
- #1473 fail-closed training memory preflight — v0.10.38.

## [1.0.0rc22] - 2026-10-31

Patch release: flywheel discipline (#82) — fresh-data enforcement and
eval-gated promotion.

### Added (#82)
- **Fresh-data discipline** — `record_round` tracks which export sha each
  training round consumed (`consumed` field on entries; historical entries
  without the field count as consumed unless labelled baseline). Recording a
  round whose export was already consumed by another round raises
  `FlywheelError` (exit 1 from the CLI); `--allow-reconsume` overrides with a
  loud warning and flags the entry `reconsumed` — never silent.
- **Eval-gated promotion** — `ft flywheel record --gate` compares the
  candidate's pass_rate against the incumbent (best prior non-discarded
  round; historical entries without a status field count as promoted).
  Regression marks the round `discarded` — the checkpoint is NOT promoted,
  and the CLI prints the verdict with the regression delta. `ft flywheel
  mark --label X --status promoted|discarded` backfills status for rounds
  recorded before the gate existed.
- **Curve rendering** — `ft flywheel status` gains a status column, a
  promoted-only trend line (discarded rounds excluded from the first→last
  acceptance delta), a discarded-rounds summary, and a `*` marker for
  re-consumed exports.
- **Registry data correction** — historical rounds fw-round1/2/3 had
  `export_sha256` pointing at each round's *own post-training eval export*
  (the data the *next* round trains on) instead of the export the round
  actually trained on. Corrected from the `from_eval.json` manifest sidecars:
  fw-round1 ← baseline-eval (`c1ba5557…`), fw-round2 ← round-1-eval
  (`7ce288f7…`), fw-round3 ← round-2-eval (`a54a6ce9…`). With the true
  consumed set in place, the round-3-eval export (`504298d3…`, 14 passed)
  is genuinely fresh training data for round-4. Backup at
  `flywheel.json.pre-82-fix`.

### Live evidence (#82, 2026-10-31, Qwen3.8-27B-8bit, 30-task set)
- **Round-4 trained and gated live.** Fresh data (14 passed samples from the
  round-3 eval export, sha `504298d3…` — verified not in the consumed set),
  chained from the *incumbent* fw-round1 adapter (LoRA rank 16, 39 iters,
  `--grad-checkpoint`). Result: 13/30 = **43.33%** (eval run
  `20261031-r4-eval2`), a −30.00pp regression vs the incumbent 73.33% — the
  gate **discarded** the round and the checkpoint was NOT promoted.
- **Flywheel claim re-scoped (honest negative result).** Chaining SFT from
  the *best* adapter on *fresh* failure data still regresses (43.33% <
  53.33% baseline). Across rounds 2/3/4, every chained round degraded; the
  flywheel's measured value is the **first high-signal round on failure
  data** (+20pp, fw-round1), not repeated chaining. Repeated SFT chaining on
  eval-fed data does not compound — cumulative over-production, not
  fixable by data freshness or a better starting checkpoint. The gate now
  enforces this mechanically: regressing rounds are discarded, only the
  first round's +20pp stands as promoted.
- **Comparability caveat:** the 30 task ids are identical to round-3's, but
  fusion-eval's task yaml files were touched upstream after 2026-10-28
  (round-4 `task_set_sha` `bbb519a1…` vs round-3 `5151e824…`); the
  regression margin (−30pp) is far larger than any plausible task-set
  drift.
- **Per-type:** bugfix 1.0 held; depfix recovered 0.67→1.0;
  long-context 1.0 recovered; refactor-multi 0.67 partial recovery;
  testwrite collapsed again 0.67→0.0 (over-production signature);
  multi-turn-repair 1.0→0.0; configfix and feature stay 0.0 (persistent
  spec-gap classes, unchanged since #78).
- **Ops notes:** fusion-eval auth broke mid-cycle — a stale
  `FUSION_CLIENT_KEY` in the shell profile loses to the current engine admin
  key; the eval script now pins `FUSION_CLIENT_KEY` from
  `~/.fusion-mlx/settings.json` and pins the 30-task id list explicitly
  (the fusion-eval task dir grew 30→60 since round-3; `--task all` would
  have produced an incomparable 60-task score). fusion-eval's instant-fail
  recovery (3×120s) absorbed 8 transient engine stalls during the run.

## [1.0.0rc21] - 2026-10-31

Patch release: dependency-metadata hygiene (#81).

### Fixed (#81)
- `pyproject.toml` pinned fusion-core as a direct URL reference
  (`fusion-core @ git+https://github.com/dahai80/fusion-core.git@v0.7.0`).
  fusion-core is not on PyPI; in the shared workspace venv (editable 0.10.0
  via `pip install -e fusion-core`), `uv pip install -e fusion-trainer`
  treated the exact git pin as a different distribution and silently
  replaced the editable install with a stale non-editable v0.7.0 — breaking
  downstream consumers that need `>=0.8.0` (fusion-agent-studio) and
  reintroducing fixed-in-0.8/0.9 bugs (e.g. missing `service_port`
  config-key prefix normalization). Replaced with the version-only
  constraint `fusion-core>=0.9.0` — sources resolve per environment:
  workspace venv = editable install (source == lock), CI = explicit
  `git+...@v0.10.0` install step added to both `ci.yml` and `nightly.yml`.
  Verified: after the fix, `uv pip install -e fusion-trainer` leaves
  fusion-core editable at 0.10.0 untouched.

## [1.0.0rc20] - 2026-10-28

Patch release: process-title remediation (#77), flywheel round-2 residual
closure (#78) — single-arm DPO gate, feature-type failure-mode analysis,
and the round-3 fresh-data experiment — plus dependency hygiene (#79) and
fusion_core.config single-point env resolution (#80).

### Changed
- **#80** — env/config resolution adopted from `fusion_core.config` v0.7.0
  (pin bumped v0.5.0 → v0.7.0). `MLXFineTuneClient` resolves the engine
  token via `resolve_api_key(env_var="FUSION_MLX_ADMIN_TOKEN")` — env first,
  then `~/.fusion-mlx/settings.json` `auth.api_key` (the fallback
  `trainer-bridge.sh` always did by hand, now in-process). `TrainerConfig`
  port/hub defaults via `service_port`/`get_env` (same env names:
  `FUSION_MLX_PORT`, `FUSION_MODEL_HUB_URL`, `FUSION_MODEL_HUB_API_KEY` —
  no behavior change). Shell registration resolves its gateway key via
  `fusion_core.config.admin_key`, and the legacy `FUSION_GATEWAY_ADMIN_KEY`
  env name is renamed `FUSION_ADMIN_KEY` everywhere (scripts, docs, tests);
  a prod-profile missing key disables registration with a loud warning —
  never fatal to the bridge. `tests/test_mlx_client.py` auth tests made
  hermetic (HOME → tmp_path + `_SETTINGS_CACHE` cleared) plus a new
  settings.json-fallback test. Workspace `check-env.py`: zero findings in
  fusion-trainer source.

### Dependencies
- **#79** — vision extras: `opencv-python>=4.8.0` →
  `opencv-python-headless>=4.10.0`. No `cv2` imports exist anywhere in
  trainer code — pure dependency swap; the headless build does not pull GUI
  frameworks (wx/qt) onto a headless server Mac.

### Fixed
- **#77** — bridge server process shows as `fusion-trainer bridge :8790` in
  ps/Activity Monitor (was bare homebrew `Python`). `setproctitle` (new core
  dependency, try/except fallback) is set in `bridge/server.py __main__`; the
  fallback logs a warning and keeps the default title if the package is
  missing. Launcher pid/pkill/health logic unaffected (verified live).
- **#78** — `ft datasets from-eval --dpo-output` on a single-arm export (one
  trajectory per task — every `fusion-eval run --arm <one>` export) now fails
  loudly with an explanation instead of silently writing a 0-pair file:
  mining is structurally empty without passed/failed siblings. Escape hatch:
  `--dpo-anyway` (warns + writes the empty file). The from-eval manifest
  gains `arms` and `dpo_structurally_empty` so downstream tooling sees WHY a
  pair file is empty. `--dpo-output` help documents the multi-arm requirement.

### Analyzed (#78 feature type 0/3 across rounds)
- Round-1 feature failures are spec-gap failures, not code failures: t3
  wrote only the test file (never implemented `slugify`), t5 implemented
  `to_roman` correctly for 1-3999 but missed hidden out-of-range/type checks
  (`to_roman(0)` must raise).
- Round-2 feature failures are all 900s timeouts with CORRECT diffs (t3
  slugify, t5 roman implementations both correct; the agent loop over-iterates
  on the write-tests/run-pytest loop and never terminates) — the same
  chained-overfit signature as the testwrite 3/3→0/3 regression.
- Round-3 (fresh round-2-eval data, 19 samples never trained on, chained from
  fw-round2) ran 2026-10-28: 14/30 = 46.67% pass (eval run `20261028-r3-eval`,
  export sha `504298d3986c`). Curve: baseline 53.33% → fw-round1 73.33% →
  fw-round2 63.33% → fw-round3 46.67% (first→last −6.67pp DOWN). Honest negative
  result — recorded in the curve registry via `ft flywheel record`.
  - **Hypothesis refuted in part:** testwrite RECOVERED 0/3 → 2/3 (t21, t22
    pass; t20 agent tests miss one mutation) — the round-2 testwrite collapse
    was overfit-driven, not capability loss. But feature stays 0/3 across all
    rounds (t3 still never implements `slugify`, t5 still misses hidden
    out-of-range checks, t10 emits `BoundedStack` taking no arguments), so
    feature failure is a persistent spec-gap, not overfit.
  - **New regressions from fresh-data chaining:** refactor-multi 1.0 → 0.0
    (t14 IndentationError, t15 forgot to create `user.py`, t16 infinite
    recursion), configfix 1.0 → 0.0 (t17/t18 900s timeouts, t19 missed a
    key-rename), long-context 1/1 → 0/1 (one spec violation left), depfix
    1.0 → 0.6667. dag improved 0.6 → 0.8. Conclusion: degradation is NOT
    purely overfit to trained data — cumulative over-production from chained
    SFT rounds degrades multi-file/config tasks on fresh data too. Stop
    chaining; the flywheel's value (round-1 +20pp) came from the first
    high-signal round on failure data, not from repeated chaining.

## [1.0.0rc19] - 2026-10-28

Eval flywheel (#76, items 11+12): consume fusion-eval#6 trajectory exports,
RLVR-style failure mining, pass-rate-vs-rounds curve registry — plus the
`gradient_checkpointing` passthrough that made 27B-8bit training machine-safe.

### Flywheel evidence (real 27B-8bit rounds, Qwen3.8-27B-8bit, 30-task set)

| round | pass_rate | delta | eval run | export sha |
|---|---|---|---|---|
| baseline | 53.33% | — | 20261026-151812 | c1ba555798a6 |
| fw-round1 | 73.33% | +20.00pp | 20261027-r1-eval | e529c0b7d43c |
| fw-round2 | 63.33% | −10.00pp | 20261027-r2-eval | a54a6ce962fa |

Trend first→last +10pp (UP). Zero-type flips held at round-2:
`long-context` 0/1→1/1 and `multi-turn-repair` 0/1→1/1 (acceptance:
at least one baseline-zero type non-zero after one round — exceeded).
Round-2 regression isolated to `testwrite` (3/3→0/3, all three hits of the
900s timeout — the chained adapter over-produces on test-writing tasks;
over-fitting signal from chaining two SFT rounds on a 22-sample eval-fed
dataset). `feature` stayed 0/3 across all rounds. Recorded honestly via
`ft flywheel record`; `ft flywheel status` prints the curve.

### Added
- `ft datasets from-eval <export.jsonl>` builds an SFT dataset from passed
  trajectories (messages schema, multi-turn preserved) with a source manifest
  (`<name>.from_eval.json`) carrying the export sha256 (#76).
- RLVR-style failure mining: `--all` exports mine near-miss failures into DPO
  preference pairs for round-2 alignment data (#76).
- Eval curve registry: `ft flywheel record` maps a training round to its
  fusion-eval run (pass_rate + per-type breakdown); `ft flywheel status`
  prints the pass-rate-vs-rounds curve with export-hash traceability (#76).
- `gradient_checkpointing` field on `SFTConfig` + `--grad-checkpoint` CLI
  flag, passed through to fusion-mlx `FineTuneConfig.gradient_checkpointing`
  (LM) / `grad_checkpoint` (VLM). Key sent only when on (older fusion-mlx
  400s on unknown keys). Required for LoRA on very large 8-bit models:
  27B-8bit + seq 4096 without it exhausted unified memory and panicked macOS
  (3 kernel panics on 2026-10-26).
- `--init-adapter-path` on `ft sft` (mirrors `ft rlsl`): initialize a fresh
  LoRA from a prior round's adapter dir (final weights or newest numbered
  checkpoint, staged via the #65 `_resolve_resume` path) — flywheel round
  chaining without fabricating a checkpoint (#76).

### Fixed
- `load_messages` over-budget fallback: when the trim loop kept a trailing
  assistant turn but dropped every user turn (or kept nothing), the sample
  went to fusion-mlx as a lone-assistant conversation and the whole job died
  with `TemplateError: No user query found in messages` (seen live on the #76
  flywheel round-1 run). The fallback now always re-prepends the last user
  query (hard-sliced), keeping the sample template-valid.

## [1.0.0rc18] - 2026-10-26

Consumes fusion-gateway#316 (hub `/api/v1/models` proxy route) and completes
the live `--publish` verification that closed #74.

### Changed
- **Hub publish through the gateway** —
  [fusion-gateway#316](https://github.com/dahai80/fusion-gateway/issues/316)
  (closed) adds the `/api/v1/models` reverse proxy to fusion-model-hub, so
  `hub_url=http://localhost:11432` (the default) works with a single gateway
  key — no separate hub key. Verified live: `publish_voice_package` 201
  (model_id `3e839f61e8974e06`, `model_type=audio`), 409-idempotent on
  re-publish, entry listed through the gateway. Stale "gateway does not proxy
  /api/v1/models, point at :11444" guidance removed from README/README_CN/
  CLAUDE.md (hub-direct remains a documented fallback).
- **#74 closed** — voice `--publish` verified end-to-end through the gateway;
  `voice_ref` synthesis 200 with a real WAV on the published package.

## [1.0.0rc17] - 2026-10-26

Consumes two long-closed upstream fixes whose trainer docs/comments had gone
stale; verified live against the local gateway + engine before writing.

### Changed
- **Gateway fine-tune proxy acknowledged** — fusion-gateway v0.8.4
  ([fusion-gateway#30](https://github.com/dahai80/fusion-gateway/issues/30),
  closed) proxies `/admin/api/fine-tune/*` 1:1 through `:11432`, including
  `/jobs/models` ([fusion-mlx#397](https://github.com/dahai80/fusion-mlx/issues/397),
  closed). Verified live: jobs/adapters/models all return 200 through
  `:11432`. The `mlx_port` default stays `11434` (one hop less, trainer's own
  admin key); `FUSION_MLX_PORT=11432` now genuinely routes fine-tune traffic
  through the gateway. Stale "gateway proxies only /v1*" claims removed from
  `config.py` comments and README.
- **Adapter registry slash-id fix acknowledged** —
  [fusion-mlx#1317](https://github.com/dahai80/fusion-mlx/issues/1317) (PR
  #1327): the registry walks 3 levels and lists slash-id adapters (verified:
  `mlx-community/Qwen3-0.6B-8bit | sft1205-a` visible). The bridge still
  deliberately publishes merged standalone models rather than hot-mounting
  raw adapters — but the docs now say so as a design choice, not an upstream
  blocker.

## [1.0.0rc16] - 2026-10-26

Consumes the three upstream fixes filed during #75 (all closed 2026-10-25).

### Added
- **Sequence packing** — `ft sft --pack-sequences` (or YAML `pack_sequences:
  true`): short samples are concatenated into rows of at most
  `max_seq_length` tokens with a block-diagonal attention mask (no
  cross-sample attention) and a per-token loss mask that still honours
  `mask_prompt` offsets. Consumes the upstream `FineTuneConfig.
  pack_sequences` field ([fusion-mlx#1400](https://github.com/dahai80/fusion-mlx/issues/1400)
  closed, fusion-mlx >= 0.10.17). The key is only sent when explicitly on —
  older fusion-mlx 400s on the unknown field. Off by default: packing
  changes batch composition, so results are not byte-comparable with
  unpacked runs. Also wired into `ft bnup-train` via `_pick`. Live-verified
  end-to-end on Llama-3.2-1B-Instruct-4bit against fusion-mlx 0.10.17
  (job record carries `pack_sequences: true`, status completed, adapter
  written).

### Changed
- Nightly CI installs base (slim) `fusion-mlx>=0.10.17` instead of
  `fusion-mlx[vlm]`: [fusion-mlx#1403](https://github.com/dahai80/fusion-mlx/issues/1403)
  (fixed via #1406) made the turboquant attention patch pass through when
  `mlx_vlm` is absent, so DPO/ORPO/reward jobs no longer die with
  `No module named 'mlx_vlm'` on a slim install. Running slim keeps that
  path regression-covered in CI.
- `eval_loop.py` / README upstream caveat updated:
  [fusion-bench#29](https://github.com/dahai80/fusion-bench/issues/29)
  closed in fusion-bench v0.4.11 — the per-candidate `adapter_path` in
  task params is forwarded to fusion-mlx chat as the `adapters` field, so
  bench scores reflect the true checkpoint weights (bench >= 0.4.11). The
  params form (not the new top-level `adapter_path` field) is sent
  deliberately: older bench releases 400 on unknown top-level keys while
  every release reads `params.adapter_path`.

## [1.0.0rc15] - 2026-10-25

Closes #75: benchmark-based checkpoint selection (eval loop), nightly
real-training CI, GRPO backend honesty, and a chat-loader truncation fix.

### Added
- `fusion_trainer/eval_loop.py` (issue #75) — the training flywheel's
  missing verification step. `BenchClient` talks to fusion-bench's task
  API (`POST /api/v1/tasks`, `GET /api/v1/tasks/{id}`) over pure HTTP
  (tenant header + Bearer; S6 URL guard). `run_checkpoint_eval`
  enumerates every numbered checkpoint (`NNNNNN_adapters.safetensors`,
  same layout as #65 resume), stages each one to a servable
  `adapters.safetensors` dir, submits one bench task per checkpoint, and
  picks the winner deterministically: bench score (higher wins, tie →
  later step) → val_loss (lower wins, tie → later step) → last checkpoint.
  The selection table persists atomically to
  `checkpoint_selection.json` next to the adapters. Fail-visible, never
  raises: a bench outage marks every row `unavailable`, records a
  `fallback_reason`, and selects on val_loss — never a silent
  mis-selection. NOTE: fusion-bench cannot yet evaluate LoRA adapters
  (no `adapters` passthrough to fusion-mlx chat) — filed upstream as
  [fusion-bench#29](https://github.com/dahai80/fusion-bench/issues/29);
  the trainer already sends `params.adapter_path` so the loop scores true
  checkpoint weights the moment bench lands passthrough.
- `ft sft --eval-select` + `--bench-url/--bench-api-key/--bench-tenant-id/
  --bench-executor` — run the checkpoint eval after a completed SFT job
  and attach `checkpoint_selection` to the run result (printed table
  included). YAML/preset support via the `sft:` section
  (`eval_select`, `bench_url`, `bench_api_key`, `bench_tenant_id`,
  `bench_executor`, `bench_suite`, `bench_params`, `bench_task_timeout`).
- `.github/workflows/nightly.yml` — nightly real-training regression run
  (macOS arm64): boots a real fusion-mlx with `Qwen3-0.6B-4bit`, runs
  `TestRealFusionMLXIntegration` + the contract suite with
  `FUSION_TRAINER_REAL=1`. The per-push ubuntu CI cannot install MLX, so
  these ~18 tests were silently skipped on every push; upstream field
  renames / job-lifecycle changes now get caught within a day.
  `workflow_dispatch` trigger for on-demand runs.

### Changed
- **GRPO backend honesty (issue #75):** `RLSLConfig.grpo_backend` now
  accepts `policy_gradient` as an alias of `grpo` (normalized in
  `TrainerConfig.__post_init__`), and the docs state plainly that the
  DEFAULT backend `rft` is a reward-filtered SFT approximation
  (RFT/RAFT-style) — no per-token policy gradient, no importance ratio.
  No behavior change; existing `rft`/`grpo` values work unchanged.

### Fixed
- `DatasetLoader.load_messages` treated `max_seq_length` TOKENS as CHARS:
  every message was individually capped at `[:max_seq_length]` chars, a
  ~4x over-truncation (4096-token budget silently amputated any turn
  over 4096 chars). Now the WHOLE conversation gets a char budget of
  `max_seq_length * 4`, trimming whole trailing turns (never mid-answer,
  so the chat template stays well-formed); exact token-level truncation
  happens upstream at batch assembly as before.

### Upstream
- [fusion-bench#29](https://github.com/dahai80/fusion-bench/issues/29) —
  LoRA adapter passthrough (`adapter_path` → fusion-mlx `adapters`).
- [fusion-mlx#1400](https://github.com/dahai80/fusion-mlx/issues/1400) —
  SFT sequence packing (`FineTuneConfig` field; mlx_lm 0.31.3 has no
  packing to expose yet). fusion-trainer sends nothing packing-related
  until it lands.

## [1.0.0rc14] - 2026-10-25

Closes #74: TTS voice-pack (voice LoRA) training pipeline, wired on
fusion-mlx #1336.

### Added
- `fusion_trainer/dataset/voice_prep.py` — `validate_voice_dataset` accepts
  the three layouts fusion-mlx's voice trainer consumes (`metadata.csv`
  `file|text`, `metadata.jsonl` `{audio,text}`, bare `<name>.wav` +
  `<name>.txt`) and fails loudly on missing audio, truncated files
  (< 1 KiB) or empty transcripts before a job is submitted.
- `fusion_trainer/voice.py` — `VoiceTrainer` + `VoiceTrainConfig` mirroring
  the upstream SPSA field set exactly (`steps`, `learning_rate`,
  `perturbation`, `decay`, `base_voice`, `lang_code`, `max_samples`,
  `dynamics_weight`, `level_weight`, `anchor_weight`, `seed`). Submit →
  poll → terminal; a failed/cancelled job returns its upstream status and
  `error`, never a faked `completed`.
- `MLXFineTuneClient` voice endpoints: `create_voice_job`, `get_voice_job`,
  `cancel_voice_job`, `list_voice_jobs`, `list_voice_backends`,
  `wait_for_voice_job` (A1 retry-on-poll-failure, 404 fast-fail via
  `_raise_if_job_lost`). Upstream reports progress over SSE only — the job
  GET carries no `events` array.
- `ft voice-train` CLI command — `--model/-m`, `--dataset/-d`,
  `--voice-id`, `--steps`, `--base-voice`, `--lang-code`, `--max-samples`,
  `--seed`, `--tenant-id`, `--publish`. Non-completed jobs exit 1.
- `ModelHubClient.publish_voice_package` — registers the trained voice
  package as `model_type=audio` (hub `ModelType.AUDIO`); 409 is idempotent
  on an exact-name match. `--publish` calls it after completion and stays
  non-blocking (a hub outage never fails the training run).
- 24 new tests (`tests/test_voice_trainer.py` + hub header coverage).

### Fixed
- `ModelHubClient` now sends `X-API-Key` alongside `Authorization: Bearer`.
  fusion-model-hub's local auth middleware reads `X-API-Key`; without it a
  direct hub call (no gateway in front) returned 401 even with a valid key.

### Notes
- Voice training requires fusion-mlx with the #1336 endpoint (voice jobs,
  `voice_ref` synthesis, trained-voice listing). Trained packages land in
  `~/.fusion-mlx/voices/<voice_id>/` and are served by fusion-mlx as
  `voice_ref` on `POST /v1/audio/speech` (unknown ref → 404, no silent
  fallback).
- Live `--publish` verification is deployment-gated on this machine: the
  local fusion-gateway has no `/api/v1/models` proxy route and the hub's
  registered key plaintext is unavailable. `publish_voice_package` is
  covered by mocked contract tests; the trainer-side call path is real.

## [1.0.0rc13] - 2026-10-23

Closes #73: the embedded trainer console now follows the shell theme toggle.

### Fixed
- `bridge/trainer.html` — class-based theming (`html.light` / `html.dark`)
  replacing the hardcoded light palette: the embedded iframe subscribes to
  the shell theme event (`{type:"event", event:"theme", payload:...}`)
  and flips palettes without reload; standalone mode reads
  `localStorage["fusion.shell.theme"]` (the shell's stored preference) at
  boot, falling back to `prefers-color-scheme`; a pre-paint head script
  sets the class before first render (no flash of wrong theme). Hardcoded
  `#fff` surfaces moved to a `--fs-surface` token so dark mode holds.
- `bridge/test_server.py` — `test_console_theme_follows_shell` asserts the
  served page wires the theme contract (classes, boot script, event
  handler, token-based surfaces).

### Notes
- #74 (TTS voice-LoRA pipeline) remains open, blocked upstream: fusion-mlx
  audio is inference-only. Filed
  [fusion-mlx#1336](https://github.com/dahai80/fusion-mlx/issues/1336)
  (voice fine-tune job endpoint); trainer-side work follows that contract.

## [1.0.0rc12] - 2026-10-23

Closes #72: the trainer console is now a first-class fusion-gateway shell
module — the shell sidebar lists 训练控制台 and the gateway proxy
(`/shell/fusion-trainer/`) is the single user-facing entry.

### Added
- `bridge/shell_registry.py` — shell module registration + heartbeat
  (same pattern as fusion-cowork / fusion-agent-studio): registers
  `bridge/manifest.json` (id `fusion-trainer`, entry `/shell/fusion-trainer/`,
  upstream `http://127.0.0.1:8790`) with `POST /api/v1/shell/manifests`,
  heartbeats every 15s, re-registers on gateway restart (404) or issued-key
  invalidation (401/403). Env-gated: `FUSION_SHELL_GATEWAY_URL` +
  `FUSION_GATEWAY_ADMIN_KEY` (trainer-bridge.sh defaults the URL to the local
  gateway and reads the admin key from the gateway config when unset).
- K1/K2 gateway-issued `fm-` module key: the registration body carries no
  `api_key`, the gateway issues the key (`raw_key`, returned exactly once),
  the bridge persists it to `~/.fusion-trainer-bridge/web_module_key` (0600)
  and uses it for heartbeat auth. Inbound bridge auth dual-accepts the
  static `FUSION_TRAINER_BRIDGE_KEY` (direct loopback) and the issued `fm-`
  key (gateway proxy injection) — constant-time compare, 401/503 fail-closed.
- `GET /health` reports `shell_registered`; bridge `__main__` configures
  logging so shell-registry INFO lines reach the log file.

### Fixed
- Upstream fusion-gateway#277 (fixed via fusion-gateway#278): the shell
  registry's re-registration keep-branch dropped the issued-key plaintext,
  wiping the proxy credential while heartbeats stayed green — a bridge
  restart left the module reachable but the proxy 401. Live-verified after
  the gateway fix: re-registration retains the key, proxy stays 200.
- `bridge/test_server.py` `_build_config` tests are now self-sufficient
  (create their own dataset) — pytest-randomly shuffles order and the old
  cross-test dependency flaked.

### Verified (live e2e, fusion-gateway 11432 + bridge 8790)
- Sidebar lists 训练控制台 online; clicking opens the four-section GUI in
  the shell iframe at `/shell/fusion-trainer/` with live presets/runs data.
- Bridge stop → entry greys `unreachable` (heartbeat expiry); restart →
  online again, key retained across the re-registration.
- Direct `:8790` access stays loopback-only and 401s without a token.
- `bridge/test_shell_registry.py` (14 cases, mocked HTTP) +
  `bridge/test_server.py` (21) green under random ordering; full suite 551
  passed; ruff clean.

## [1.0.0rc11] - 2026-10-22

Closes #71: Trainer console GUI migrated verbatim from f-ecommerce (migration
study `fusion/insight/f-ecommerce-migration-1022.md`, E4) with business logic
generalized — no e-commerce semantics remain.

### Added

- **trainer-bridge console** (`bridge/`): FastAPI facade (`server.py`) around
  the fusion_trainer library serving the four-section Trainer GUI
  (`trainer.html`: datasets / training / merge & publish / evaluation) at `/`.
  Bridge owns the fusion-mlx admin key + RunManager lifecycle (runs survive
  console refreshes); loopback bind, bearer-token auth on every data route,
  browser never sees the admin key.
- Dataset upload pipeline: CSV (instruction,input,output) or chat JSONL →
  schema detection + validation via `fusion_trainer.dataset.formats`, 50 MB
  cap, name sanitization, list/delete.
- Run lifecycle endpoints (`POST /runs`, list/detail/stop/merge) with one
  active run per machine (409 on second), dataset existence checks, and
  preset registry resolution.
- Merge & publish flow: `POST /runs/{id}/merge` produces a standalone model
  in `~/.fusion-mlx/models` (auto-enters the `/v1/models` catalog);
  `POST /publish` promotes the newest merged model of a completed run into
  the published slot; `POST /publish {"run_id": ""}` reverts to base.
  Hot-mount of raw adapters is not used (engine registry is 2-level; slash
  model ids nest deeper — filed fusion-mlx#1317).
- Evaluation endpoint `POST /eval`: same instruction to base and published
  tuned model, side-by-side answers.
- Preset registry as a config file (`~/.fusion-trainer-bridge/presets.json`,
  seeded from `SFT_PRESETS`, re-read per request — editable without a
  release; corrupt/empty falls back to built-ins with a warning).
- Lifecycle script `scripts/trainer-bridge.sh` (start/stop/status/log; reads
  the engine key from `~/.fusion-mlx/settings.json`, never hardcoded).
- Explicit training-memory budget: `SFTConfig.memory_limit_gb` (default 0 =
  upstream default) passed through to fusion-mlx `FineTuneConfig`; the
  bridge computes 85%-line − resident − 6 GB headroom (floor 8) because the
  upstream default assumes a dedicated training box.
- `MLXFineTuneClient(source=...)` emits `X-Fusion-Source` — merge/serve
  model-management endpoints are gated on it (fusion-mlx auth).

### Fixed

- Tiny-run progress: `steps_per_report` capped to `iters` (a 2-step smoke run
  previously completed with zero progress events, UI stuck at 0/3); completed
  runs report `step == total_steps`; `list_runs`/`run_status` now carry the
  `model` field for the GUI run table.

### Tests

- `bridge/test_server.py`: 21 cases (TestClient, mocked RunManager/MLX
  client) covering auth gate, CSV/JSONL validation, config building, publish
  (409s, merged-model promotion, newest-merge pick, revert), eval, page
  serving.
- Core: memory-budget passthrough (omitted at 0 / passed when set),
  tiny-run `steps_per_report` cap, completed-run step/model fields,
  `X-Fusion-Source` header. Suite: 551 passed, 15 skipped.

## [1.0.0rc10] - 2026-10-15

Closes #65: SFT checkpoint resume + engine-restart detection, and closes the
data-path gaps found during live verification.

### Added

- `ft sft --resume` / `ft bnup-train --stage sft|pipeline --resume`
  (issue #65): checkpoint-aware restart. Finds the newest numbered
  checkpoint (`NNNNNNN_adapters.safetensors`) in the run's fusion-mlx
  adapter dir, stages it (upstream loads exactly `adapters.safetensors`,
  so the source checkpoint is never overwritten by the resumed job's first
  save), submits the job with `init_adapter_path` (fusion-mlx #1182,
  v0.10.12+), and subtracts the checkpoint step from `iters` (the upstream
  step counter restarts at 0). Already fully trained → completed no-op
  result pointing at the checkpoint. No checkpoint → loud warning, fresh
  start.
- `SFTConfig.init_adapter_path` — resume field, mirrors `RLSLConfig`.
- Engine-side job-loss detection (#65 pt2): a 404 on the job poll now
  fails fast as `interrupted` instead of zombie-polling to the 24h
  deadline (`MLXFineTuneError.status_code` carries the HTTP status; the
  transient retry/backoff semantics are unchanged). A `cancelled` terminal
  status (fusion-mlx restart recovery marks RUNNING/QUEUED jobs cancelled)
  gets the same interrupted treatment with the latest checkpoint printed
  and a `--resume` hint.
- `job_state.json` liveness state under the run output dir (#65 pt3/4):
  `job_id / adapter_dir / step / resume_step / status`, persisted
  atomically at job submit, on every step change, and on every terminal
  path — engine-side `os._exit` bypasses atexit, so state is only as
  durable as the last poll.
- 6th DPO negative generator `make_param_drift` (#57 follow-up,
  sft-dpo-1013 §2.4): rewrites one `params.<name>` reference in a
  pipeline eval expression to a drifted name (Chinese synonym / English
  abbreviation / case / suffix variants, vocabulary from real e2e
  failures). JSON and schema stay valid — the flaw surfaces only at eval
  execution (KeyError on the drifted key), which is exactly the E1 class
  static validation cannot catch. `generate_negatives` gives param_drift
  up to 2 priority slots per sample (highest contrast density for the
  top failure mode).

### Fixed

- `deduplicate` silently deleted every messages/ShareGPT-format sample
  (the E5 docstring said empty-key samples are kept; the code dropped
  them). A chat-format SFT dataset loaded "successfully" with 0 samples
  and trained on an empty file. Chat turns are now serialized into the
  dedup key — identical conversations dedup, distinct ones survive.
- `SFTTrainer.train` routed chat-format datasets through the flat loader,
  whose `_normalize` drops samples without input/output fields. New
  `DatasetLoader.load_sft` sniffs the dominant schema and routes chat
  datasets through `load_messages` (messages pass through to
  `stage_dataset` for the tokenizer chat template).
- `_apply_extras` (sft CLI) dropped YAML `save_every` / `log_every` /
  `quant_bits` / `max_steps` — a config-file `save_every: 50` ran with the
  default 500 and wrote no numbered checkpoint, leaving `--resume`
  nothing to resume from.

## [1.0.0rc9] - 2026-10-13

Closes #63: adds a true quantized DoRA (`qdora`) method so the BNUP
Qwen2.5-32B SFT config matches the PRD baseline (16-bit LoRA / 8-bit
QDoRA) instead of falling back to 8-bit QLoRA.

### Added
- **`qdora` method (#63):** `SFTConfig.method` now accepts `qdora` —
  maps to fusion-mlx `fine_tune_type="dora"` + `quantize_base=True` +
  `quant_bits` (frozen quantized base + DoRA adapters). mlx_lm 0.31.3
  `DoRALinear.from_base` accepts `QuantizedLinear`, so the combination
  works — verified numerically (forward + gradient flow) and live
  end-to-end on real hardware: a QDoRA job on Llama-3.2-1B-Instruct-4bit
  via fusion-mlx quantized the base to 8-bit, trained DoRA adapters
  (`lora_a`/`lora_b`/`m` magnitude tensors in adapters.safetensors),
  completed, and the adapter mounted for inference.

### Changed
- `bnup_sft_32b` preset + `configs/bnup_sft_32b.yaml`: `method` qlora →
  qdora (PRD baseline per #63; quant_bits stays 8).
- CLI `--method` help text now lists `qdora`.

### Tests
- 3 new tests: `test_run_training_qdora_real` (config mapping: dora +
  quantize_base + quant_bits=8 + lora fields), `test_run_training_qdora_
  quant_bits_clamped` (same (4,8) gate as qlora), updated bnup preset/
  config assertions (method=qdora). 523 pass, 15 skip.

## [1.0.0rc8] - 2026-10-10

Closes the #57 test coverage gap: DPO negative factory tests now call the
real `validate_dsl_str` from fusion-core to prove each generated negative
actually fails DSL schema validation (true negatives, not structural
heuristics).

### Tests
- **`test_bnup_negatives.py` (#57 gap):** added 7 `validate_dsl_str`-based
  tests — one per generator (`make_broken_json`, `make_direct_answer`,
  `make_wrong_visual_type`, `make_missing_tag`, `make_format_drift`) plus
  a comprehensive `test_all_negatives_fail_dsl_validation` on
  `generate_negatives` output, and a `test_valid_dsl_passes_validation`
  baseline proving the positive sample passes (so the negatives are
  meaningful). The test file's docstring already claimed "proving they
  are true negatives" — now it actually does. 521 pass, 15 skip.

## [1.0.0rc7] - 2026-10-10

Closes the 3 acceptance-review gaps on #58 (merge) + #59 (BNUP config):
`ft merge --publish` TypeError crash, missing post-merge sample
validation, and missing `presets/bnup.py` preset module.

### Fixed
- **`ft merge --publish` TypeError (#58 Gap 1):** the merge CLI called
  `publish_adapter_if_enabled(adapter_name=..., ...)` but the actual
  signature is `(config, result)` — crashed every time `--publish` was
  used. Fixed by adding `ModelHubClient.publish_merged_model(name,
  base_model)` (POSTs `model_type="mlx"`, no `base_model_id` — a merged
  model is a standalone model, not a LoRA adapter) and calling it
  directly from the merge CLI. Idempotent on 409 via `_find_existing_model`.
- **Post-merge sample validation (#58 Gap 2):** `merge_and_export` now
  generates a Socratic-format inference sample from the merged model
  (via `chat_completions` + `DEFAULT_VALIDATION_PROMPT`) and runs
  `validate_merged_model` to check `<thinking>`+`<dsl>` format. Best-effort
  (fail-visible warning on generation error, never gates the merge). New
  `generate_validation_sample` helper. The merge CLI prints the validation
  result.

### Added
- **`presets/bnup.py` module (#59 Gap 1):** dedicated BNUP preset module
  with `BNUP_SFT_32B`, `BNUP_DPO_32B`, `BNUP_PIPELINE` — the single source
  for BNUP training presets. SFT/DPO values sourced from the existing
  `sft/presets.py` + `rlsl/presets.py` (no duplication); `BNUP_PIPELINE`
  adds orchestration config (`sft_preset`, `dpo_preset`, `chain_adapter`,
  `merge`).

### Tests
- 17 new tests: `publish_merged_model` (4), `generate_validation_sample` (4),
  `merge_and_export` validation wiring (2), `presets/bnup.py` (7). 514 pass,
  15 skip.

## [1.0.0rc6] - 2026-10-10

DPO adapter chaining — wires the now-landed upstream `init_adapter_path`
(fusion-mlx #1149, closed via PR #1152) so the BNUP pipeline runs true
two-stage SFT→DPO training instead of an independent DPO LoRA.

### Added
- **`RLSLConfig.init_adapter_path`** — SFT adapter path to chain DPO/ORPO on.
  Flows into `_build_dpo_config`/`_build_orpo_config` and is only sent to
  fusion-mlx when non-empty (empty = fresh LoRA, backward compat). The policy
  model loads with the SFT adapter; the reference model stays base-only
  (DPO theory).
- **BNUP pipeline auto-chaining** — the `bnup-train --stage pipeline` closure
  now captures the SFT result's `final_model_path` and passes it to the DPO
  stage's `init_adapter_path`. The merge step merges the DPO adapter (carrying
  the SFT→DPO signal) with the base. Falls back to fresh LoRA with a loud
  warning if SFT produced no adapter path.
- **`--init-adapter-path` CLI flag** on both `rlsl` and `bnup-train`, so a
  standalone `dpo` stage can chain on a prior SFT adapter manually.
- **ORPO chaining** — `ORPOTrainer._build_orpo_config` gains the same
  `init_adapter_path` passthrough (same upstream DPOService path).

### Changed
- `configs/bnup_dpo_32b.yaml` — replaced the stale adapter-chaining-gap note
  with chaining documentation; added `init_adapter_path: ""` field.
- `_invoke_sft`/`_invoke_rlsl` now return their result dicts (previously
  `None`) so the pipeline can extract `final_model_path`/`adapter_path`.

### Tests
- DPO/ORPO config-builder tests assert `init_adapter_path` is included when
  set and omitted when empty.
- RLSLEngine end-to-end test asserts `init_adapter_path` reaches the
  fusion-mlx job config.
- CLI tests assert the flag is registered and flows into `RLSLConfig`.

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
