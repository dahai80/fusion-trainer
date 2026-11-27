---
doc_id: fusion-trainer/rollout-tap
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "Gateway rollout tap 设计（RFT/GRPO 数据源）", en: "Gateway rollout tap design (RFT/GRPO data source)" }
audience: engineer
status: draft
version: 1.0.0rc25
updated: 2026-11-10
owner: "@dahai"
source_of_truth:
  - docs/rollout-tap.md
---

# Rollout tap: gateway-side collection feeding RFT/GRPO (#86)

Design reference adopted from microsoft/agent-lightning v1.0
(proxy-side rollout collection: the serving proxy mirrors production
LLM traffic into a rollout store; the trainer consumes the store as
RL data — no in-training sampler required).

Fusion maps this pattern onto the existing topology with **no second
proxy**: fusion-gateway (`:11432`) is already the single choke point
in front of fusion-mlx (`:11434`) for every domain app.

## Seams

| # | Seam | Owner | Status |
|---|------|-------|--------|
| 1 | Rollout tap middleware (mirror, zero added latency) | fusion-gateway | upstream issue filed |
| 2 | Rollout sink (JSONL store + versioned schema) | fusion-gateway | upstream issue filed |
| 3 | Trainer-side consumption (sink records → RFT/GRPO datasets) | fusion-trainer | this repo, shipped rc25 (`ft datasets from-rollouts`) |
| 4 | No second proxy | — | gateway tap reuses the existing :11432 path |

## Rollout record schema (v1)

One JSON object per line in the sink file. Gateway-owned fields are
written by the tap; `outcome` is written by whichever producer can
observe the result (cowork node completion, app feedback hook). A
record without `outcome` is still valid — it feeds GRPO prompt sets
but never the positive SFT set.

Outcome sidecars (#97): producers that cannot write into the tap file
(e.g. office-api `rollout-outcomes.jsonl`) append outcome records to a
separate JSONL instead — same schema, identified by carrying `outcome`
and no `source` field (the tap contract always writes `source` and
never `outcome`). The trainer joins them reader-side by `rollout_id`
(inline outcomes win): `ft datasets from-rollouts --outcomes <sidecar>`
and the dashboard `rollout_section` both apply the join; sidecar lines
are never counted as separate trajectories.

```json
{
  "schema_version": 1,
  "rollout_id": "ro-01HV...",
  "tenant_id": "tenant-a",
  "source": "cowork:flow-42/node-3",
  "model": "qwen3-27b-8bit",
  "created_at": "2026-11-10T08:00:00Z",
  "messages": [{"role": "user", "content": "..."},
               {"role": "assistant", "content": "..."}],
  "finish_reason": "stop",
  "latency_ms": 812,
  "prompt": "first user turn (duplicate kept for GRPO convenience)",
  "outcome": {"signal": "task_success", "score": 1.0, "detail": "flow node passed"}
}
```

Required fields: `schema_version`, `rollout_id`, `model`,
`messages`. Optional: everything else. `messages` follows the D1
chat schema (`role` in user/assistant/system/tool, non-empty string
`content`) — the same validation `DatasetLoader.load_messages`
applies, so tap records and staged datasets stay interchangeable.

## Opt-in matrix

Collection is opt-in per API key / tenant, configured on the
gateway tap (seam #1). Defaults are all-off; fusion-trainer never
reads a sink that was not produced for its tenant.

| Dimension | Values | Default |
|-----------|--------|---------|
| tenant | any registered tenant | off |
| API key | per-key override (deny beats allow) | off |
| routes | `/v1/chat/completions`, `/v1/completions` | both |
| PII gate | `redact` (drop content, keep shapes) / `raw` | redact |
| outcome backfill | reader-side join by `rollout_id` (trainer owns it: `ft datasets from-rollouts --outcomes <sidecar>` + dashboard `rollout_section`; #97) | none |

## Reward-signal mapping (deterministic, Rule 5)

`outcome.signal` → scalar reward, fixed in
`fusion_trainer/dataset/rollout_ingest.py` — no model routing:

| signal | default score | meaning |
|--------|---------------|---------|
| `task_success` | 1.0 | downstream task/flow node succeeded |
| `user_accepted` | 1.0 | human accepted the completion |
| `user_edited` | 0.5 | human kept it after editing |
| `user_rejected` | 0.0 | human rejected |
| `error` | 0.0 | upstream/finish error |
| `none` | excluded | no outcome observed yet |

An explicit numeric `outcome.score` in [0, 1] overrides the signal
default. `--min-score` (default 1.0) selects the positive set for
SFT; records below the bar are counted loudly, never fabricated into
negatives (negative mining stays the from-eval DPO path's job).

## Trainer-side consumption (shipped)

```
ft datasets from-rollouts --sink rollouts.jsonl \
    --output sft.jsonl [--grpo-output prompts.jsonl] \
    [--failures-output failures.jsonl] [--min-score 1.0]
```

- SFT (`--output`): positive rollouts (`score >= min-score`) with
  valid messages → `{"messages": [...]}` D1 samples — trains via the
  existing `ft sft` path unchanged.
- GRPO prompts (`--grpo-output`): every record with a non-empty
  prompt → `{"prompt", "reward", "metadata"}` — the `reward` ref
  field survives `DatasetLoader.load_grpo` and documents the observed
  outcome for sandbox/analysis; on-policy sampling still decides the
  trained completions.
- Failure channel (`--failures-output`, #111): scored-negative
  rollouts (`reward < min-score`, unobserved excluded) written as full
  records with their outcome — the input to the resynthesis stage. An
  empty failure set writes an empty file (honest), never an error.
  The manifest carries `weakness` (per-signal failure counts + bounded
  sample rollout_ids; per-task breakdown when outcomes carry
  `task_id`), `struggle` (per-rollout deterministic struggle features
  over non-zero-scored failures — redundant calls, error signals,
  normalized score: the first-pass judge for step-wise expert
  injection, #116), and `efficiency` (offline redundant consecutive
  tool-call rate — the BEFORE number for any future efficiency reward;
  the budget anchor must be fixed before training, never
  policy-relative).
- Efficiency budgets (`--budgets-output`, #117 CARE-PO): per-task
  frozen tool-call budgets — the p50 tool-call count of each task's
  positive baselines (reward >= 1.0, `outcome.task_id` required; tasks
  with fewer than 3 positives are omitted loudly). Pin the budget into
  the GRPO round config BEFORE the first training step; the anchor is
  difficulty-anchored and frozen, never policy-relative.
  `budget_penalty(tool_calls, budget, redundant_consecutive)` is the
  matching bonus-only reward helper (a no-tool-call trajectory scores
  0, never negative — the CARE-PO collapse is structurally impossible).
  An empty budget set (no task coverage yet) writes an empty file.
- Skill library (`ft skills mine`, #123 gradient-free path 2):
  procedural skills mined from positive rollouts — a skill is a
  consecutive assistant tool-call chain (normalized like the
  redundant-call detector) that succeeded in >= `--min-successes`
  distinct rollouts (default 2; success_count counts distinct rollouts,
  repeats inside one rollout count once). Output JSONL sorted by
  success_count: `{skill_id, tools, chain, success_count, task_ids,
  example_rollout_id, mined_at}` — trajectories stay in the tap, the
  library references them by rollout id (fusion-skills retrieval
  alignment). An empty library is honest. Schema v2 trust requirements
  (frontier watch #16, Zep Agent Skills pattern): every skill carries
  field-level evidence citations ({rollout_id, task_id, signal, score,
  verifier} — verifier = office validator name@version when the outcome
  was validator-verified, else `outcome_signal`) plus a versioned
  admission block recording the gate (verified-attempts-only is
  structural: reward >= 1.0 is the only candidate source).
- Skill retrieval (`load_skill_library` / `render_skill_block`,
  `ft skills show`): the harness contract — fail-visible loader
  (missing file raises, malformed/newer-schema lines skipped loudly)
  and the deterministic prompt-injection block (compact skill list with
  success_count + verifier provenance, explicit max_skills/max_chars
  budgets). fusion-skills consumes the same renderer.
- Sidecar manifest `<output>.from_rollouts.json` (atomic write)
  carries the sink sha256 + per-signal counts; `ft sft` auto-detects
  it beside the dataset and records `source_export_sha256` in the run
  result (same traceability as from-eval, #76 requirement 3).

## Guardrails

- 100% offline: sink is a local/intranet JSONL file; no egress.
- Opt-in only, redact-by-default at the tap; trainer never widens it.
- Reward-hacking countermeasures: deterministic signal mapping; no
  self-grading (trainer cannot mark its own rollouts positive);
  explicit `score` bounded to [0, 1]; prompts without outcomes are
  usable for GRPO but can never enter the positive SFT set.
- Mirror tap is failure-isolated: sink write failure degrades to a
  no-op log on the gateway (upstream contract); trainer ingestion is
  fail-visible (bad lines skipped with counts, zero-record sink
  raises).

## Acceptance mapping (issue #86)

1. Design doc — this file.
2. Rollouts → GRPO/RFT datasets without harness changes — seam #3
   shipped; end-to-end through a live tap gated on seam #1/#2
   (fusion-gateway issue).
3. Perf: tap must add no measurable gateway latency — owned by the
   fusion-gateway issue (mirror write is async, off the request path).

## Non-goals

- No second proxy, no trainer-side sampling loop (agent-lightning's
  decorator API is out of scope — fusion's trainer is pure HTTP
  delegation).
- No DPO mining from tap records in this iteration (from-eval keeps
  that role; a rollout has at most one outcome, preference pairs
  need sibling arms).
