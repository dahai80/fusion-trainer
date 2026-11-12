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
| outcome backfill | app-observable producers only | none |

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
    --output sft.jsonl [--grpo-output prompts.jsonl] [--min-score 1.0]
```

- SFT (`--output`): positive rollouts (`score >= min-score`) with
  valid messages → `{"messages": [...]}` D1 samples — trains via the
  existing `ft sft` path unchanged.
- GRPO prompts (`--grpo-output`): every record with a non-empty
  prompt → `{"prompt", "reward", "metadata"}` — the `reward` ref
  field survives `DatasetLoader.load_grpo` and documents the observed
  outcome for sandbox/analysis; on-policy sampling still decides the
  trained completions.
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
