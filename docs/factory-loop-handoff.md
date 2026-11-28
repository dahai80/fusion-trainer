---
doc_id: fusion-trainer/factory-loop-handoff
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "Factory 轨迹闭环：数据契约 + 评估门 + 晋升标准", en: "Factory-trajectory loop: dataset contract, eval gate, promotion criteria" }
audience: engineer
status: draft
version: 1.0.0rc50
updated: 2026-11-28
owner: "@dahai"
source_of_truth:
  - docs/factory-loop-handoff.md
  - docs/rollout-tap.md
---

# Factory-trajectory loop: dataset contract, eval gate, promotion criteria (#136)

Handoff doc first (role split per #136): cowork owns trajectory export,
trainer owns distillation + eval gate + promotion, the model catalog owns
registration. This document fixes the three contracts before any
implementation. Competitive frame: Factory.ai's moat is custom models
trained on customer codebases; the fusion counterpart is the flywheel
(cowork run trajectories -> condition-compile distillation (#639) ->
trainer SFT -> factory model slot).

## 1. Dataset contract (cowork -> trainer)

- **Source**: cowork trajectory JSONL exports (condition-compile output
  from cowork#639), one record per completed run.
- **Trainer-side format**: the existing `{"messages": [...]}` chat SFT
  schema (same validation as `DatasetLoader.load_messages` and the
  rollout-tap D1 schema — docs/rollout-tap.md). Tap records and cowork
  exports stay interchangeable by construction.
- **Ingestion**: `ft datasets from-rollouts` is the reference reader for
  tap-shaped exports; cowork exports that already carry `messages` are
  accepted directly. A dedicated `ft datasets from-cowork` wrapper lands
  only if cowork's shape diverges (do not fork the schema preemptively).
- **Filters (all deterministic, rule 5)**:
  1. successful runs only — run status passed / outcome
     `task_success` (score >= 1.0);
  2. gate-passed outputs — the run's own verification (cowork node
     gates / office validators) must have passed;
  3. PII / classification screen — per identity#119 flag: records
     flagged or unclassifiable are dropped loudly (counted, never
     silently included).
- **Fresh-data discipline (#82)**: the export sha256 is recorded at
  ingestion; re-training on a consumed export raises unless
  `--allow-reconsume` (flagged, never silent).

## 2. Trainer job spec

- **Model**: per-tenant LoRA on the local 27B base (Qwen3.8-27B-4bit),
  8-bit QLoRA, `use_mxfp8` per rc2 defaults; LoRA rank/alpha follow the
  existing SFT presets.
- **Isolation**: one adapter per tenant; adapter run name carries the
  tenant id (`<tenant>-<date>-sft`) so the catalog and the gateway route
  can key on it.
- **Eval gate before promotion (#134 line)**: the trained adapter is
  evaluated on a fusion-bench/fusion-eval task set for the domain, and
  promotes only via `record_round(gate=True, gate_mode="wilson")` —
  Wilson 95% CI strict non-overlap vs the incumbent (or the registered
  floor band). No unverified numbers: every promotion/rejection appends a
  claims-registry row with run ids (C-1).

## 3. Promotion criteria

- **Threshold**: the eval gate blocks promotion below the registered
  floor band (E-2 noise floor for the domain, repeat-3 majority vote).
  A candidate that merely beats a weak incumbent but sits inside the
  floor band is discarded — `gate_promotion` enforces this.
- **Sample size**: pre-registered per E-3 (n=60 minimum for a real
  separability claim; n=30 bands are ~±13pp and cannot separate honest
  improvements).
- **Registration**: a passing adapter is registered in the model catalog
  as a tenant-scoped adapter on the 27B base (model_type `llm`, adapter
  path + run ids in metadata). The demand dual-draft local arm can then
  route to the tenant adapter; cost/quality deltas feed commerce#16
  metering.

## 4. Acceptance loop (offline synthetic tenant)

One synthetic tenant loop executed offline:
10 trajectories -> dataset -> LoRA -> eval report -> catalog registration.
No customer data leaves the local stack at any step (all paths local:
cowork export dir, trainer runtime_cache, engine adapters dir, catalog
registry) — a path audit runs as part of the acceptance evidence.

## 5. Open items

- [ ] cowork export shape confirmation (does #639 output carry
      `messages` directly or a wrapper?).
- [ ] identity#119 flag mechanics (flag field name / drop-vs-quarantine).
- [ ] model catalog tenant-adapter registration surface (model-hub
      metadata fields for tenant scoping).
- [ ] commerce#16 metering payload (which quality/cost fields).
