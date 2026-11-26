---
doc_id: fusion-trainer/task-registry
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: Task registry (issue #120)
audience: developer
status: draft
version: 1.0
updated: 2026-11-26
owner: "@dahai"
source_of_truth:
  - fusion_trainer/tasks.py
  - fusion_trainer/cli.py
---

# Task registry (issue #120)

Trainer-side home for the office data-engine pieces handed off from
fusion-office #49/#50: task definitions with known-valid solution corpora,
the validator audit gate, and cross-app task synthesis.

## Registry file

Append-only JSONL at `{FUSION_TRAINER_CACHE}/tasks/registry.jsonl`.
Latest record per `task_id` wins (same contract as the outcome sidecar).
Every record requires:

- `task_id` (non-empty string)
- `curriculum_stage` (1, 2, or 3)
- `validator` object: `name`, `version` (non-empty strings), `status`
  (`untested` | `passing` | `failing`)
- `known_valid_solutions`: list of non-empty strings (human-verified
  correct solutions for the task)

## Validator audit gate (VerifierBench)

A task may emit training signal only when its validator is `passing`
AND the known-valid corpus is non-empty. Everything else is masked from
the positive set — a validator that accepts nothing looks like "strong
rejection" while silently poisoning the batch.

```bash
ft tasks add --file tasks.json        # single object or list
ft tasks list                         # per-task gate status
ft tasks gate                         # allowed vs masked report
```

## Cross-app synthesis (stage 2)

Synthesizes one template task per edge of the office-api entity-tool
graph (GET /v1/graph, office-api >= 0.1.27). Deterministic, LLM-free.
Synthesized tasks start `untested` with an empty corpus, so the gate
masks them until both are filled in. Already-registered task ids are
never re-upserted (a re-upsert would reset a passing validator).

```bash
ft tasks synthesize --graph-url http://127.0.0.1:<office-api-port>/v1/graph
```
