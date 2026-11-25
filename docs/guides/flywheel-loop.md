---
doc_id: fusion-trainer/flywheel-loop
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "飞轮闭环演练：训练 -> 合并 -> 发布 -> 服务 -> 评估 -> 记录", en: "Flywheel loop walkthrough: train -> merge -> publish -> serve -> eval -> record" }
audience: engineer
status: draft
version: 1.0.0rc31
updated: 2026-11-25
owner: "@dahai"
source_of_truth:
  - docs/guides/flywheel-loop.md
---

# Flywheel Loop Walkthrough: train -> merge -> publish -> serve -> eval -> record

The closing leg of the training flywheel, end to end (#95). Every step is a
real command; the loop invariant is: **the checkpoint recorded in the new
flywheel round == the published adapter == the model id fusion-mlx serves.**

## 1. Train

```bash
ft sft --dataset runtime_cache/flywheel/round1/sft.jsonl \
       --model mlx-community--Qwen3.5-4B-MLX-4bit \
       --lora-rank 8 --num-epochs 1
```

Or from the trainer console (`http://127.0.0.1:8790/`): upload the dataset
on the Datasets tab, start the run on the Training tab. Runs execute as
background threads in the bridge process and stream `events.jsonl`.

Memory note: fusion-mlx's admission gate rejects jobs over budget. For a 4B
LoRA run alongside resident serving models, pass `--memory-limit-gb 20`
(console: Memory budget field) or enable 8-bit quantized base training.

## 2. Merge + Publish

```bash
ft merge --run-id <run_id>        # writes the merged standalone model to ~/.fusion-mlx/models
ft publish --run-id <run_id>      # registers it in fusion-model-hub; fusion-mlx serves it
```

Console equivalents: Merge & Publish tab (`POST /runs/{id}/merge`,
`POST /publish`).

## 3. Probe: confirm the served model id

```bash
curl -s http://127.0.0.1:11434/v1/models | python3 -m json.tool | grep <adapter-name>
curl -s http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<published-model-id>", "messages": [{"role": "user", "content": "ping"}]}'
```

The model id returned here must equal the publish output's model id.

## 4. Eval + auto-record

Start the watcher once (it stays alive across eval runs):

```bash
ft flywheel watch --dir <fusion-eval-results-dir> --gate
```

Then run fusion-eval against the published model as usual. When
`<results-dir>/<eval-run>/summary.json` lands, the watcher records a gated
flywheel round automatically — no manual `ft flywheel record`. The gate
promotes or discards vs the incumbent (#82); the verdict (including
discards) shows on the dashboard round chart.

For CI/cron, use `--once` for a single scan pass. Consumption state lives in
`{cache}/flywheel_watch.json` (summary sha256) — re-runs never double-record;
a summary rewritten with new content is recorded as a new round.

## 5. Verify on the dashboard

`http://127.0.0.1:8790/dashboard` shows: the new round in the progression
chart, the gate verdict in the 轮次管控 card, and the model pill reflects the
latest recorded round's model. Loop invariant: round checkpoint == published
adapter id == probed model id.
