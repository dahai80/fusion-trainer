---
doc_id: fusion-trainer/tool-trajectories
product: fusion-trainer
class: platform
entry: developer
doc_type: guide
title: { zh: "工具调用轨迹（#88）", en: "Tool-calling trajectories (#88)" }
audience: engineer
status: draft
version: 1.0.0rc31
updated: 2026-11-25
owner: "@dahai"
source_of_truth:
  - docs/tool-trajectories.md
---

# Tool-Calling Trajectories (#88)

Train on agent trajectories that contain function/tool calls — e.g. records
captured from the fusion-gateway rollout tap.

## Supported message schema

Standard OpenAI-compatible chat messages. Three turn shapes:

```jsonc
// 1. text turn
{"role": "user", "content": "Make the sheet bold"}

// 2. assistant tool call (content may be empty; payload in tool_calls)
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "sheet_bold", "arguments": {"range": "A1:B2"}}}
  ]
}

// 3. tool result (tool_call_id required)
{"role": "tool", "tool_call_id": "call_1", "content": "ok"}
```

`content` may also be a list of typed parts (`[{"type": "text", "text":
"..."}]`); it is passed through verbatim.

## Sample shape

```jsonc
{
  "messages": [ ...turns above... ],
  "tools": [{"type": "function", "function": {"name": "sheet_bold", "parameters": {...}}}]
}
```

The sample-level `tools` array is optional; when present it is forwarded to
`tokenizer.apply_chat_template(messages, tools=...)` by fusion-mlx
(mlx_lm.tuner `ChatDataset`), so templates that render tool grammar
(Llama-3.x, Qwen2.5+) get it.

## How it trains

- **With a tool-aware chat template** (default): tool fields survive the
  whole pipeline (`DatasetLoader.load_messages` → `stage_dataset` →
  mlx_lm `apply_chat_template`) and the tokenizer template renders them.
- **Without tool support in the template**: flatten offline with
  `fusion_trainer.dataset.formats.serialize_chatml(messages)` — assistant
  tool calls become deterministic `TOOL_CALL {"arguments": ..., "name":
  ...}` JSON lines (sorted keys) and tool results become
  `TOOL_RESULT[call_id] ...` lines; feed the result as `{"text": ...}`
  samples.

## Ingest entry points

- `ft datasets from-rollouts` (rollout tap sink) keeps `tool_calls` /
  `tool_call_id` and an optional record-level `tools` array.
- `ft datasets from-eval` keeps them for SFT export; DPO pair mining
  renders tool-call turns as `TOOL_CALL` lines so chosen/rejected text is
  never emptied by tool-only turns.

Invalid tool structure (orphan `tool` turn without `tool_call_id`, tool
call missing `function.name`/`arguments`) is dropped with a loud warning —
never silently mutated.

## 中文说明

支持 OpenAI 兼容的 tool-calling 轨迹:assistant 轮可带 `tool_calls`,`tool`
轮必须带 `tool_call_id`;样本顶层可选 `tools` 数组会透传给上游 chat
模板。模板不支持 tool 语法时,用 `serialize_chatml` 确定性展开为
`TOOL_CALL` / `TOOL_RESULT` 文本行。非法 tool 结构会被大声告警丢弃。
