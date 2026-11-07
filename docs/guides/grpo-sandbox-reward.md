---
doc_id: fusion-trainer/grpo-sandbox-reward
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "GRPO 沙箱奖励实战", en: "GRPO sandbox reward in practice" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - docs/guides/grpo-sandbox-reward.md
---

# GRPO 沙箱奖励实战 / GRPO sandbox reward in practice

面向需要在 fusion-trainer 里跑带沙箱奖励的 GRPO 训练的工程师。前置：fusion-mlx 存活且已加载文本模型；数据集为 JSONL。

For engineers running GRPO with sandbox reward in fusion-trainer. Prerequisites: fusion-mlx alive with a text model loaded; dataset in JSONL.

## 1. 数据集格式 / Dataset format

每行一个 JSON 对象，二选一 / One JSON object per line, either shape:

```json
{"prompt": "Solve: 12 * 7 = ?", "answer": "84"}
{"prompt": "Write a function that returns the factorial of n", "tests": "assert f(5) == 120"}
```

`answer` 走 math 奖励（答案归一化），`tests` 走 code 奖励（执行补全代码 + 断言）。

`answer` selects the math reward (answer normalization); `tests` selects the code reward (executes the completion plus assertions).

## 2. 启动训练 / Launch

```bash
ft rlsl --config configs/rlsl_14b.yaml \
    --model qwen2.5-7b-4bit \
    --dataset /path/to/grpo_prompts.jsonl \
    --method grpo \
    --sandbox code \
    --group-size 8 \
    --grpo-iters 1 \
    --run-name my_grpo_run
```

## 3. 选择后端 / Pick a backend

| Backend | Behavior | Endpoint |
|---|---|---|
| `rft` (default) | reward-filtered SFT approximation (RFT/RAFT-style): sample -> score -> top-k -> SFT/LoRA job. Not true policy gradient. | `/admin/api/fine-tune/jobs` |
| `grpo` / `policy_gradient` | true policy-gradient GRPO server-side in fusion-mlx; trainer hosts a local reward endpoint fusion-mlx calls back | `/api/fine-tune/grpo/jobs` |

`policy_gradient` is an alias of `grpo` (normalized in `TrainerConfig.__post_init__`). True `grpo` requires fusion-mlx #363 Phase 2. Set via YAML `rlsl.grpo_backend: grpo`.

## 4. 沙箱约束 / Sandbox constraints

- code 奖励会执行模型输出 —— 必须显式 `FUSION_CODE_SANDBOX_TRUSTED=1`，否则拒绝执行。 / The code reward executes model output — set `FUSION_CODE_SANDBOX_TRUSTED=1` explicitly or execution is refused.
- macOS 默认走 seatbelt（`sandbox-exec`）：禁网、数据读取限制在解释器前缀 + stdlib + tmpfs 工作区、超时杀进程；`~/.ssh`、`/etc`、同级项目源码读取被拒。 / macOS defaults to seatbelt (`sandbox-exec`): network denied, data reads confined to the interpreter prefix + stdlib + tmpfs workspace, killed on timeout; reads of `~/.ssh`, `/etc`, and sibling project source are denied.
- 非 macOS 回退 rlimit 并告警；可用 `FUSION_CODE_SANDBOX_BACKEND=seatbelt|rlimit` 强制。 / Non-macOS falls back to rlimits with a warning; force with `FUSION_CODE_SANDBOX_BACKEND=seatbelt|rlimit`.
- `--sandbox off`（默认）不执行任何代码；`auto` 依据字段选择但仍是 opt-in。 / `--sandbox off` (default) executes nothing; `auto` picks by field but remains opt-in.

## 5. 验证 / Verify

run 结束后检查 `{cache_path}/runs/<run_id>/events.jsonl` 的奖励事件与 `metrics.prom` 的 reward 曲线；详见 `docs/troubleshooting.md`。

After the run, inspect reward events in `{cache_path}/runs/<run_id>/events.jsonl` and the reward curve in `metrics.prom`; see `docs/troubleshooting.md`.
