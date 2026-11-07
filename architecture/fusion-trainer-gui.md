---
doc_id: fusion-trainer/gui-spec
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "fusion-trainer GUI 需求与原型", en: "fusion-trainer GUI mockup spec" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - architecture/fusion-trainer-gui.md
---

# Fusion-Trainer GUI 需求与高保真原型

> 目标:梳理 fusion-trainer 需要的 GUI,形成 md 格式高保真原型,向 fusion-studio 提需求落地,完成集成。
>
> 范围限定:本文档由 fusion-trainer 侧产出。可修改的代码仅限本仓库
> (fusion-trainer/);fusion-studio、fusion-agent-studio 属上游,按项目规则
> "上游先 issue 再 PR,跟着提交落地 code",GUI/RPC 改动以 issue/PR 形式提需求。

---

## 1. 集成现状(调查结论)

### 1.1 trainer 侧无 GUI/RPC 服务

fusion-trainer 目前**只有 CLI**(Typer `app`,三条命令:`sft`/`rlsl`/`info`)。
没有任何 JSON-RPC/UDS/FastAPI server 模块。`pyproject.toml` 声明的
`fastapi`/`uvicorn` 在代码中**未使用**。唯一的 server 是 `rlsl/reward_server.py`
(`RewardServer`/`RMRewardServer`,stdlib `ThreadingHTTPServer`),是 GRPO
内部 reward 回调,非 studio 面向。

### 1.2 fusion-agent-studio 中央 daemon 已分发 `trainer.*` RPC

`fusion-agent-studio/agent_runtime/daemon_server.py`(`DaemonServer`)监听
`/tmp/fusion-studio.sock`,newline-delimited JSON-RPC 2.0,被 fusion-studio 的
`IPCClient.swift` 调用。`TrainerDispatcher`(sub-dispatcher)已注册:

| method | 现状 |
|---|---|
| `trainer.trajectories.list` | 占位 |
| `trainer.sft` | 启动后台训练 task |
| `trainer.rlsl` | 启动后台训练 task |
| `trainer.info` | 系统/硬件信息 |
| `trainer.runs.status` | 终态状态(**无 live progress**) |

`TrainingService`(`agent_runtime/training_service.py`)**进程内 import** fusion-trainer
(`from fusion_trainer import TrainerConfig, SFTTrainer, SFT_PRESETS`,
`from fusion_trainer.rlsl import RLSL_PRESETS, RLSLEngine`),用
`asyncio.to_thread(trainer.train)` 跑训练。`_RUNS` dict 内存登记 run_id→status。

**缺口**(GUI 需要但 TrainingService 暴露不足):
1. 无 live progress(只有终态)— GUI 无法画 loss 曲线
2. 超参只暴露 model/preset/dataset,未暴露 full TrainerConfig 字段
3. `method` 只允许 `dpo|orpo|grpo`,**缺 `rlsl`**(完整闭环)
4. 无 adapters/checkpoints 列表(产出物不可见)
5. 无 run 列表/历史(registry 不持久化)

### 1.3 集成架构决策

```
fusion-studio GUI (SwiftUI)
    │  JSON-RPC 2.0 over UDS /tmp/fusion-studio.sock
    ▼
fusion-agent-studio DaemonServer  ← 上游,issue/PR
    │  TrainerDispatcher → TrainingService
    ▼
fusion-trainer (in-process import)  ← 本仓库,补 RunManager + live API
    │  HTTP
    ▼
fusion-mlx (真实训练引擎)
```

**本仓库改动**(Task 3):在 `fusion_trainer/` 加 `RunManager`(异步 run 启动 +
live progress 事件流 + run registry + adapters/checkpoints 列表),让
TrainingService 能 surface richer API。

**上游改动**(Task 4/5,issue/PR 不动代码):
- fusion-agent-studio:扩 `trainer.*` RPC(live progress、full config、
  `method=rlsl`、adapters list、run history)
- fusion-studio:注册 UpstreamService、`IPCTrainerMethods.swift`、
  TrainingView→真实 TrainerView、导航入口

---

## 2. GUI 总体设计

### 2.1 信息架构

对齐 fusion-studio 现有导航(自定义 4 列,非 NavigationSplitView;三级枚举
registry:SidebarSection/ProductSheet/Module)。

```
Sidebar  │  Column 2 (list)     │  Column 3 (detail)        │  Column 4 (inspector)
─────────┼──────────────────────┼───────────────────────────┼─────────────────────
Trainer  │  ▸ Runs              │  Run detail (loss curve)  │  Hyperparams summary
         │  ▸ New SFT Run       │  Config summary           │  Memory estimate
         │  ▸ New RLSL Run      │  ...                      │
         │  ▸ Datasets          │  Dataset rows preview     │
         │  ▸ Adapters          │  Adapter metadata         │
         │  ▸ Presets           │  Preset compare table     │
         │  ▸ Settings          │  Paths / ports / hub      │
```

顶层导航枚举(建议给 fusion-studio):

```swift
// 上游 issue:在 AppState.swift 三级 registry 注册 Trainer
SidebarSection.trainer
  ├─ Module.trainerDashboard      // Runs 列表 + 监控
  ├─ Module.trainerNewSft         // SFT 配置启动
  ├─ Module.trainerNewRlsl        // RLSL/GRPO 配置启动
  ├─ Module.trainerDatasets       // 数据集浏览
  ├─ Module.trainerAdapters       // 产出 adapter
  ├─ Module.trainerPresets        // 预设对照
  └─ Module.trainerSettings       // 路径/端口/状态
```

### 2.2 数据契约(GUI ↔ RPC)

GUI 消费以下 `trainer.*` RPC(扩展现有 thin surface):

| method | 入参 | 出参 | 用途 |
|---|---|---|---|
| `trainer.info` | `{}` | `{version, mlx_port, mlx_alive, models[], gpu_mem_gb, presets[]}` | Settings/Dashboard |
| `trainer.presets.list` | `{kind:"sft"\|"rlsl"}` | `[{name,description,memory_estimate_gb,hyperparams}]` | New Run 下拉 |
| `trainer.datasets.list` | `{}` | `[{name,category,n_samples,format,path}]` | Datasets 面板 |
| `trainer.datasets.preview` | `{name,limit}` | `{rows:[...],fields[],n_total}` | Dataset detail |
| `trainer.sft.start` | `TrainerConfig(JSON)` | `{run_id,status}` | New SFT Run |
| `trainer.rlsl.start` | `TrainerConfig(JSON, method∈dpo\|orpo\|grpo\|rlsl)` | `{run_id,status}` | New RLSL Run |
| `trainer.runs.list` | `{limit}` | `[{run_id,name,method,status,created,final_adapter}]` | Runs 列表 |
| `trainer.runs.status` | `{run_id}` | `{status,step,total_steps,metrics,phase}` | 终态 |
| `trainer.runs.progress` | `{run_id, since_step}` | `{events:[{type,step,...}]}` | **live loss 曲线** |
| `trainer.runs.stop` | `{run_id}` | `{status:"cancelling"}` | 停止 |
| `trainer.adapters.list` | `{model?}` | `[{name,base_model,method,adapter_path,run_id}]` | Adapters 面板 |
| `trainer.adapters.delete` | `{name}` | `{ok}` | 清理 |

`TrainerConfig` JSON = `TrainerConfig` dataclass 序列化
(`dataset` + `sft` + `rlsl` + `seed` + `model_hub_path` + `cache_path` + `mlx_port`)。
完整字段见 §3/§4 表。

### 2.3 live progress 事件

源自 `MLXFineTuneClient.wait_for_job` 的 `on_progress(job, new_events)` 回调,
经 RunManager 落盘 + 内存 ring buffer:

```jsonl
{"type":"train_loss","step":120,"train_loss":0.4123,"learning_rate":1.8e-05,"tokens_per_second":820,"it_sec":2.1,"peak_memory_gb":15.8}
{"type":"val_loss","step":120,"val_loss":0.3891}
{"type":"grpo_step","step":1,"loss":0.21,"mean_reward":0.62,"mean_ratio":1.04,"n_samples":8}
{"type":"phase","phase":1,"name":"reward_model_train"}
{"type":"status","status":"completed","adapter_path":"...","adapter_name":"..."}
```

`trainer.runs.progress` 返回 `since_step` 之后的事件;GUI 轮询(2s,对齐
`poll_interval=2.0`)或 SSE。**Rule 5**:progress 是确定性数据流(RunManager
回放落盘事件),不经模型。

---

## 3. 面板 1 — Dashboard / Runs

### 3.1 需求

- 最近 run 列表(status chip:running/completed/failed/cancelling)
- 选中 run → loss 曲线(train_loss + val_loss 双线)+ GRPO mean_reward
- 实时刷新(running run)
- 一眼看清:method / model / step / ETA / peak mem

### 3.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer › Dashboard                                            [⟳ Refresh]    │
├──────────────────────┬──────────────────────────────────────────────────────┤
│  Runs (12)           │  my_first_finetune  ●completed                         │
│                      │  SFT · LoRA · qwen2.5-7b-4bit · 2026-08-18 23:40        │
│  ● my_first_finetune │                                                        │
│    SFT · LoRA · ✓    │  Loss                                                   │
│  ● grpo_math_14b     │   0.50 ┤        ╲╲╲                                    │
│    GRPO · running 42%│   0.40 ┤    ╱╲╲   ╲╲                                   │
│  ○ dpo_align_7b      │   0.30 ┤  ╱╲╲        ╲╲___                             │
│    DPO · failed      │   0.20 ┤╱╲╲                                              │
│  ○ sft_code_lora     │   0.10 ┤──────────────────────── step                  │
│    SFT · completed   │        0    50   100  150  200  250  300               │
│  ...                 │                                                        │
│                      │  ── train_loss   -- val_loss                           │
│  [+ New SFT Run]     │                                                        │
│  [+ New RLSL Run]    │  Stats                                                 │
│                      │   Steps      312 / 312   ✓                             │
│                      │   Peak mem   15.8 GB / 96 GB                           │
│                      │   Time       10m 24s                                   │
│                      │   Adapter    ~/.fusion-mlx/adapters/.../my_first...    │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

Inspector(Column 4)超参摘要:

```
┌─ Hyperparams ──────────┐
│  method      lora      │
│  lora_rank   16        │
│  lr          2.0e-5    │
│  epochs      3         │
│  batch       1 ×4 acc  │
│  preset      sft_7b_lora│
│  est. mem    16 GB     │
└────────────────────────┘
```

---

## 4. 面板 2 — New SFT Run

### 4.1 表单字段(映射 SFTConfig + DatasetConfig)

| 分组 | 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|---|
| Model | `sft.model` | select(fusion-mlx models) | — | 必填,来自 `trainer.info.models` |
| | `sft.model_type` | lm / vlm | lm | |
| Method | `sft.method` | lora / qlora / full | lora | |
| | `sft.lora_rank` | int | 16 | method≠full 显示 |
| | `sft.lora_alpha` | float | 32.0 | |
| | `sft.lora_dropout` | float | 0.05 | |
| | `sft.quant_bits` | 4 / 8 | 4 | method=qlora 显示 |
| Dataset | `dataset.path` / `dataset.hub_dataset` | file picker / hub select | — | 二选一 |
| | `dataset.val_ratio` | float | 0.05 | |
| | `dataset.max_samples` | int | 0 | 0=全部 |
| | `dataset.max_seq_length` | int | 4096 | |
| | `dataset.input_field` / `output_field` | text | text / output | |
| Hyperparams | `sft.learning_rate` | float | 2e-5 | |
| | `sft.num_epochs` | int | 3 | |
| | `sft.max_steps` | int | 0 | 0=按 epoch 算 |
| | `sft.batch_size` | int | 1 | |
| | `sft.gradient_accumulation_steps` | int | 4 | |
| | `sft.lr_scheduler` | cosine/... | cosine | |
| | `sft.warmup_steps` | int | 100 | |
| | `sft.weight_decay` | float | 0.01 | |
| | `sft.max_grad_norm` | float | 1.0 | |
| Precision | `sft.use_mxfp8` | bool | true | **显示 ⏳ 上游阻塞提示**(fail-visible) |
| Checkpoint | `sft.save_every` | int | 500 | |
| | `sft.save_total_limit` | int | 3 | |
| Run | `sft.run_name` | text | sft_20261107 | |
| | `seed` | int | 42 | |
| Preset | `--preset` | select(10 SFT presets) | — | 选中后填充上面字段,可覆盖 |

### 4.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  New SFT Run                                                  [Cancel][Start]│
├─────────────────────────────────────────────────────────────────────────────┤
│  Preset  [sft_7b_lora          ▾]   LoRA 7B-class · est. 16 GB               │
│           ──────────────────────────────────────────────────                 │
│           sft_7b_full        Full FT 7B   · 32 GB                            │
│           sft_14b_qlora      QLoRA 14B    · 28 GB                            │
│           sft_32b_qlora      QLoRA 32B    · 48 GB                            │
│           sft_vlm_7b_qlora   QLoRA VLM 7B · 35 GB                            │
│           sft_code_7b_lora   Code LoRA 7B · 20 GB                            │
│           sft_finance_7b_lora Finance 7B  · 16 GB                            │
│           sft_science_7b_lora Science 7B  · 16 GB                            │
│           sft_router_light_1_5b_lora 1.5B  · 6 GB                            │
│           sft_router_light_3b_full   3B full· 12 GB                          │
│                                                                              │
│  ┌─ Model ──────────────┐  ┌─ Method ───────────────────────┐                │
│  │ Base model [qwen2.5-7b-4bit ▾]│ │ ( ) lora  (•) qlora  ( ) full │          │
│  │ Type       [lm      ▾]        │ │ LoRA rank  [16]  alpha [32.0] │          │
│  └────────────────────────┘     │ │ dropout    [0.05] quant [4bit]│          │
│                                  └───────────────────────────────┘           │
│                                                                              │
│  ┌─ Dataset ───────────────────────────────────────────────────────────┐    │
│  │ Source  (•) Hub  ( ) File                                            │    │
│  │ Hub     [sft/code_alpaca        ▾]   n≈12000 · jsonl                 │    │
│  │ Path    /path/to/data.jsonl (if File)                                │    │
│  │ Val ratio [0.05]  Max samples [0=all]  Max seq [4096]                │    │
│  │ input_field [text]  output_field [output]                            │    │
│  │ [Preview rows ▾]                                                      │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Hyperparameters ───────────────────────────────────────────────────┐    │
│  │ Learning rate [2.0e-5]  Epochs [3]  Max steps [0=auto]               │    │
│  │ Batch size [1]  Grad accum [4]  LR scheduler [cosine ▾]              │    │
│  │ Warmup [100]  Weight decay [0.01]  Max grad norm [1.0]              │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Precision & Checkpoint ────────────────────────────────────────────┐    │
│  │ [x] MXFP8 mixed precision   ⚠ upstream-blocked (#402) — 默认精度跑  │    │
│  │ Save every [500]  Keep last [3] checkpoints                          │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─ Run ─────────────────────────────────────────────────────┐               │
│  │ Run name [sft_20260818_2340]   Seed [42]                  │               │
│  └───────────────────────────────────────────────────────────┘               │
│                                                                              │
│  Memory estimate: ~16 GB / 96 GB M5    [▶ Start training]                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 面板 3 — New RLSL Run (DPO / ORPO / GRPO / RLSL)

### 5.1 表单字段(映射 RLSLConfig)

method 切换决定显示哪些组:

| method | 显示组 |
|---|---|
| dpo | Model + Dataset(preference)+ DPO + Run |
| orpo | Model + Dataset(preference)+ ORPO + Run |
| grpo | Model + Dataset(grpo prompt)+ GRPO + Sandbox + Run |
| rlsl | Model + Dataset(preference)+ Reward Model + GRPO + Sandbox + Run(闭环) |

字段表:

| 分组 | 字段 | 类型 | 默认 |
|---|---|---|---|
| Method | `rlsl.method` | dpo/orpo/grpo/rlsl | grpo |
| Model | `sft.model` | select | — |
| Dataset | `dataset.path`/`hub_dataset` | — | — |
| DPO/ORPO | `rlsl.beta` | float | 0.1 |
| Reward Model | `rlsl.reward_model` | select(adapter) | — | rlsl 专用 |
| | `rlsl.reward_num_epochs` | int | 2 | |
| | `rlsl.reward_learning_rate` | float | 1e-5 | |
| GRPO | `rlsl.grpo_backend` | rft / grpo | rft | grpo backend 需 #363 |
| | `rlsl.group_size` | int | 8 | |
| | `rlsl.grpo_iterations` | int | 1 | |
| | `rlsl.reward_top_k` | int | 4 | |
| | `rlsl.reward_scale` | float | 1.0 | |
| | `rlsl.clip_epsilon` | float | 0.2 | grpo backend |
| Sandbox | `rlsl.sandbox_mode` | off/code/math/auto | auto | |
| | `rlsl.max_tool_calls` | int | 3 | |
| Generation | `rlsl.response_length` | int | 512 | |
| | `rlsl.temperature` / `top_p` | float | 0.7 / 0.9 | |
| Training | `rlsl.learning_rate` | float | 5e-6 | |
| | `rlsl.num_epochs` | int | 1 | |
| Preset | `--preset` | select(3 RLSL presets) | — | 程序化用,GUI 暴露 |

### 5.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  New RLSL Run                                                  [Cancel][Start]│
├─────────────────────────────────────────────────────────────────────────────┤
│  Method  ( ) dpo  ( ) orpo  (•) grpo  ( ) rlsl (closed loop)                 │
│  Preset  [rlsl_grpo_14b         ▾]   GRPO 14B rft · est. 62 GB              │
│                                                                              │
│  ┌─ Model ──────────────┐  ┌─ Dataset (GRPO prompts) ─────────────────┐     │
│  │ [qwen2.5-7b-4bit ▾]  │  │ Hub [grpo/math_gsm8k      ▾]  n≈4000     │     │
│  └──────────────────────┘  │ Prompt schema: {prompt, answer, tests}   │     │
│                             │ [Preview ▾]                              │     │
│                             └──────────────────────────────────────────┘     │
│                                                                              │
│  ┌─ GRPO ─────────────────────────────────────────────────────────────┐     │
│  │ Backend   (•) rft (reward-filtered SFT)  ( ) grpo (policy-grad #363)│     │
│  │ Group G   [8]   Iters [1]   Top-k [4]   Reward scale [1.0]          │     │
│  │ Clip eps  [0.2]   (grpo backend only)                              │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  ┌─ Sandbox reward ─────────────────────────────────────┐                   │
│  │ Mode  ( ) off  ( ) code  ( ) math  (•) auto           │                   │
│  │ auto → tests 字段在则 code,answer/solution 在则 math │                   │
│  │ Max tool calls [3]                                    │                   │
│  └────────────────────────────────────────────────────────┘                   │
│                                                                              │
│  ┌─ Generation ──────────────────┐  ┌─ Training ───────────────────┐        │
│  │ Resp len  [512]               │  │ LR [5.0e-6]  Epochs [1]      │        │
│  │ Temp [0.7]  top_p [0.9]       │  │ Save every [200]            │        │
│  └────────────────────────────────┘  └───────────────────────────────┘        │
│                                                                              │
│  Memory estimate: ~62 GB / 96 GB M5    [▶ Start training]                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

method=rlsl 时追加 Reward Model 区:

```
┌─ Reward Model (Phase 1, closed loop) ─────────────────────────────┐
│ RM adapter  [rm_adapter_v3 ▾]  (or train new from this dataset)   │
│ RM epochs [2]  RM lr [1.0e-5]                                     │
│ Phase 2: GRPO real policy-grad, RM-rewarded (RM reward server)    │
│ Loop: RM train (#424) → reward/score (#431) → GRPO (#363)        │
└────────────────────────────────────────────────────────────────────┘
```

---

## 6. 面板 4 — Datasets

### 6.1 需求

- 列出 hub datasets(`{model_hub_path}/datasets/` 扫描,按 category)
- 预览前 N 行 + schema 推断
- 标记适用 method(sft/preference/grpo/multimodal)
- 只读(hub 不可写)

### 6.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer › Datasets                                            [⟳ Rescan]    │
├──────────────────────┬──────────────────────────────────────────────────────┤
│  ▸ sft (4)           │  sft/code_alpaca                                       │
│    code_alpaca       │  ┌────────────────────────────────────────────────┐  │
│    intent_cls        │  │ n_samples  12000   format  jsonl               │  │
│    gsm8k_sft         │  │ fields     text, output                        │  │
│  ▸ preference (2)    │  │ path       ~/fusion-model-hub/datasets/sft/... │  │
│    dpo_pairs         │  │ fits       SFT                                 │  │
│  ▸ grpo (3)          │  └────────────────────────────────────────────────┘  │
│    math_gsm8k        │                                                        │
│    code_tests        │  Preview (first 5)                                     │
│  ▸ multimodal (1)    │  ┌────────────────────────────────────────────────┐  │
│  ▸ domain (5)        │  │ {"text":"def add(a,b):...","output":"return..."}│  │
│                      │  │ {"text":"Write a fn...","output":"def ..."}      │  │
│                      │  │ ...                                              │  │
│                      │  └────────────────────────────────────────────────┘  │
└──────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 7. 面板 5 — Adapters(产出物)

### 7.1 需求

- 列出 fusion-mlx 已训 adapter(`GET /v1/adapters` 或 `ADAPTERS_PATH`)
- 显示 base_model / method / 来源 run / adapter_path
- 删除 / 导出 / 在 fusion-mlx 加载验证

### 7.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer › Adapters                                                          │
├──────────────────┬──────────────────────────────────────────────────────────┤
│  qwen2.5-7b-4bit │  my_first_finetune                                       │
│  ├ my_first_… ✓  │  base_model   qwen2.5-7b-4bit                            │
│  ├ intent_cls ✗  │  method       lora (rank 16)                             │
│  ├ dpo_align ✓   │  run          my_first_finetune · 2026-08-18             │
│  qwen2.5-14b     │  adapter_path ~/.fusion-mlx/adapters/qwen2.5-7b-4bit/... │
│  └ grpo_math ✓   │  status       loadable                                   │
│                  │                                                          │
│                  │  [Load in fusion-mlx]  [Export]  [Delete]                │
└──────────────────┴──────────────────────────────────────────────────────────┘
```

---

## 8. 面板 6 — Presets 对照

### 8.1 需求

- 10 SFT presets + 3 RLSL presets 对照表
- memory_estimate_gb + 关键超参 + 描述
- 一键 "New run from preset"

### 8.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer › Presets                              [SFT (10) | RLSL (3)]       │
├─────────────────────────────────────────────────────────────────────────────┤
│  Preset                  Method  Rank  LR      Epochs  Mem    [New run]      │
│  sft_7b_lora             lora    16    2.0e-5  3       16 GB  [▶]            │
│  sft_7b_full             full    —     1.0e-5  2       32 GB  [▶]            │
│  sft_14b_qlora           qlora   16    1.0e-4  3       28 GB  [▶]            │
│  sft_32b_qlora           qlora    8    1.0e-4  2       48 GB  [▶]            │
│  sft_vlm_7b_qlora        qlora   16    1.0e-4  3       35 GB  [▶]            │
│  sft_code_7b_lora        lora    32    3.0e-5  3       20 GB  [▶]            │
│  sft_finance_7b_lora     lora    16    1.0e-5  5       16 GB  [▶]            │
│  sft_science_7b_lora     lora    16    1.0e-5  3       16 GB  [▶]            │
│  sft_router_light_1_5b   lora    16    3.0e-5  4        6 GB  [▶]            │
│  sft_router_light_3b     full    —     2.0e-5  3       12 GB  [▶]            │
│  ──────────────────────────────────────────────────────────────────────────  │
│  rlsl_grpo_7b            grpo/rft —     5.0e-6  1       22 GB  [▶]            │
│  rlsl_grpo_14b           grpo/rft —     5.0e-6  1       62 GB  [▶]            │
│  rlsl_grpo_14b_true      grpo/grpo —    5.0e-6  1       62 GB  [▶]            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 面板 7 — Settings

### 9.1 需求

- 显示/编辑路径与端口(env 覆盖)
- fusion-mlx 健康状态(`trainer.info.mlx_alive`)
- GPU 内存水位
- 清理 cache

### 9.2 高保真原型

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Trainer › Settings                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  Engine                                                                [⟳]   │
│   fusion-mlx   ● alive   port 11432   models loaded: 2                      │
│   [Start fusion-mlx]  [Stop]  [Open logs]                                   │
│                                                                              │
│  Paths                                                                       │
│   model_hub_path   ~/fusion-model-hub              (env FUSION_MODEL_HUB)   │
│   cache_path       ./runtime_cache                  (env FUSION_TRAINER_CACHE)│
│   adapters (mlx)   ~/.fusion-mlx/adapters           (read-only ref)          │
│                                                                              │
│  Port                                                                        │
│   mlx_port  [11432 ▾]   11432=gateway  11434=direct (env FUSION_MLX_PORT)   │
│                                                                              │
│  Hardware                                                                    │
│   GPU mem   15.8 / 96 GB   unified memory   M5                               │
│   [Clear runtime_cache]  (保留 adapters,清 staged_data/checkpoints 临时件)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. 实施分工与交付件

### 10.1 本仓库(fusion-trainer)— Task 3,可改

新增 `fusion_trainer/run_manager.py`:

- `RunManager`:异步启动 SFT/RLSL run(包装 `SFTTrainer.train`/`RLSLEngine.run`,
  `asyncio.to_thread`),`_RUNS` registry(run_id→状态+ring buffer+落盘 events)
- live progress:`on_progress` 回调 → 事件 append → `runs/{run_id}/events.jsonl`
  + `deque(maxlen=1000)`(对齐 `ExperimentLogger` 模式)
- API:`list_runs`/`run_status`/`run_progress(since_step)`/`stop_run`/
  `list_adapters`(扫 fusion-mlx `ADAPTERS_PATH`)/`list_datasets`(扫 hub)/
  `list_presets`/`info`
- fail-visible:MXFP8 阻塞、grpo backend 缺 #363 均显式提示,不静默降级
- 日志:每个 run 日志落 `runs/{run_id}/run.log`

测试:复用 `tests/test_sft_trainer.py` 的 `_patch_mlx_client` 模式 mock
`MLXFineTuneClient`;RunManager 用 tmp 目录 + 假 events。验证后清过程数据,
只留最终件 + 日志。

### 10.2 上游 fusion-agent-studio — Task 4,issue→PR(不动本仓库外代码)

扩 `TrainerDispatcher`/`TrainingService`:

- 新 RPC:§2.2 表全部 method
- `method=rlsl` 放行(现只 dpo/orpo/grpo)
- live progress:`trainer.runs.progress` 转 RunManager
- full TrainerConfig 透传(不止 model/preset/dataset)
- adapters/datasets/presets list 转发

### 10.3 上游 fusion-studio — Task 5,issue→PR

- `AppState.swift` registry 注册 `SidebarSection.trainer` + 7 Module
- `UpstreamService` 注册 fusion-trainer(经 agent-studio daemon,**不**直连)
- `IPCTrainerMethods.swift`:`call("trainer.*")` 封装
- `TrainerView.swift` + 子视图(Dashboard/NewSft/NewRlsl/Datasets/Adapters/
  Presets/Settings),按 §3-§9 原型
- 状态桥:`ObservableObject` + `@EnvironmentObject`(对齐现有模式,无 SwiftData)
- 复活现有 dead `TrainingView.swift`→真实 TrainerView

### 10.4 验收(Task 6)

- 本仓库:`pytest tests/` 全绿,ruff clean
- RunManager:mock 下 SFT/RLSL run 能启动/落 events/list/progress/stop
- 集成:agent-studio daemon 能调到 RunManager,live loss 进 Studio 曲线
- README/CLAUDE.md 更新 GUI 集成章节

---

## 11. 约束遵循

- 只改 fusion-trainer/;上游 issue→PR→落地 code 流程(Task 4/5)
- 4 空格缩进,无 docstring,默认带日志
- MXFP8/grpo backend 阻塞 fail-visible,不静默
- Rule 5:progress 数据流确定性,不经模型
- Rule 12:run 失败大声报错,不静默吞
- 测试验证后清过程数据,只留最终件 + 日志
- Rule 6:per-task token cap,到点停
