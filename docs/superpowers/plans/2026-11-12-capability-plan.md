---
doc_id: fusion-trainer/capability-plan-2026-11
product: fusion-trainer
class: platform
entry: service
doc_type: guide
title: { zh: "能力补齐与超越计划（2026-11）", en: "Capability catch-up and leapfrog plan (2026-11)" }
audience: engineer
status: draft
version: 1.0.0rc25
updated: 2026-11-12
owner: "@dahai"
source_of_truth:
  - docs/superpowers/plans/2026-11-12-capability-plan.md
---

# fusion-trainer 能力补齐与超越计划（2026-11）

输入：`~/fusion/insight/fusion-trainer-evolution-2026-11.md`（业界洞察 + T1-T9 差距表）。
基线：v1.0.0rc25（2026-11-10），~14.4k LOC，697 测试，纯 HTTP 委托制。
本文档 = 差距 → 可执行交付的映射：每项给出仓库内交付物、上游依赖（issue 路由）、验收标准、里程碑。

## 0. 总原则（先立防线，再谈能力）

1. **委托制不破** — trainer 永不 import mlx/mlx-lm/mlx-vlm；梯度循环只在 fusion-mlx。一切新能力先问"能否走 HTTP job API"，不能就先提上游 issue。
2. **fail-visible** — 不伪装成功、不静默降级（MXFP8/VLM QLoRA 有先例）；新功能默认继承。
3. **消费形态冻结在三种** — in-process（RunManager）/ CLI / bridge HTTP；新消费方一律走 bridge，接口面不再发散（T8 的解法是"不再新增"，不是"再统一一次"）。
4. **数据可信可审计** — 任何 trajectory/rollout → dataset 的 reward 映射必须显式、确定性（Rule 5）、入 manifest/审计 JSONL；脏数据宁可拒收不可静默进训练集。
5. **bench 选优语义不反转** — bench 分优先、val loss 显式回退、last checkpoint 兜底，契约钉死在测试里。
6. **nightly 真训练回归保留** — 委托链断裂必须被真梯度测试抓住。

## 1. 阶段一：作业可靠性（0-3 个月，对应 T1/T2 + 内存护栏）

目标：长 RL run 不再"崩溃即全废"，双宿主派活不再互撞。

### 1.1 T1 — RL 作业 resume（🔴 最高优先）

- **仓库内**：
  - `MLXFineTuneClient` 增 DPO/ORPO/GRPO job 的 `init_adapter_path` 透传（DPO 已有 #1149 链式路径，复用其通道）
  - `wait_for_dpo_job`/`wait_for_grpo_job`/`wait_for_orpo_job` 接入 `_raise_if_job_lost` 404 快失败 + `interrupted` 结果（SFT #65 已有模板，逐一对齐）
  - `job_state.json` 落盘扩展：`job_type` + 每类 RL 的 step 语义
  - 编号 checkpoint 发现复用 `eval_loop.list_checkpoints`（不重写第二份）
- **上游依赖（先提 issue）**：
  - fusion-mlx：DPO/ORPO/GRPO job 端点接受 `init_adapter_path` 并产出编号 checkpoint（GRPO 需按 step 保存 policy adapter）
- **验收**：GRPO 千步级 run 中途杀 fusion-mlx → 重启后 `ft rlsl --resume` 从最新 checkpoint 续跑，单测 404 快失败 + interrupted 报告 + 续跑 e2e（nightly 真训练覆盖）。
- **里程碑**：M1（~4 周）。

### 1.2 T2 — 作业队列与双宿主互斥（🔴）

- **仓库内**：
  - `RunManager` 增 pending 队列（优先级 + FIFO），`start_sft/start_rlsl` 在有活跃 run 时入队而非拒绝
  - 跨进程互斥：`{cache_path}/run.lock`（fcntl）+ 锁持有者心跳，agent-studio in-process 与 bridge/cowork 派活汇合到同一把锁
  - 队列状态暴露：`run_manager.status`/bridge `/api/runs` 显示 queued 前缀
- **验收**：两个宿主同时派活 → 一跑一排队，无互撞；进程崩溃锁自动释放（心跳过期）；单测覆盖锁竞争与心跳过期。
- **里程碑**：M1-M2（~6 周，与 1.1 并行不同模块）。

### 1.3 统一内存护栏

- **仓库内**：跑前 `psutil` 预检（可用内存 vs `memory_estimate_gb`，presets 已有估计）→ 不足时 fail-visible 给降档建议（`--grad-checkpoint` / `--max-seq-length` / quant_bits），不自动改参
- **上游依赖**：fusion-mlx job API 接受 `memory_limit_gb`（#83 DPO 已加过 `memory_limit_gb`，推动泛化到 SFT/GRPO 端点）
- **验收**：27B-8bit 无护栏参数直接提交 → 明确报错 + 建议参数；带建议参数 → 通过。
- **里程碑**：M2。

## 2. 阶段二：实验与数据治理（3-9 个月，T3/T4/T7/T5）

### 2.1 T3 — 实验横向比较（🟡）

- **仓库内**：bridge GUI 升级——跨 run 曲线叠加（events.jsonl 已流式落盘）、配置 diff 视图（run_meta.json 结构化对比）；导出 CSV/Markdown 摘要
- **明确不做**：optuna 级 sweep（消费形态收敛原则；网格搜索留给外部脚本调 CLI）
- **验收**：bridge 上任选 2+ run 叠加 loss 曲线 + 并排配置 diff。
- **里程碑**：M3-M4。

### 2.2 T4 — 数据集 lineage 落地（🟡）

- **仓库内**：统一 lineage 记录格式——from-eval / from-rollouts / from-dpo sidecar 已有 sha256，补"变换链"字段（source → transform → output，飞轮每轮 `ft flywheel record` 自动写入）；`ft datasets info <path>` 打印完整 lineage
- **上游依赖**：fusion-model-hub 数据集版本 API（登记 lineage 元数据，hub 为唯一登记处，trainer 只生产记录）
- **验收**：任一训练 run 可回溯"哪份 export/sink → 哪次变换 → 哪个数据集文件"，hub 端可查。
- **里程碑**：M4-M5。

### 2.3 T7 — reward-hacking 监控（🟡，业界失败模式论文直转指标）

- **仓库内**（全部基于已有 events 流，Rule 5 确定性计算）：
  - reward 分布漂移：滑动窗口均值/方差告警阈值
  - 长度膨胀：completion 长度分位数逐 step 趋势
  - 格式钻营：`<answer>` 等格式 token 占比异常
  - 指标进 `metrics.prom`（#enterprise-3 通道复用）+ bridge 面板红黄绿
- **验收**：注入人工漂移的 events 序列 → 三类指标各自触发告警；nightly 正常 run 不误报。
- **里程碑**：M4-M5。

### 2.4 T5 — 多 Mac 分布式编排面（🟡）

- **仓库内**：编排层 only——节点清单/分片配置生成/聚合进度展示；训练本体经 fusion-mlx 的 mx.distributed job API
- **上游依赖**：fusion-mlx 暴露分布式训练 job API（mlx-lm `mx.distributed` 已存在，需要 serving 化封装）
- **前置**：T1 完成（分布式 run 更需要 resume）
- **验收**：2 Mac LoRA run，trainer 侧全程可见节点/进度/最终 adapter。
- **里程碑**：M6+（本期最末，依赖最重）。

## 3. 阶段三：飞轮深化与超越（9-18 个月，T9/T6 + 差异化加固）

### 3.1 超越主线：把"数据飞轮"从资产变成护城河

业界空位（insight §3）：轨迹→训练数据无平台内建。fusion 已有最完整闭环，超越打法 = 三个入口收全 + 一个出口验证：

1. **入口一（已通）**：fusion-eval 轨迹 → `ft datasets from-eval`（#76）
2. **入口二（已通）**：gateway rollout tap sink → `ft datasets from-rollouts`（#86，rc25；tap 本体在 fusion-gateway#608 推进）
3. **入口三（本计划新增）**：cowork DAG 运行结果 → 直接产 multi-arm export（passed/failed 兄弟），打通 DPO 挖矿数据源（#83 证明 DPO 卡在数据厚度不在算法）
   - 交付：`ft datasets from-cowork <flow-export>`（schema 与 from-eval 对齐，复用 eval_ingest 挖矿）
4. **出口（本计划核心超越项）**：**bench 任务即 GRPO 验证器**——fusion-bench 任务库直接作为 GRPO reward 源（verifiers 方向，eval_loop 管线已有 90%）
   - 交付：`RewardServer` 新后端 `bench`：prompt 映射 bench task → bench task API 打分 → 返回 reward；RL 闭环从"手工数据集"升级为"任务库无限出题"
   - 防线：打分走 bench HTTP，确定性部分留在本地；bench 不可用 fail-visible 降级为 sandbox（与 eval_loop 回退语义一致）
   - 上游依赖：fusion-bench 任务 API 支持程序化逐题打分（v0.4.11 adapter 透传已验证通路）
- **验收（出口）**：GRPO run 用 bench 后端 reward 完成 + eval 选优提升可复现；bench 停机 → 显式降级 sandbox 并告警。

### 3.2 T9 — rollout 效率联动（与 fusion-mlx 演进档合拍）

- trainer 侧零代码：fusion-mlx 落前缀缓存/continuous batching 后 GRPO rollout 自动受益
- 仓库内：rollout 吞吐指标（samples/s）进 events + 监控面板，量化验证受益；评估"专职 rollout 模型配置"建议文档
- **里程碑**：跟随 fusion-mlx 发版，被动跟进 + 主动测量。

### 3.3 T6 — 量化训练跟进

- MXFP8：上游（fusion-mlx fp8 dtype）解锁后，把现有 flag 从"fail-stop"切到真通路（#402/#425 历史路径重启）
- VLM quantize 字段：向 fusion-mlx 提 issue 推动 `VLMFineTuneConfig` 增加 quantize，消灭 QLoRA 静默降级
- **里程碑**：跟随上游，issue 先行。

## 4. 里程碑总表

| 里程碑 | 内容 | 依赖上游 | 预计 |
|--------|------|----------|------|
| M1 | T1 RL resume + T2 队列互斥 | fusion-mlx RL checkpoint API issue | 0-6 周 |
| M2 | 内存护栏 | fusion-mlx memory_limit 泛化 | 4-8 周 |
| M3-M4 | T3 实验对比 + T7 hacking 监控 | 无 | 3-5 月 |
| M4-M5 | T4 lineage + 入口三 from-cowork | fusion-model-hub 版本 API | 4-6 月 |
| M5-M6 | bench-as-verifier（超越核心） | fusion-bench 打分 API | 6-9 月 |
| M6+ | T5 分布式编排 | fusion-mlx 分布式 job API | 9 月+ |
| 跟随 | T9 rollout 效率 / T6 量化训练 | fusion-mlx | 持续 |

## 5. 上游 issue 清单（先提 issue 再落代码）

| # | 仓库 | 议题 | 服务于 |
|---|------|------|--------|
| U1 | fusion-mlx | RL job（DPO/ORPO/GRPO）编号 checkpoint + init_adapter_path | T1 |
| U2 | fusion-mlx | memory_limit_gb 泛化到全部 fine-tune 端点 | 内存护栏 |
| U3 | fusion-model-hub | 数据集版本/lineage 登记 API | T4 |
| U4 | fusion-mlx | 分布式训练 job API serving 化 | T5 |
| U5 | fusion-bench | 程序化逐题打分 API（GRPO verifier 用） | 超越出口 |
| U6 | fusion-mlx | VLMFineTuneConfig quantize 字段 | T6 |
| 已提 | fusion-gateway#608 | rollout tap middleware + sink | 入口二 |

## 6. 执行纪律

- 每项独立分支 + PR + CI 绿 + nightly 覆盖后才合入；🔴 项（T1/T2）优先级永远高于 🟡。
- 上游 issue 未关前，仓库内只做 mock 可测的编排/数据面，不做端到端承诺。
- 每 M 结束跑一次 flywheel 真实 round 验证回归不劣化（bench 选优 + eval set 不变）。
- 本计划文档随里程碑推进更新状态（status: draft → in-progress → done per section）。
