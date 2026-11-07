---
doc_id: fusion-trainer/troubleshooting
product: fusion-trainer
class: platform
entry: service
doc_type: troubleshooting
title: { zh: "fusion-trainer 故障排查", en: "fusion-trainer troubleshooting" }
audience: engineer
status: draft
version: 1.0.0rc23
updated: 2026-11-07
owner: @dahai
source_of_truth:
  - docs/troubleshooting.md
---

# fusion-trainer 故障排查 / Troubleshooting

训练本身在 fusion-mlx 进程内执行 —— 大多数训练失败先看 fusion-mlx 侧：`~/fusion/fusion-mlx/start.sh log -f` 与 `start.sh doctor`。trainer 侧每个 run 在 `{cache_path}/runs/<run_id>/events.jsonl` 落盘事件。

Training itself executes inside fusion-mlx — check the fusion-mlx side first: `~/fusion/fusion-mlx/start.sh log -f` and `start.sh doctor`. On the trainer side, every run writes events to `{cache_path}/runs/<run_id>/events.jsonl` (default cache `./runtime_cache`).

## 引擎连接 / Engine connectivity

**症状 / Symptom**: `create_job` 失败、connection refused、job 一直 pending。

1. 确认引擎存活 / Engine alive: `~/fusion/fusion-mlx/start.sh status`（PID、端口、已加载模型）。
2. 端口是否被改 / Port mismatch: trainer 默认 `11434`（fusion-mlx 直连）。若流量应走 fusion-gateway，设 `FUSION_MLX_PORT=11432`（gateway 会校验 client key 并注入上游凭证）。
3. 认证 / Auth: `MLXFineTuneClient` 用 `resolve_api_key(env_var="FUSION_MLX_ADMIN_TOKEN")` —— env 优先，其次 `~/.fusion-mlx/settings.json` 的 `auth.api_key`。401 时先确认这两处。
4. 模型未加载 / Model not loaded: `ft info` 与 fusion-mlx `/v1/models` 对照模型别名（如 `qwen2.5-7b-4bit`）。

## 内存预算被拒 / Memory-budget rejection

**症状 / Symptom**: 提交大模型训练被 fusion-mlx 拒绝（默认按物理内存 60% 预算，假设专用训练机）。

- 原因：同一引擎还在服务大模型。trainer-bridge 会显式计算 `memory_limit_gb`（85% 安全线 − 常驻 − 6 GB 余量，下限 8），直连 CLI 时没有这层。
- 处理：减小模型档位（7B->LoRA 而非 32B full）、换 QLoRA（`--method qlora`）、或停掉占用内存的推理任务后重试。各 preset 文件头部有预估内存（如 sft_7b 约 16 GB）。

## 发布失败 / Publish failures

**症状 / Symptom**: 训练成功但 published 标记为 false。

- 设计行为：publish 失败**告警但不阻塞**，exit 0。看日志中的 hub 错误详情。
- 检查 `FUSION_MODEL_HUB_URL`（默认 `http://localhost:11432`，需 fusion-gateway 存活）与 `FUSION_MODEL_HUB_API_KEY`。
- `FUSION_TRAINER_PUBLISH=1` 会强制开启发布，排查时注意它是否被环境残留设置。
- 事后补发 / Retroactive: `ft publish <run_id>`。

## 沙箱奖励不执行 / Sandbox reward refuses to run

**症状 / Symptom**: GRPO code-mode 直接拒绝执行模型输出。

- 这是刻意的门禁：code 奖励会执行模型输出，必须显式 `FUSION_CODE_SANDBOX_TRUSTED=1`。
- 后端选择：macOS 默认 seatbelt（`sandbox-exec`，禁网、受限读、超时杀进程）；非 macOS 回退 rlimit 并大声告警。可用 `FUSION_CODE_SANDBOX_BACKEND=seatbelt|rlimit` 强制。
- `--sandbox auto` 仅在数据集含 `tests`（code）或 `answer`（math）字段时选择对应奖励，且仍是 opt-in —— 无门禁时默认 `off`。
- 参见 / See: `fusion_trainer/rlsl/isolation.py`, `fusion_trainer/rlsl/sandbox.py`。

## 配置与训练参数 / Config & training parameter errors

- `--method dora` 不生效？旧 `use_dora` 字段已删除（死字段），必须用 `--method dora`。
- `use_mxfp8: true` + `method: full` 直接报错：MXFP8 映射到 8-bit QLoRA 路径，与 full（解冻基座）矛盾，fail-stop 是预期行为。
- VLM run 中 `method: qlora` 退化为普通 LoRA 并有警告：上游 `VLMFineTuneConfig` 无量化字段（fusion-mlx#797），这是明确记录的限制，不是 bug。
- CLI 没有的字段（`lora_alpha`、`warmup`、`weight_decay` 等）必须走 preset 或 YAML —— 显式 CLI flag 永远覆盖 preset。

## 训练控制台 (bridge) / Trainer console (bridge)

**401**: 所有数据路由需要 bearer token —— `export FUSION_TRAINER_BRIDGE_KEY=<token>` 后再 `./scripts/trainer-bridge.sh start`。仅回环绑定，无 token 一律 fail-closed。

**控制台刷新后训练消失？** 不会：run 是 bridge 进程内的后台线程，刷新页面不杀训练；bridge 重启才会。

**shell 侧边栏训练入口变灰**: bridge 每 15 秒向 fusion-gateway 心跳；entry 灰色即 bridge 未运行。重新 `./scripts/trainer-bridge.sh start`。模块注册用 K1-issued key，持久化在 `~/.fusion-trainer-bridge/web_module_key`（0600），丢失后重启 bridge 会重新注册。

**预设改了没生效**: `~/.fusion-trainer-bridge/presets.json` 每请求重读；若文件损坏/为空会回退内置预设并告警 —— 检查 JSON 合法性。

## checkpoint 选择 (eval-select) / Checkpoint selection

**bench 分数与预期不符**: 确认 fusion-bench >= 0.4.11（旧版不转发 `adapter_path`，分数反映基座模型）。
**bench 挂了训练会失败吗？** 不会：所有 checkpoint 行标 `unavailable`，记录 `fallback_reason`，按 val_loss 选择 —— 永不静默出错。选择明细在 adapters 目录旁的 `checkpoint_selection.json`。

## 稳定性排查 / Stability issues

连续多轮运行后行为异常（句柄泄漏、状态串场）：

```bash
FUSION_MLX_ADMIN_TOKEN=<key> ./scripts/load_test.sh          # 5 rounds default
ROUNDS=10 ./scripts/load_test.sh
FUSION_TRAINER_LOAD=1 FUSION_MLX_ADMIN_TOKEN=<key> \
    pytest tests/test_load_stability.py -m load -v
```

`load_test.sh` 首个失败轮次即非零退出；需已加载文本模型。

## CI / 测试跳过 / CI & skipped tests

ubuntu CI 上集成测试全部 skip：MLX 无 Linux wheel，live-engine 测试由 `FUSION_TRAINER_REAL=1` 门控，仅 nightly macOS arm64 runner 解锁（`gh workflow run nightly -R dahai80/fusion-trainer` 手动触发）。本地跑真实训练测试需本机 fusion-mlx 存活。

## 仍未解决 / Still stuck

- run 级详情: `{cache_path}/runs/<run_id>/events.jsonl`
- 审计轨迹: `{cache_path}/audit.jsonl`（`start_sft`/`start_rlsl`/`stop_run`/`publish_run`/`delete_adapter` 各一行）
- 上游引擎日志: `~/fusion/fusion-mlx/start.sh log -f`
- API 冻结面与版本行为: `docs/API_STABILITY.md`; 版本历史: `CHANGELOG.md`
