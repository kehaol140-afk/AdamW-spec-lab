# RUNNER_READINESS — 运行器就绪报告

## 1. 代码组成（全部新建于本项目，零 V16 写入）

| 文件 | 作用 | 状态 |
|---|---|---|
| `src/dv_core.py` | vendored 数学内核：`n_a / schedule_eta / apply_step / param_fingerprint / eval_docs / token_weighted / paired_bootstrap` | ✅ 完成 |
| `src/dv_runner.py` | layer-1/2 串行 runner：scratch 分支（seed 流）、逐步行日志（含 q_t/opt_step_seconds）、锚点机制记录（含全臂 SVD）、eval、端点保存、layer-2 双重封锁（gate 文件 + holdout_v2 状态） | ✅ 完成（GPU 未执行） |
| `configs/frozen_v1.json` | 预注册（臂/门/反转规则/seeds/禁项） | ✅ 冻结 |
| `tests/test_dv_cpu.py` | 8 项 CPU 单元测试 | ✅ **8/8 PASS**（fr_minimind, CPU-only） |

## 2. Vendoring 增量（dv_core vs campaign_v16/core.py，全部列出）

- **D1** `apply_step` 删除 `begin_optimizer/bump`（V16 账本写入）→ 改捕 `meta["opt_step_seconds"]`
- **D2** `torch.cuda.synchronize()` 加可用性守卫（CPU 测试/CPU 执行）
- **D3** `eval_docs` 输出路径参数化（本项目 raw/）；删 resource/count_forward/bump；device 参数化
- **D4** betas/eps 由 PCFG 读取改为内联字面量（与 config/local_v1_lowmem.json 数值相等：(0.9,0.95)/1e-8）

其余逐行保持原文（含 q_override 注释与位移校正数学）。**任何触碰更新语义的改动 = 零**。

## 3. 测试证据（2026-10-11 实跑）

- `test_S11_is_bit_native`：S(1,1) 与原生 AdamW 逐步位级一致（vendored 复验 V16 冻结测试语义）✅
- `test_S8_displacement_math`：S(a=8) 目标块净位移 = −8ηA（含衰减 (1−ηλ)），其他参数 b=1 原生 ✅
- `test_F_P_direction_and_norm`：‖D‖=‖A‖（fp32 cast 舍入内）、q_t>0 ✅
- `test_q_override_exact`：q_override 后位移范数精确（≤1e-8，V16 A2 实测 ≤2.1e-9 同级）✅
- `test_schedule_eta` / `test_bootstrap_and_tokenweighted`：调度与统计逐值 ✅
- `test_eval_docs_split_load_and_guard`：划分加载 + EVAL_WITHOUT_STATE_ORIGIN 守卫 ✅
- **`test_V16_ledger_untouched`：执行 vendored apply_step 前后 V16 budget_ledger.json 字节不变** ✅（隔离性硬证据）

## 4. GPU 前验证门（开跑时自动执行，零额外成本）

1. **复刻位级门**：dv_S1/dv_F8/dv_P8 与 V16 b_S1/b_F1/b_P1 的 step 日志（eta/loss_sum/q_t）应逐步一致（同平台/seed/数据流/代码语义）；任何不一致 → 停
2. 端点身份：param_fingerprint + T + cursor 落盘；eval 行带 state_id=scratch_seed{N} + param_fingerprint
3. layer-2 双锁：`layer1_gate.json{pass:true}` **且** `holdout_v2_manifest.json{status:MINTED_SEALED}`，否则 runner 拒跑（代码内硬检查，非纪律约定）

## 5. 已知限制

- runner 无断点续跑的滚动 checkpoint（2048/4096 步单臂 ≤30 分钟，失败整臂重跑成本可接受；与 V16 的 128 步滚动相比是简化，非语义差异）
- 锚点 SVD 对全部臂开放（V16 仅 S-plain）——成本 ~ms 级，换取 F/P 臂的谱可观测性；SVD 次数无协议上限约束（新项目自有预算）
- 资源门控为最小实现（无 PauseBudget/温度监控）——运行于实验室 GPU0 空闲窗口，人工监督；如需无人值守长跑可后续补
