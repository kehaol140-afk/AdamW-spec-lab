# V16_FINAL_RECONCILED — V16 战役最终对账（2026-10-12 新增，不修改任何旧文件）

本文件解决三类历史不一致：(a) 旧人工报告停留在 STOP_RESOURCE；(b) `findings.json` stats 为空；(c) Round-1 审计（2026-10-12 上午）对路由数字的误读。**所有已有正式判读文档一律引用不重复**：`interpretation/V16_{A1_SELECTION_AUDIT,A1_SERVER_RESULTS,A2_DECISION,B_RESULTS,C_SEED_RESULTS,FINAL_MECHANISM,CROSS_PLATFORM_CHECK,NEXT_GATE}.md`（复制自 `audits/adamw_a_v16_exec_20261009_171502UTC/`，SHA 见该目录 SHA256SUMS*）。

## 1. 抃账一：报告时间线（哪些文档是权威的）

| 文档 | 写作时点 | 状态 | 权威性 |
|---|---|---|---|
| `campaign_v16/V16_FINDINGS.md` | 10-03 Windows 迁移包 | **STALE**（仍写 STOP_RESOURCE） | 仅历史 |
| `V16_FINAL_REPORT.md`（项目根） | 10-03 | **STALE**（PRELIMINARY，A1 31/36 态） | 仅历史 |
| `campaign_v16/findings.json` / `STATE.json` | 10-10 18:08 finalize 重生成 | verdict=COMPLETE_WITH_SIGNAL；`stats:{}` 空（见 §3） | 机器态权威 |
| `audits/.../V16_A2_DECISION.md` 等 8 篇 | 10-10 完赛后审计层 | **现行权威判读** | ✅ |
| 本文 | 10-12 | 对账与勘误 | ✅ |

## 2. 抃账二：Round-1 审计两处数字误读（ERRATA）

### 2.1 「P1 对 S1 点收益 +0.1144 命中路由规则 1」——**错误，撤回**

- 事实链（`campaign_v16/run.py:633-671` plan_route_b 逐行核对）：
  - `gain_dev(F1) = dev[S1] − dev[F1] = 4.240565 − 4.118871 = +0.121694 ≥ 1e-4`，且 F1 无 screen 文件 → `gain_screen=None` → 旁路通过 → **F1 入选 C**
  - `gain_dev(P1) = +0.152653`，但 P1 有 screen 值 → `gain_screen = screen.get(S1, 0) − screen[P1] = 0 − 4.126135 = −4.126 < 0` → **P1 被拒出 C**（`V16_B_RESULTS.md` §3 已完整披露该实现缺陷）
- Round-1 写的 "+0.1144" = dev(S1)=4.240565 − screen(P1)=4.126135，是**跨 split 混合量，无任何协议地位**；正确引用值为 dev 口径 **P1 −0.1527 / F1 −0.1217**（相对最佳普通 S，`V16_B_RESULTS.md` §1，本次已从 `raw/eval/b_*__H2048__development.jsonl` 逐文档复算一致：4.087912/4.118871/4.240565）。
- 复算证据：`recompute/outputs/V16_A2_recompute.json`（A2 四臂）与本节 B 表（本次脚本聚合，见 `recompute/`）。

### 2.2 「A2 无书面裁决标签 / interpretation missing」——**不准确，修正**

- `findings.json:28 "stats": {}` 确为空（finalize 的 `c_` 前缀 bug，`V16_C_SEED_RESULTS.md` 披露）；**但 A2 的完整书面裁决早已存在**：`interpretation/V16_A2_DECISION.md`（含主表、5 组配对 CI、裁决三点与边界）。Round-1 仅查看了 campaign_v16/ 目录内文件而未查 audits，结论"跑完但无书面判读"应修正为"机器 stats 块空缺，人工判读完整存在于 audits（已复制入本仓 interpretation/）"。

## 3. 抃账三：每阶段结论与证据指针（一次看清）

| 阶段 | 裁决 | 关键数值（复算核对） | 权威文档 | 原始证据 |
|---|---|---|---|---|
| A0/A1 36 条 | RANGE_UNRESOLVED（角点 (8,2)） | 三族 H256 dev 5.43426/5.43149/5.43145 | interpretation/V16_A1_SELECTION_AUDIT.md | raw/eval/a1_*__H{64,256}__development.jsonl |
| A2 4 条 | 方向效应保留 ≥96%，幅度轨迹效应 1.7e-4 | F@qS−S=−0.00266 CI[−0.00346,−0.00183]（本次复算 −0.002658） | interpretation/V16_A2_DECISION.md | raw/eval/a2_*__H256__development.jsonl |
| B 18 条 | F_OR_P_SUSTAINED；低β1/LR 边界/角点配方三问全闭合 | P1 dev 4.087912 / F1 4.118871 / S1 4.240565 | interpretation/V16_B_RESULTS.md | raw/eval/b_*__H2048__{development,screen}.jsonl |
| C 6 条 | PRELIMINARY_SEED_ROBUST；runner COMPLETE_WITH_SIGNAL | S−F = +0.036276/+0.094018/+0.151494（本次复算 6 位小数一致）；CI95 全离 0 | interpretation/V16_C_SEED_RESULTS.md | raw/eval/c_*__H4096__final_holdout.jsonl |
| D 机制材料 | 收益=「方向替换×幅度放大」组合，非新鲜度/对齐度 | cos(g,A)≈0.4；S6 cos 0.93 无增益；A 谱头份额 0.19→0.07 | interpretation/V16_FINAL_MECHANISM.md | raw/anchors.jsonl（72 行）+ budget_ledger（SVD 36/48） |

## 4. 已知实现层缺陷（全部已披露、未修改冻结代码）

1. route_b screen 缺席基线默认 0 → P1 误拒出 C（§2.1）
2. finalize() stats 块 `c_` 前缀缺失 → findings.stats 空（审计层已用同一冻结例程补算）
3. Windows 中断残留 findings 首次跳过（DECISIONS.md:24-27 勘误留痕）
4. STATE.stop_note "27/36" 陈旧（部署审计已勘误）
5. `V16_FINDINGS.md` 未随完赛更新（本文件 §1 表替代指引）

## 5. 证据边界（与 FINAL_MECHANISM Q6 一致，不得夸大）

单目标块（L1 FFN down_proj）、单规模 4.66M、单数据集；P1 跨 seed 未执行；a=8 仍为网格边界；holdout 已开封（本轮封存纪律终结）；PRELIMINARY——不构成对更大规模的主张。规模迁移证据见 `experiments/adamw-direction-validation/` 与 `experiments/adamw-direction-30m/`（均为 dev 级单 seed）。
