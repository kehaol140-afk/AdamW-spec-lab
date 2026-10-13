# RESULTS — muon-ns-allocation（按时间顺序）

## R1 P0/P1（09-29）：单组重分配全阴性

`outputs/p0_p1_v1_1/verdict.json:30-37` Q1=NO_LOCAL_SUPPORT；`check_rows.csv`（396 行）：28 单组最大 |B|=3.8e-5、0/4 同向、21 负 7 弱正 ≤1e-6 → **0/28 达 δ=1e-5**。幅度对照（`amplitude_controls.csv`）：**全局更新 ×0.5：s512 +3.4235e-4（3/4 batch 同向）/ s1536 +5.8e-5**；×2 为负。即唯一强信号是幅度类，比一切谱候选大一个量级。

## R2 R0/R1（09-29）：统一时钟成立、结构零增量

`R0_GATE.json:68` gate_pass=true（maxdiff 1.49e-8）。R1 配对（`paired_effects.csv` 全 16 行，512 状态关键值）：

| 比较 | mean（512） | 判定 |
|---|---:|---|
| 1_global_gain（a0.5 vs a1.0） | **+9.958e-4** | 全局幅度信号 |
| 2/3 组倍率（b1.25 vs b1.0） | −7.85e-6 | 组级零/负 |
| 4/5/6 匹配结构 NS3 | −3.79e-5 | 结构无增量 |
| Z_NO_MUON_DATA（诊断） | +1.72e-4 | — |
| 1536 状态 global_gain | +1.304e-4 | 4/4 同向 |

verdict.json:3 GLOBAL_ONLY_LOCAL_SIGNAL=true。

## R3 R2（09-29）：256 步延续与一步符号翻转

`r2_amplitude_continuation_v1/paired_effects.csv`（12 行原始）：

| 状态 | t | G_MH（NS3 匹配） | G_AH | G_ratio |
|---|---|---:|---:|---:|
| 512 | 1 | **−1.463e-5** | +1.507e-4 | −1.653e-4 |
| 512 | 256 | +0.038273 | **+0.072974** | −0.034701 |
| 1536 | 1 | +1.936e-4 | +1.266e-4 | +6.705e-5 |
| 1536 | 256 | +0.003128 | +0.006579 | −0.003451 |

**一步筛查符号翻转实证**（512 MH：一步 −1.5e-5 → 256 步 +3.8e-2）。

## R4 R2F（09-29→30）：B_L 解释 101.7%，AH 关闭

`lrwd_attribution.csv`（本仓 recompute CK MSA_attribution 精确复算）：

| 状态 | V_N | V_AH | V_L | B_L=V_N−V_L | B_L/AH 增益 |
|---|---:|---:|---:|---:|---:|
| 512 | 2.924853 | 2.851879 | 2.850637 | **0.074216** | **101.70%** |
| 1536 | — | — | — | — | 100.5%（报告值） |

AH 状态优势在共同尾部反转（R_N<0）；D 未触发（D_required=false）。勘误：首版 wrong-decay → r2f_corrected_v1 复核结论不变（wrong_vs_corrected.csv + R2F_CORRECTED_REPORT_CN.md）。裁决 `AH_BRANCH_CLOSURE_CN.md:3`：**CLOSED_NO_AH_SPECIFIC_INCREMENT_THIS_SETTING**。

## R5 R2F-NL（09-30）

`nl_effects.json`：G_NL s512 +5.952e-2 / s1536 +1.837e-3；AH 相对 L |Δtail|≤8e-4 < τ=1e-3 → 无实用增量。

## R6 终局（09-30）

`NEXT_RESEARCH_DECISION_CN.md:46`：**HOLD_NO_NEW_TOPIC**。横断教训四条（见 docs/05）。成本：P1 14 分支各 ~20 min GPU；R2/R2F/R2F-NL 75 个 .pt 的 SHA 索引 AH_ASSET_INDEX.csv（17 行）。

## 证据等级说明

全部轮次=单父轨迹（3101）两恢复点、V_Q 256 docs 筛查级；无跨 seed；结论为「本设置内阴性」而非普适否定（SOURCE_BLOCKED 论文未核对——语义边界已冻结）。
