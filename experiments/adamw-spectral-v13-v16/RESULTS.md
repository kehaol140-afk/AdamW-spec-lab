# RESULTS — adamw-spectral-v13-v16（按时间顺序；完整表；不覆盖原始数值）

## R0 父 gate（只读基座）

`results/local_gate.json`：BASELINE_TUNING_UNRESOLVED，Δ=−2.12e-6，reasons=[NO_LOCAL_SIGNAL]，next=STOP。基线：V12 位级重放（`replay_audit_v12/`）。

## R1 V13（10-02）：测量完成

四窗口 COMPLETE（`readiness_audit_v13/results/window{0..3}_complete.json`）。关键测量：top-8 能量份额集中度 ≈21×（均匀基线 0.018）。成本 3,439,912 token-forwards（V14 findings historical_v13_budget 记录）。

## R2 V14（10-02）：rank-1 谱修正阴性

`mechanism_screen_v14/findings.json`：**status=SCALE_RANGE_UNRESOLVED**；数值校准 max|Δ|=2.51e-8。dev 筛查（128 docs，`raw/dev/` 128 文件）+ confirm（20，`raw/confirm/`）：谱 vs 标量点差 2.41e-6（CI 跨零）；fresh_gradient +1.10e-4、fresh_preconditioned +6.89e-5（下界为正）。**结论：有限 rank-1 谱修正无可靠独立增量；信号指向方向而非谱形。**

## R3 V15（10-02）：方向-视界单 seed 信号

`direction_horizon_v15/findings.json`：verdict=**BRANCH_SIGNAL_WITHIN_GRID**，numeric_resolution_actual=0.0。F Δ_H64=+0.0014509 CI[+0.000886,+0.002030]；P Δ_H64=+0.000972 CI[+0.000480,+0.001470]（dev 128 confirm；数据 `raw/eval/` 29 分支）。证据级：单 seed 3407。

## R4 V16-A1 + a1_select（10-10，服务器 GPU0 ~72 min）

36/36 完成（`raw/eval/a1_*__H{64,256}__development.jsonl` 72 文件 + `raw/steps/` + `raw/q_series/`）。三族最优均在 **(a=8, b=2) 角点** → RANGE_UNRESOLVED（幅度维度未解析）；b 效应 ~0.11 nats ≫ 方向效应 ~0.003。选优冻结（STATE.json:5-24，02:53:09）：

| 族 | 选优分支 | development H256 |
|---|---|---:|
| S | a1_S_a8_b2 | 5.434258 |
| F | a1_F_a8_b2 | 5.431488 |
| P | a1_P_a8_b2 | 5.431446 |

36 格全表见 `interpretation/V16_A1_SERVER_RESULTS.md` 与 `raw/eval/`。Windows 31 条旧数据（`raw_win_pilot_20261003/`）ARCHIVED_READ_ONLY，不参与选优。

## R5 V16-A2（10-10，~12 min）：方向归因成立

四条共同范数分支（`raw/eval/a2_*__H256__development.jsonl`；本仓 recompute CK V16_A2 复算一致，舍入 ≤3e-6）：

| 比较 | Δ（H256） | CI95 | 解读 |
|---|---:|---|---|
| F@qS − S 原生 | **−0.00266** | [−0.00346, −0.00183] | F 方向在 S 范数轨迹上仍优 |
| P@qS − S 原生 | **−0.00257** | [−0.00325, −0.00190] | 同 |
| S@qF − F 原生 | **+0.00294** | [+0.00207, +0.00377] | 换入 S 方向丢全部优势 |
| S@qP − P 原生 | **+0.00298** | [+0.00227, +0.00367] | 同 |
| （对照）原生 S−F | +0.00277 | [+0.00193, +0.00359] | — |
| 纯范数效应（S@qF−S） | +0.00017 | — | 幅度轨迹效应 ~1.7e-4，比方向效应小 16× |

H64 同向（F@qS −0.00755 [−0.00894,−0.00616]，≈原生差的 97%）。裁决：**方向效应保留 ≥96%（双向验证）；F/P 不可分（差 9e-5）**。完整判读 `interpretation/V16_A2_DECISION.md`。

## R6 V16-B（10-10，~3.6h）：从头 18 分支全表

dev H2048（STATE.json selections.b.all_dev_h2048；`raw/eval/b_*__H2048__development.jsonl`）：

| # | 分支 | NLL | 相对最佳 S |
|---|---|---:|---:|
| 1 | b_P1_a8_lr0.001 | **4.087912** | −0.1527 |
| 2 | b_F1_a8_lr0.001 | 4.118871 | −0.1217 |
| 3 | b_P5_a4 | 4.125185 | −0.1154 |
| 4 | b_F5_a4 | 4.202970 | −0.0376 |
| 5 | **b_S1_plain_lr0.001（最佳 S）** | 4.240565 | 0 |
| 6 | b_S6_tb0（目标块 β1=0） | 4.244823 | +0.0043 |
| 7 | b_S5_tb05 | 4.252784 | +0.0122 |
| 8 | b_S3_plain_lr0.006 | 4.259105 | +0.0185 |
| 9 | b_S7_ab05（全模型 β1=0.5） | 4.270198 | +0.0296 |
| 10 | b_P2_a8_lr0.003 | 4.279804 | +0.0392 |
| 11 | b_F2_a8_lr0.003 | 4.302745 | +0.0622 |
| 12 | b_P3_a8_lr0.006 | 4.304484 | +0.0639 |
| 13 | b_F3_a8_lr0.006 | 4.323032 | +0.0825 |
| 14 | b_S4_a8_b2_lr0.001（A1 角点） | 4.340627 | +0.1001 |
| 15 | b_S2_plain_lr0.003 | 4.355896 | +0.1153 |
| 16 | b_P4_lr0.0003 | 4.618405 | +0.3778 |
| 17 | b_F4_lr0.0003 | 4.618892 | +0.3783 |
| 18 | b_S8_lr0.0003 | 4.660342 | +0.4198 |

三问闭合：低 β1 不能解释（S5/S6/S7 全部 ≥ 最佳 S）；LR 0.001 非边界（外扩 0.0003 全部 +0.38~+0.42）；A1 角点配方不从头迁移（S4 +0.100）。路由输出 F_OR_P_SUSTAINED（recipes=S1+F1；P1 因 screen 实现缺陷被拒，`interpretation/V16_B_RESULTS.md` §3）。screen 复核：P1 4.126135（`raw/eval/b_P1_a8_lr0.001__H2048__screen.jsonl`）。

## R7 V16-C（10-10，~2.4h）：3-seed holdout

final_holdout（1024 docs，一次开封；`raw/eval/c_*__H4096__final_holdout.jsonl`；本仓 recompute **6 位小数精确**）：

| seed | S1 plain | F1 a8 | S−F | CI95 | CI97.5 |
|---|---:|---:|---:|---|---|
| 3408 | 3.631140 | 3.594864 | **+0.036276** | [+0.03229, +0.04043] | [+0.03172, +0.04105] |
| 3409 | 3.524625 | 3.430607 | **+0.094018** | [+0.08850, +0.09963] | [+0.08774, +0.10038] |
| 3410 | 3.642411 | 3.490917 | **+0.151494** | [+0.14284, +0.16038] | [+0.14172, +0.16163] |

均值 +0.0939 / 中位 +0.0940；3/3 同向；CI 单位=doc 级 bootstrap（1000 次，seed 9071），**非独立 seed CI**；seed 散布 +0.036~+0.151。PRELIMINARY_SEED_ROBUST 判据全部满足 → runner 终态 **COMPLETE_WITH_SIGNAL**。dev 轨迹（s3408 示例）：F−S 在 H512~H4096 = −0.128/−0.054/−0.048/−0.038（全视界领先、无短视界反转）。缺口：P1 跨 seed 未执行（screen 缺陷）。

## R8 V16-D（随 B 采集）：机制材料

`raw/anchors.jsonl`（72 行=18 分支×4 锚点×3 矩阵）+ SVD 36 次（仅 S 臂，冻结设计）。核心观察（`interpretation/V16_FINAL_MECHANISM.md`）：cos(g,A)≈0.24→0.40（S1）；S6 把 cos 推到 0.93 仍无增益；S1 的 A 谱头份额 0.194→0.070、σ₁ 37.2→26.8。

## R9 终局与勘误指针

verdict 链：findings.json/STATE.json（10-10 18:08 重生成）= COMPLETE_WITH_SIGNAL；旧 STOP_RESOURCE 文档与 Round-1 "+0.1144" 误读的对账见 [V16_FINAL_RECONCILED.md](V16_FINAL_RECONCILED.md)。原始 CSV/JSON 路径：本目录 `campaign_v16/raw/**`、`readiness_audit_v13/results/**`、`mechanism_screen_v14/raw/**`、`direction_horizon_v15/raw/**`（逐文件 SHA 见 `MANIFEST.sha256`）。
