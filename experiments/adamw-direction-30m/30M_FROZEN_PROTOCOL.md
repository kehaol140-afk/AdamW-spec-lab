# 30M_FROZEN_PROTOCOL — T1 冻结协议（执行前发布，机器载体 configs/frozen_30m_t1.json）

协议 ID：`ADAMW_DIRECTION_SCALE30M_T1_20261011`。状态：PREREGISTERED_FROZEN（2026-10-11，GPU smoke 标定后、首个 T1 run 启动前冻结）。冲突时以 JSON 为准。

## 0. 上游门（G0）状态

dv layer-1 已落地：F8−S8 = −0.204090 [−0.2149,−0.1929]、P8−S8 = −0.235050 [−0.2461,−0.2232]，双主门通过、反转未触发、复刻门 3/3 位级。→ 按 CROSS_SCALE_DECISION_GATE §1：**F/P 臂全量进入 T1，主问题 = 方向残余的规模迁移**。

## 1. 五类候选臂（全部纳入；13 runs，单 seed 3421，T=2048 步 = 16.78M tokens/run）

| # | run id | 族 | a | 层 | LR | 说明 |
|---|---|---|---|---|---|---|
| 1–3 | s30_S1_lr{3e-4,1e-3,3e-3}_L1 | S | 1 | L1 | 网格 | **A0 调优 AdamW 基线**（30M 独立 LR 建立） |
| 4–6 | s30_S{2,4,8}_lr*_L1 | S | {2,4,8} | L1 | lr* | 目标块尺度对照（30M 独立 a 网格，不照搬 a=8） |
| 7 | s30_F{a*}_lr*_L1 | F | a* | L1 | lr* | 当前梯度方向（范数=‖A_t‖ 精确匹配） |
| 8 | s30_P{a*}_lr*_L1 | P | a* | L1 | lr* | 二阶预条件方向（P 强制包含） |
| 9 | s30_S{a*}_lr*_L2 | S | a* | L2 | lr* | 层对照（深度对齐 ℓ=2） |
| 10 | s30_{胜者}{a*}_lr*_L2 | F 或 P | a* | L2 | lr* | 层对照（胜出族） |
| 11–13 | s30_M_lr{5e-3,1e-2,2e-2} | Muon | — | — | 网格 | 调优 Native Muon 参照（M 线接线，wd 0.02；不进门） |

共同点：同 seed 3421 初始化、同 token 流（游标 0）、同 batch 8192 tok/步、同 schedule（warmup128+cosine→0.1）、同 clip 1.0、同评价（development 256，视界 {512,1024,2048}）。

## 2. 选择规则（冻结；只用 development H2048 token 加权 NLL）

- **lr\***：3 点最小；差距 <1e-4 视为并列 → 取较小 LR；最优落网格边界 → 记 `RANGE_UNRESOLVED_LR`（不自动外扩）
- **a\***：{2,4,8} 最小；若 |NLL(best)−NLL(a=4)|<1e-3（平坦）→ a\*=4；best=a=8 边界 → 记边界标志（外扩 {12} 须新授权，不自动）
- **胜者族**：F-a\* 与 P-a\* 的 dev H2048 较小者（并列取 F，保守）
- **Muon 最优**：3 点最小；best=2e-2 边界 → 记 `RANGE_UNRESOLVED_MUON`（参照臂，不影响门）

## 3. 门与停止条件（沿用 CROSS_SCALE_DECISION_GATE §3，零调整）

- **T1 主门**：F-a\*−S-a\*（同 lr\*/同层/同 a\*）dev H2048 ≤ −1e-3 且配对 bootstrap（1000 reps，seed 9071）CI95 全负；或 P-a\* 同款
- **不被 a 网格吸收**：S-a 曲线在 a\* 邻域平坦后差值仍存
- **层一致性**：L-depth 对照与 L-index 同号，或记 `LAYER_POSITION_CONFOUNDED` 并分层表述
- **停止**：F/P 对 S-a\* 差 <1e-3 或 CI 跨零 → `NO_DIRECTION_TRANSFER_AT_30M`；网格边界穷尽 → `RANGE_UNRESOLVED`；任何非有限值 / zero_norm_fallback → 全停报告

## 4. 记录口径（全臂）

逐步：eta / loss_sum / token_count / nll / A_norm / D_norm / q_t / opt_step_seconds / wall_cum；
锚点（步 204/512/1024/1536）：目标块三矩阵 cos(g,A) / cos(g,m) / σ₁ / 头部能量份额（含去重守卫）；
终点：param_fingerprint（初始+终点）、每臂 train/eval/optimizer 秒分列、端点 checkpoint（~120MB×13，不入证据包）。

## 5. 预算（smoke 标定，授权窗口 2 GPU-h）

| 项 | 值 |
|---|---|
| 训练 | 10 × 6.99 min（AdamW 族）+ 3 × 5.18 min（Muon）= 85.4 min |
| 评估 | 13 runs × 3 视界 × 256 docs ≈ 13 min |
| **合计投影** | **≈ 1.64 h ≤ 2.0 h 授权窗口**（余量 ~18%）；超预算 +20% → shrink rule 报告后停 |
| 已消耗 | smoke 0.005 h |

## 6. 禁止事项

跨模型 NLL 绝对值比较；事后更换评价集/扩网格；修改 V16/M 线/SpecMuon 文件；旧 Muon checkpoint 作共同初始化；本轮多 seed / holdout_v2 开封 / 100M。
