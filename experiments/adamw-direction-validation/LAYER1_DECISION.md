# LAYER1_DECISION — Layer-1 冻结门判定（预注册规则，零事后调整）

日期：2026-10-11。协议 ADAMW_DIRECTION_VALIDATION_V1_20261011；判定输入 = `raw/layer1_analysis.json` + `layer1_gate.json`。

## 1. 判定（逐条对照 frozen_v1.json 原文）

### 主门 1：`F8 − S8 ≤ −1e-3 AND paired-bootstrap CI95 entirely below 0`（development H2048）
- 实测：Δ = **−0.204090**，CI95 = [−0.214854, −0.192917]
- **→ PASS**

### 主门 2：`P8 − S8 ≤ −1e-3 AND CI95 below 0`
- 实测：Δ = **−0.235050**，CI95 = [−0.246108, −0.223252]
- **→ PASS**

### 反转规则：`|S8−F8|<1e-3 AND |S8−P8|<1e-3 → DIRECTION_AND_SCALE_CONFOUNDED`
- 实测：0.204 与 0.235，均 ≫ 1e-3
- **→ NOT TRIGGERED**

### Layer-2 触发：`at least one primary comparison passes`
- 两个主门均通过
- **→ layer1_gate.json 已写入 {pass: true}；Layer-2 获得代码级解锁资格（仍需用户授权）**

## 2. 判定的证据等级声明（预注册边界，逐字遵守）

- 本判定基于**单 seed（3407）、2048 步、development 集**的开发级证据；
  CI 仅为文档抽样误差，不构成训练 seed 级推断
- 复刻门 3/3 位级通过（S1/F8/P8 vs V16 对应臂，2048 步 × 9 字段）→ 结果与 V16 冻结语义同源可比
- 不得据本结果宣称"方向修正通用有效"；该问题属于 Layer-2（seeds 3411–3413 × 4096 步 × holdout_v2 一次开封）

## 3. 对 V16 历史结论的修订状态

- V16 缺口（从未运行 S(8,1) 对照）已由 dv_S8 补齐
- Layer-1 证据方向：**V16 F/P 的收益不能由 8× 幅度解释（幅度单独为负 +0.082），方向修正是必要成分**；
  0.20–0.24 的匹配尺度优势为单 seed 开发级幅度，待 Layer-2 确认
- 未触发反转规则 → 无需按 DIRECTION_AND_SCALE_CONFOUNDED 修订 V16 归因

## 4. 停止点

按授权边界：**本层结束即停**。Layer-2 不自动执行；等待 LAYER2_GPU_GATE.md 所列条件 + 正式授权。
不修改 V16 / SpecMuon P1 任何冻结产物；holdout_v2 保持 MINTED_SEALED（哈希复核 3e0f7653… 未变）。
