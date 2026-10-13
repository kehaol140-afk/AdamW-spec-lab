# F_P_S_MATCHED_BASELINE_PLAN — 同尺度匹配基线比较计划（预注册）

冻结载体：`configs/frozen_v1.json`（protocol_id ADAMW_DIRECTION_VALIDATION_V1_20261011，状态 PREREGISTERED_FROZEN_NOT_EXECUTED）。本文件是其人读摘要；冲突时以 JSON 为准。

## 1. 设计原则

- **唯一新自由度 = 目标块方向（S/F/P）× 目标块幅度 a**；其余一切（LR、schedule、数据流、优化器状态规则、评估、seed、步数）与 V16 B 冻结设置逐项相同
- 不复用、不修改 V16 任何文件；数学内核 vendored（dv_core.py，delta D1-D4 见 RUNNER_READINESS）
- 旧 final_holdout 已开封 → 永不用于新选优；layer-1 用旧 development/screen（dev 级，披露其历史）；layer-2 用待铸造的 holdout_v2

## 2. Layer 1：开发级短预算（2048 步，seed 3407，LR 0.001）

| 臂 | family | a | b | 身份 |
|---|---|---|---|---|
| dv_S1 | S | 1 | 1 | **V16 b_S1 复刻**（bitwise 验证门） |
| dv_S8 | S | 8 | 1 | **缺失格**（V16 从未运行；本计划核心） |
| dv_F8 | F | 8 | 1 | V16 b_F1 复刻 |
| dv_P8 | P | 8 | 1 | V16 b_P1 复刻 |
| dv_S4 | S | 4 | 1 | 可选中点（幅度单调性检查） |

- 评估：development H512/1024/2048（+screen 作次级）；逐步记录 q_t / A_norm / D_norm / opt_step_seconds；锚点（10/25/50/75%）记录 cos(g,A)/cos(g,m)/头部能量份额/σ₁（三矩阵，全部臂）
- **主比较（预冻结门）**：`F8−S8 ≤ −1e-3 且配对 bootstrap CI95 全负`；P8 同
- **反转规则（预冻结）**：若 |S8−F8|<1e-3 且 |S8−P8|<1e-3（H2048）→ 记 `DIRECTION_AND_SCALE_CONFOUNDED`，V16 归因修正为"方向及幅度组合收益不能独立归因于方向"
- **运行器验证门（零额外成本）**：dv_S1/dv_F8/dv_P8 的 step 日志应与 V16 b_S1/b_F1/b_P1 的 eta/nll/q_t 逐步一致（同平台同 seed 同数据流；容差=平台内位级，预期 bitwise）——不一致则停止并诊断
- Layer-2 触发：≥1 个主比较通过

## 3. Layer 2：独立 seed 长预算确认（4096 步）

- 臂 S1/S8/F8/P8（**P8 强制包含**——修复 V16 screen 事故）× seeds **{3411,3412,3413}**（已核 V16 registry/minimind 配置/M 线记录：3407-3410、2030、3101 均已占用，3411-3413 无冲突）
- warmup 256（同 V16 C）；数据缓存 8.39M ≤ 8.5M token（恰好覆盖，scratch 从 0）
- **holdout_v2**：≤1024 文档，仅取 untouched 池（1068 候选，见 NEW_EVAL_SPLIT_MANIFEST）；全部 12 终点身份锁定后**开封一次**，之后不得调参
- 推断单位 = 训练 seed：逐 seed 差值 + 均值/标准差/极差/同号率；文档 bootstrap 仅作固定模型下的评价误差，不冒充独立重复
- 旧 V16 3-seed 结果只作历史证据引用，不与新 holdout 合并

## 4. 明确不做（本轮及可见后续）

- a 网格不外扩（冻结 {1,4,8}；a=8 仍边界的事实作为 V16 遗留边界报告，不自动加测 a=16）
- 不引入 SVD/谱整形/新优化器候选（那是 SpecMuon/AdamSpec 线的事，且 AdamSpec 的 gate 之一正是本计划的结果）
- 规模/模块外推（30M、其他层、多块、Muon 几何比较）仅列为后续建议，见 NEXT_GPU_GATE §4

## 5. 失败/停止条款

- 任一复刻臂与 V16 对应 step 日志不一致 → 停（运行器或环境回归）
- 非有限值 / zero_norm_fallback / MASK_OR_SHIFT_CHANGED → 停
- holdout_v2 铸造失败（语料缺失或去污后 <512 文档）→ layer-2 不启动（不降级用旧 holdout）
- 资源不足 → 保存状态停止，不删任何产物
