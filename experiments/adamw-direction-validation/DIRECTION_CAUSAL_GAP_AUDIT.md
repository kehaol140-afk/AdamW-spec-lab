# DIRECTION_CAUSAL_GAP_AUDIT — V16 方向/幅度因果缺口审计（零 GPU）

日期：2026-10-11。性质：只读审计 + 新项目预注册依据。未修改任何 V16 文件；全部证据以 file:line / 记录字段定位。

## Q1 F/P 的真实数学更新公式

来源：`adamw_spectral_pilot/campaign_v16/core.py:385-468 apply_step`（冻结 source_hash `4afaa78e…`）。

逐步语义（每 optimizer step，梯度已按 valid_total 平均 + 全局 clip 1.0）：
1. 对每个参数解析计算 AdamW 增量：`m⁺=β1·m+(1−β1)g`，`v⁺=β2·v+(1−β2)g²`，`A=(m⁺/(1−β1^t⁺))/(√(v⁺/(1−β2^t⁺))+ε)`（core.py:406-408；与 opt.step 内部演化同一状态——**状态更新本身三族完全相同**，只改参数位移）
2. 方向定义（core.py:418-432）：
   - **S**：`D = A_target`（AdamW 动量商方向，原样）
   - **F**：`D = N_A(g_clip)`（当前 clip 后梯度，Frobenius 精确匹配到 ‖A_target‖：`n_a` fp64 重标定，core.py:375-383）
   - **P**：`D = N_A(g_clip/(√(v̂⁺)+ε))`（单步预条件当前梯度，范数同上匹配）
3. 位移（core.py:449-459）：先原生 `opt.step()`（全部参数按 AdamW 更新+解耦衰减），再无梯度校正：目标块 `θ += η(A−aD)` → 净位移 `−ηaD` 且衰减 `(1−ηλ)` 保留；其他参数 `θ += η(1−b)A_p` → 净位移 `−ηbA_p`。`S(1,1)` 走 bit-native 跳过（test_v16 验证；本审计 vendored 版复验 PASS）
4. `q_t = η·a·‖D‖`（core.py:461）；q_override 时 fp64 精确重标定到外部范数序列（A2 实测偏差 ≤2.1e-9）

## Q2 a、b 的作用域

- **a**：仅目标块 `model.layers.1.mlp.down_proj.weight`（campaign_config.json `"target"`）的位移倍率
- **b**：其余全部 45 个参数的位移倍率（b=1 → 原生 AdamW；b=2 → 两倍；b=0.5 → 半倍）
- B/C 全部臂 b=1；b≠1 仅存在于 A1（续训）网格与 S4/S5/S6/S7（均为 b=2）

## Q3 F/P 是否引入额外梯度/裁剪/隐式归一化

- **无额外梯度计算**：F/P 与 S 共享同一 backward（micro 2×accum 4）；`g_clip` 就是 clip 后的 `param.grad`（core.py:419）
- **无额外裁剪**：全局 clip 1.0 对三族同样先作用（run.py:361）
- **有显式（非隐式）归一化**：`n_a` 把 ‖D‖ 精确设为 ‖A_target‖——这是设计核心（同范数比较），并在 A2 中进一步被 q_override 外部范数轨迹替代。方向向量本身不被归一化到单位长（保持 ‖A‖ 尺度）
- 额外成本：每步 2 次 fp64 范数 + 一次逐元素重标定（V16 账本 direction_seconds 分列；P 另有单步 v̂ 计算）

## Q4 B 阶段是否缺少 S(a=8,b=1) 对照 —— **确认缺少**

QUEUE.jsonl 全 18 条 B 臂实查（本审计逐行核对）：

| 臂族 | (a,b) |
|---|---|
| b_S1/S2/S3、b_S8 | **(1,1)** |
| b_S4/S5/S6/S7 | **(8,2)**（S5/S6/S7 为其 β1 变体） |
| b_F1-F3、b_P1-P3、b_F4、b_P4 | **(8,1)** |
| b_F5、b_P5 | (4,1) |

→ **S(a=8,b=1) 从未运行**（续训 A1 有 S×a 网格但 b 全网格；从头场景无 (8,1)）。V16 C 的 F−S 差（0.036-0.151）是 **F(8,1) − S(1,1)**：方向与目标块幅度 a=8 完全混杂。`b_S4(8,2)` 不能替代（b=2 本身从头有害：4.3406 vs S1 4.2406）。**这是 V16 裁决的核心未闭合缺口。**

## Q5 P 被 screen 排除的完整原因链

1. `campaign_config.json hard_budget.weights_only_max=12` + `train_branch save_endpoint` → endpoint 滚动删除按队列顺序：b_S1-S3、b_F1-F3 的 endpoint 被删（`campaign_v16/deletions.jsonl` 7 条 "weights-only cap; regenerable by rerun"）
2. `run.py plan_b_select`（:611-633）：screen 循环 `for r in rows[:8]`（**队列顺序**，非分数顺序）+ `if not endpoint.is_file(): continue` → 仅队列第 7、8 位（P1、P2）有 endpoint 被评估 → `STATE.selections.route.screen = {P1:4.126, P2:4.327}`
3. `run.py plan_route_b`（:633-673）：`gain_screen = (screen.get(best_s_id, 0) - screen.get(cid, 0)) if cid in screen else None`：
   - F1 ∉ screen → gain=None → 门 `(gain_screen is None or ≥0)` 经 **None 旁路通过**
   - P1 ∈ screen → best_s（S1）缺席 → 默认 0 → gain = 0−4.126 = **−4.126 < 0 → P 被拒**
4. 结论：C 配方选择由 **checkpoint 清理顺序 × 缺席基线默认值** 驱动，非实测 screen 比较（S1 真实 screen 值未知；dev 排序 P1>F1 且 screen 内部 P1<P2 与 dev 一致）。未修改冻结代码；新项目规则：**P8 在 layer-2 强制包含**（frozen_v1.json `p_inclusion`）

## Q6 全部比较的超参/数据/范数口径

| 项 | 值（B/C 全臂一致） |
|---|---|
| LR | 峰值 0.001（网格 {0.001,0.003,0.006}+外扩 0.0003 的最优点；三族同 LR 公平） |
| schedule | warmup 128(B)/256(C) 线性 → cosine 至 0.1·peak |
| betas/eps | (0.9, 0.95) / 1e-8（矩阵组 wd=0.1，1D 组 wd=0） |
| clip | 全局 1.0（梯度先 ÷valid_total） |
| 数据 | 8.5M uint32 缓存（sha `c91382dc…`），scratch 从游标 0；B 单流 seed3407，C 每 seed 独立流（3408-3410） |
| 更新范数 | 逐步 q_t 落盘（S1 从头 q̄≈0.084@205 步 → anchors.jsonl；a=8 臂 q_t=η·8·‖A‖） |
| 评估 | development 256（B 选择 + C 训练中）、screen 256（B 路由）、final_holdout 1024（C 一次开封） |

## Q7 旧 final_holdout 身份与开封事实

- 身份：1024 文档 = 220 final_holdout_reserved + **804 不在当前分区账本的 legacy 文档**（本审计 NEW_EVAL_SPLIT_MANIFEST GAP_2 详述该跨账本现象）；ids+token 序列封存于 `eval_sets.json`（digest 入 data_manifest）
- 开封：2026-10-10 V16 C 终点锁定后一次性开封，产出 findings holdout 数字——**已烧毁**：任何新研究不得将其用作独立评价或选优依据（新项目 frozen_v1 `forbidden` 条目）

## Q8 device_task_seconds (0.121h) ≠ 实际 ~6.5h GPU 的原因

`device_task_seconds` 的累计点（core.py grep 全量）：
- **core.py:449**：仅包住 `opt.step()` 调用的 wall 片段（≈7ms/步 × 62,464 步 ≈ 435s ≈ 0.121h ✓ 自洽）
- **run.py:353（backward）与 core.py:504（eval）显式加 0.0**；forward 从不计入
→ 该指标度量**优化器内核执行秒**，不是 GPU 占用时间。24h "设备任务"硬上限实际约束的是 opt.step 内核时间，远宽于直觉；实际 GPU 墙钟 ~6.5h（B 3.6h + C 2.4h + A2 0.2h + 评估）。新项目成本核算以墙钟为准（TRAINING_COST_ESTIMATE.md）。

## 结论

V16 的 "F/P 方向收益" 在三个层级（A2 范数对照、B 终点、C 跨 seed）都恒为 **F(8,1)/P(8,1) vs S(1,1)** 的比较——**从未与同尺度 S(8,1) 对照**。方向归因的成立前提是 S(8,1) 追不上 F/P；该对照缺失即本项目 layer-1 的全部理由。反向可能同样真实：若 S(8,1)≈F/P，V16 结论须修正为"幅度×方向组合收益，方向独立贡献未证实"。
