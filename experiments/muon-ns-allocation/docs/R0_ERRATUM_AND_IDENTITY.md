# R0_ERRATUM_AND_IDENTITY — 旧轮缺陷登记与本轮身份闭环

日期：2026-09-29。依据：R0_R1_EXECUTION_PROTOCOL_v1_1 §3.3 与 references/INDEPENDENT_REAUDIT_CN.md §4.2。

## 1. 旧 P0/P1 的 LR 时点混用（登记，不重写旧赢家）

**实测证实**（R0_GATE.json）：
- step512 checkpoint 保存组 LR=0.008799 = lr_at(512)（该 update 消费的值）；下一步应消费 lr_at(513)=0.008794（相对差 5.7e-4）。
- step1536：保存 0.002358 = lr_at(1536)；下一步 0.002353（相对差 2.1e-4）。
- 旧 P0/P1 的 native θ₀⁺ 由恢复后的 `opt.step()` 产生 → 使用 **lr(s)**；候选 u 向量用 `lr_at(s+1)` → **lr(s+1)**。二者不一致（独立复查 §4.2 的推断在服务器上得到证实）。
- 影响范围：旧轮所有 native-候选差的绝对量级含一个 O(5e-4) 相对 LR 偏移；旧 raw 候选负值结论的方向不受威胁（偏移远小于测得的负差？不外推——旧结论维持 PARTIALLY_VERIFIED 原标签，本轮以统一时钟重测）。
- **本轮 R0 修复**：恢复后把 Muon 与 aux 全部参数组写入 lr(s+1)；native step、手动组装、WD、aux 与全部候选用同一时钟。统一时钟下手动组装 vs 原生 step 最大差 1.49e-8/2.98e-8（浮点路径差，<5e-8 门槛）。

## 2. 旧选择规则不可追溯项（UNKNOWN，不倒签）

- 旧 P0/P1 并列（tie）规则在实现中以 `(min(V), label)` 隐式处理，无显式 near-best/tie 键；median γ 的来源代码在 `run_p1_joint_and_lock.py`（`sorted(vals)[len//2]`）——可追溯但**未在旧协议文本中预注册** → 登记为规则证据 UNKNOWN，本轮前瞻规则（near-best ≤m+1e-6 + 六级 tie 键）先落盘于 `src/r1_selection.py` 与本协议，再查看校准结果。
- 旧轮 512 状态"no_op"判定的 1e-9 容差来源未预注册（同上登记）。

## 3. 本轮身份清单（R0_GATE.json + selection_lock.json）

| 项 | 值 |
|---|---|
| 父轨迹 | seed3101 单轨迹（P0/P1 轮创建，未重训）；step512 SHA `7abd22ec189f5833…`（完整 SHA 见 R0_GATE.json），step1536 `d0f032c88158fdea…`，step2048 备存未用 |
| 安装 Muon | `/…/torch/optim/_muon.py` SHA `38e95918…`（与历轮一致） |
| 模型/路由 | MiniMind 30,025,216 参数；O 组 = 8 层 `self_attn.o_proj.weight`（512×512 ×8），预提名固定，未看结果改组 |
| 数据 | 生产流 manifest（p=0.5, seed2030）连续切片：512 态 [520:528)、1536 态 [1544:1552)；**与旧 P0/P1 probe 窗口（[512:520)∪[1536:1544)）互斥（已验证）**；每 batch 128 窗/65,280 有效 targets（32,640/域） |
| 评价 | Q_select/Q_P1_check（P0P1_Q_v1 组级划分，历史开发暴露保留）；Q_P2_endpoint/confirm 零访问（access_guard） |
| 代码 | r1_probe/r1_selection/run_r1 SHA 全录于 selection_lock.json |

## 4. R0 数值门（通过）

- 统一时钟下 a=b=1/NS5 与原生 `opt.step()` 一致（1.5e-8/3.0e-8）；aux/WD/计数一致；A→B→A 恒等；grad-hash 不变；payload 不可变；AdamW step 别名修复沿用（0 共享）。
- **等范数（O 组整体，FP64 归约）**：c=nu5/nu3 实测 512: 1.3035–1.3285、1536: 1.2356–1.2428（逐 batch）；intended 相对差 ≤1e-16、**realized 相对差 ≤2.2e-7**（≤1e-4 门槛）——真实写入参数后的 data-delta 验证通过（norm_match_checks.csv 逐点保存）。
- e_repeat = 0（同参数点重复评价，数值重复性；**非采样不确定性**）。
