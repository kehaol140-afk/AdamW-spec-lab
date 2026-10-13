# THEORY — muon-ns-allocation

## 1. 科学问题

有限 Newton-Schulze 迭代预算 K 在参数组间如何分配才最优？Muon 的正交化输出 O 的**谱结构/迭代分配**是否存在超越标量调参的改进空间？

## 2. 优化器与更新对象

底座：30M MiniMind + Muon（m1 配方：lr2.0/wd0.02/p0.5/seed2030 父轨迹，来自 muon_transition M 线）。**作用对象 = O = NS5(H)**，其中 H=Muon 动量缓冲（**不是**梯度 G、混合 M、更新 A 或权重 W——README 常见误读，本线明确排除）。

## 3. 核心数学（取自 `src/` 与冻结配置）

Newton-Schulze（torch `_muon.py` 冻结快照 SHA `38e95918…`；specmuon native_ns.py 同源 vendored）：

```
NS5(X): X ← X·(a·XᵀX + b·I)·(c·XᵀX + d·I)?  — 实际 quintic 定点:
每步 X ← (b X + c (X Xᵀ) X + d (X Xᵀ)² X)/归一, 系数 (3.4445, −4.7750, 2.0315), fp64, 5 次迭代
```

逐轮算子（P0/P1→R2F）：

```
P0/P1 重分配: 组 g 的 NS 迭代次数 K_g ∈ {3,5,7}（ΣK 守恒重排）→ NS_{K_g}(H_g)
R0 统一时钟: 每 NS 步后归一化时序统一（manual vs step 实现对拍 maxdiff 1.49e-8）
R1 组倍率: O_g × γ_g（组间幅度重分配）; 等范数: ‖O_g‖ 匹配
R2 延续臂: N/AH/L/D 自恢复点 256 步（AH=NS 幅度混合; L=LR×0.5; D=decay×0.5）
R2F B_L 归因: B_L = LR×0.5 + decay×0.5 组合臂；G=V(参照)−V(臂) 配对增益
   归因比例 = B_L 增益 / AH 增益 = (V_N−V_L)/(V_N−V_AH)
R2F-NL: 非线性尾部变体（同预算）
逐矩阵 Frobenius 范数匹配至 native ‖NS5(u)‖（范数守恒，只改谱形/分配）
```

## 4. 修改前后差异与动机

native：全部组统一 NS5。修改：重分配 K_g / 组倍率 / 统一时钟 / 幅度混合 / LR-decay 归因。动机：若 NS 预算在组间有异质最优分配，可在零额外计算下提升——P0/P1 是最便宜的检验。结果全部阴性后，R2F 的价值转为**归因**：把表面「结构收益」分解到 LR/decay 标量。

## 5. 已验证 vs 仍假设

已验证：28 单组重分配无一达 δ=1e-5；全局更新 ×0.5 是唯一强信号（+3.42e-4/+5.8e-5，比谱候选大一个量级）；统一时钟 V(N)−V(G) 全局正、组级结构零增量；AH 256 步收益被 B_L 解释 101.7%/100.5%（可由 lrwd_attribution.csv 复算：B_L=V_N−V_L=0.074216 vs AH 增益=V_N−V_AH=0.072974）；一步筛查符号翻转（512 MH 一步 −1.46e-5 → 256 步 +0.0383，R2 paired_effects 原值）。
仍假设/未解决：SOURCE_BLOCKED 三论文（AMO/SAMuon/Hyperball）原文未核对，禁止二手重建；K 在组内（非组间）的分配未测；更大 K>7 未测（终局决策不再立项）。
