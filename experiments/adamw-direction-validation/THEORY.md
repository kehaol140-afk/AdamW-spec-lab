# THEORY — adamw-direction-validation

## 1. 科学问题

V16 裁决的核心未闭合缺口（`DIRECTION_CAUSAL_GAP_AUDIT.md:42,79`）：V16 的 F−S 差（+0.036~+0.151）全部来自 **F(8,1) vs S(1,1)** 比较——方向与 8× 目标块幅度完全混杂；S(8,1)（=原生方向+8× 幅度）从未运行。若 S(8,1)≈F/P，V16 必须改写为「幅度×方向组合收益」；若 S(8,1)<S(1)<F/P，方向替换是必要成分。

## 2. 优化器与更新对象

与 V16 完全相同（AdamW 复刻、目标块 `layers.1.mlp.down_proj`、4.66M MiniMind）。数学内核 `src/dv_core.py` = `campaign_v16/core.py` 逐字 vendored + 4 个非语义增量（D1 删账本写入 / D2 CUDA 守卫 / D3 eval 路径参数化 / D4 内联 betas；math byte-identical，文件头注释文档化）。sha `b8943249f87599fb…`。

## 3. 核心公式（同 V16 THEORY §3，此处为臂定义）

```
dv_S{1,4,8}: D = A(target), 有效更新 = η·a·D          （a=1/4/8，纯幅度轴）
dv_F8:       D = g_clip·‖A‖/‖g_clip‖,   更新 = η·8·D  （当前梯度方向）
dv_P8:       D = (g_clip/(√v̂+ε))·‖A‖/‖·‖, 更新 = η·8·D（预条件方向）
b=1 全部臂；三臂共享同 q_t 位移范数轨迹（CRN）
```

关键控制：**q_t 匹配审计**（`LAYER1_UPDATE_NORM_AUDIT.md:46-49`：F/P/S8 的 q_t 逐 step 相对偏差 0.1-0.2%，S1 天然=η·‖A‖）——排除「范数失配冒充方向收益」。

## 4. 修改前后差异与动机

见 V16 THEORY §4-5（同源）。本实验的增量是**补齐因果分解缺失格**：{方向}×{幅度} 的 2×2 中，V16 只有 (S,1)/(F,8)/(P,8) 三格，本实验补 (S,4)/(S,8) 并把 (S,1)/(F,8)/(P,8) 用同一 runner 复刻（位级）。

## 5. 已验证 vs 仍假设

已验证：幅度单调有害（S1<S4<S8 逐级恶化，三视界一致）；方向必要性（F8/P8 同时优于 S8 与 S1）；P>F（+0.031 CI 全正）；范数失配排除；与 V16 位级复刻（mismatches=0）。
仍假设：跨 seed（layer-2 未跑）；跨规模（→30M 线）；holdout 级证据（holdout_v2 封存未开）。
