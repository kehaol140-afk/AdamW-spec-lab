# 05_NEGATIVE_RESULTS_AND_LESSONS — 负结果与教训总册（与正结果同权重发布）

阴性结果是本实验室的主要产出形态之一。以下均为**正式裁决**（冻结协议+原始数据+书面判读），非「没做出来」的委婉说法。

| # | 线 | 裁决 | 一句话内容 | 证据 |
|---|---|---|---|---|
| N1 | V14 | SCALE_RANGE_UNRESOLVED | 有限 rank-1 谱修正无可靠独立增量（谱vs标量 2.41e-6，CI 跨零） | experiments/adamw-spectral-v13-v16 (findings.json) |
| N2 | MSA P0/P1 | 重分配阴性 | 28 单组 K∈{3,5,7} 重分配 0/28 达 δ=1e-5；21 负 7 弱正 ≤1e-6 | muon-ns-allocation check_rows.csv + verdict.json |
| N3 | MSA R2F | AH 关闭 | 256 步 AH 收益被 B_L（LR×0.5+decay×0.5）解释 101.7%/100.5%；AH 状态优势共同尾部反转为劣势（R_N<0） | lrwd_attribution.csv + AH_BRANCH_CLOSURE_CN.md |
| N4 | MSA R2F-NL | 无实用增量 | G_NL 存在但 AH 相对 L \|Δtail\|≤8e-4<τ=1e-3 | nl_effects.json |
| N5 | SpecMuon P1 Head | NO_EFFECT | s512 −0.0051 → s1536 −0.001 回落屏下，双恢复点不成立 | P1_RESULTS.csv + P1_DECISION.md |
| N6 | SpecMuon P1 Bulk | SINGLE_POINT | 仅 s512 过 2e-3 屏；LR×0.5 网格比 Head/Bulk 好 0.071-0.09 nats | 同上 |
| N7 | 100M Group-A/B | **PILOT_FAIL — NO_GAIN_OVER_STRONG_MUON** | 最佳候选劣 Muon 锚 +0.1367；V/O 扩展有害 +0.0997；动量解释被 low-μ 排除；三族 LR 网格低缘 RANGE_UNRESOLVED×3；33.5M tokens 下 14/15 臂 NLL 严格单调上升 | PILOT_100M_DECISION.md + analysis.json gate |
| N8 | Hyperball A1 | 范数保持无终点收益 | MuonH 3.27921 vs Muon 3.27906（Δ+1.5e-4，种子噪声内；范数约束机械生效 261.928 恒定） | baselines/hyperball-a1-evidence |
| N9 | 旧仓：GTTR 规则化 | 全部失败 | F2a/F2b 方向不确认；D1 不超固定 LR 网格 | 旧仓 archived/fr_gttr |
| N10 | 旧仓：continuation | STOP | C1 +1.587 / C2 +0.019（Holm p=1.0） | 旧仓 archived/continuation_adaptation |
| N11 | 旧仓：SAU | NO_METHOD_GAIN | 敏感性可观测但不可利用 | 旧仓 archived/optimizer_sensitivity |
| N12 | 旧仓：transition | R=0.0000 | p* 不因优化器改变 | 旧仓 archived/muon_transition |
| N13 | 旧仓：temporal T2b | NO_USEFUL_SIGNAL | 候选极差低于筛查线 30× | 旧仓 archived/muon_temporal_response |

## 横断教训（多条独立线共同指向）

1. **标量幅度/LR 一阶、结构二阶**：N2 vs ×0.5 +3.4e-4；N3 B_L；N6 LR 网格；N7 LR 边界——任何新结构候选必须先过同预算 LR/幅度网格 hurdle
2. **一步筛查不可信**（R2 符号翻转实证）
3. **状态补丁不保留**（R2F R_N<0）
4. **范数保持无便宜收益**（N8）
5. AdamW 域唯一存活正信号（方向替换）也须声明：dev 级单 seed、不敌 Muon（30M −0.2725）

## 附：未执行 ≠ 阴性

Direction layer-2 / 30M T2 / holdout_v2 / P1 跨 seed / 100M 确认层 / NorMuon 消融：**未执行**，不得引用为任何方向的证据。
