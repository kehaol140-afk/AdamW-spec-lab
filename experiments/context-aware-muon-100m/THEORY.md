# THEORY — context-aware-muon-100m

## 1. 科学问题

把 Muon 正交化的输入从「纯动量缓冲」改为「当前梯度与动量的凸混合」（上下文感知），能否在 100M 规模、认真调参的强基线下取得增益？（动机：30M 时 Muon 大幅领先 AdamW；NS 输入的「陈旧性」是直觉上最可疑的自由度。）

## 2. 优化器与更新对象

模型 MiniMind 100,784,000 参数（h768/13L/12Q-6KV/FFN2432/seq256/tied）。Muon 本体 = torch `_muon.py`（SHA `38e95918…`，momentum 0.95/nesterov/NS5/(3.4445,−4.7750,2.0315)/eps1e-7/original-shape scaling）；`B_muon_rms` = match_rms_adamw 更新幅度变体（与 AdamW 更新 RMS 对齐，`configs/baseline_registry.json`）。

## 3. 核心数学（取自 `src/context_aware_muon/group_muon.py` 与 `ns5.py`）

```
g = 当前梯度;  buf = Muon 动量缓冲 (μ=0.95)
Group-A（FFN 39 矩阵）: O_mix = NS5( λ·g + (1−λ)·buf ),  λ ∈ {0.65, 0.85}
Group-B（再含 V/O 26 矩阵）: 同式
Q/K 投影保持 native NS5(buf)
逐矩阵 fp64 范数匹配至 native 输出范数 → 方向变化与幅度变化解耦
lam_anchor（ns5.py; group_muon.py:151-156）: λ = 1−μ 时混合输入与 native Nesterov 输入
    (1−μ)·g + μ·buf 逐位相同（bitwise 锚点，用于等价性验证）
```

实测方向诊断（C_A_lam0p85，65 个 diag 步）：cos(O_mix,O_nat) mean 0.477（0.338-0.9998）、cos(O_mix,g) mean 0.594、norm_rel_err_max 1.5e-16——混合真实改变方向（~61-64°），非安慰剂；范数匹配到机器精度。

## 4. 修改前后差异与动机

native：O=NS5(buf)（纯历史）。修改：O=NS5(λg+(1−λ)buf)（历史×当前凸混合）+ 严格范数匹配。动机链：若「陈旧动量方向」是 Muon 的瓶颈，混合应产生净收益；若收益存在，还需回答它是否只是「降低有效动量」（low-μ 对照臂排除该解释）。

## 5. 已验证 vs 仍假设

已验证（本 pilot 尺度，全部阴性）：Group-A 最佳 λ=0.85 仍劣强 Muon 锚 +0.1367 [CI +0.122,+0.152]；V/O 扩展有害 +0.0997 [CI +0.082,+0.116]；**动量解释被排除**（low-μ0.9 自身劣锚 +0.127）；三族 LR 最优全落网格低缘（RANGE_UNRESOLVED×3，按冻结边界规则不外扩）；tokens 不足（14/15 臂 NLL 严格单调抬升）。
仍假设/未执行：确认层（多 seed）——按裁决不开；更大 tokens 预算下的位次；NorMuon/PE/VR 变体（staging 卡 G-2 锚点悬置，与官方仓逐行等价性未验证——网络阻断）。
