# 02_METHODS_AND_EQUATIONS — 方法与公式总表

逐线详细推导见各 [THEORY.md](../experiments/)；此处为跨线公式对照与統一记号。

## 统一记号

g=当前梯度（clip 后）；m,v=AdamW 动量/二阶矩；β₁,β₂=(0.9,0.999)；ε=1e-8；A(p)=m̂/(√v̂+ε)（AdamW 预条件方向）；H=Muon 动量缓冲（μ=0.95）；NS5=Newton-Schulze 5 次迭代（quintic 系数 3.4445/−4.7750/2.0315，fp64）；η=当前 step LR（cosine→0.1×peak）。

## 方法族对照（全部范数受控）

| 方法 | 更新公式 | 范数控制 | 源码锚点 |
|---|---|---|---|
| 原生 AdamW | u=η·A | — | minimind trainer 复刻 |
| S(a) 幅度 | u=η·a·A | 无（a 即幅度） | campaign_v16/core.py:457 |
| F(a) 梯度方向 | u=η·a·D, D=g·‖A‖/‖g‖ | ‖D‖=‖A‖ fp64 | core.py:418-419, n_a:351 |
| P(a) 预条件方向 | u=η·a·D, D=(g/(√v̂+ε))·‖A‖/‖·‖ | 同上 | core.py:420-423 |
| A2 范数互换 | D←D·q_src/(η·a)（逐 step） | 位移范数=源臂序列（≤2.1e-9） | core.py:425-436 |
| Muon native | O=NS5(buf), u ∝ O | torch _muon.py（SHA 38e95918） | 各线 vendored |
| NS-K 重分配 | O_g=NS_{K_g}(H_g), K_g∈{3,5,7} | ΣK 守恒 | muon-ns-allocation src/p1_engine.py |
| Head/Bulk SpecMuon | Y=(U·diag(w))@Vh, w_i=1+λ·(σ_i/σ₁) / 1+λ(1−σ_i/σ₁) | Frobenius=‖NS5(u)‖ | specmuon operators.py:61/76 |
| ExactPolar | Y=U@Vh | 同上 | operators.py:36 |
| Group-A/B 混合 | O_mix=NS5(λg+(1−λ)buf) | 逐矩阵=native 范数（1.5e-16） | context group_muon.py:151-156 |
| B_L 归因臂 | LR×0.5 **且** decay×0.5 | — | r2f_recipe.py（勘误链 defect_archive） |

## 统计口径（全仓统一）

token 加权 NLL=Σloss_sum/Σtoken_count；配对比较=doc 级 paired bootstrap（1000 次，seed 9071，numpy default_rng；pooled 统计量）；CI95/CI97.5（Bonferroni）；单位=doc 重采样——**非独立 seed CI**；seed 间散布另行报告（V16-C 表）。门阈值均预注册（各 frozen config）。

## 判据分级

CONFIRMED（多 seed/封存）> DEV_SINGLE_SEED（冻结协议+单 seed dev）> MEASUREMENT > STAGED/DESIGN。全部「未执行」与阴性同列（[05](05_NEGATIVE_RESULTS_AND_LESSONS.md)）。
