# THEORY — specmuon-pilot

## 1. 科学问题

Muon 正交化输出 O 的**谱形**（奇异值分布）可否通过固定先验倾斜改进训练？谱结构假设（来自 muon-ns-allocation 的谱集中观察）是否在 AdamW 域也成立（远期迁移候选）？

## 2. 优化器与更新对象

30M MiniMind + Muon（父代=muon-ns-allocation m1 配方 seed 3101，恢复点 s512/s1536）；作用对象 = O=NS5(H) 的谱形（SVD 重加权），逐矩阵 Frobenius 范数匹配至 native ‖NS5(u)‖。

## 3. 核心数学（逐行取自 `src/specmuon/operators.py`）

```
svd_basis(M) = U, S, Vh = SVD(M)
native_update(M) = vendored_zeropower_via_newtonschulz(M, ns_steps=5)   # native_ns.py, torch 冻结同源
_scale_to(Y, target_norm) = Y · target_norm/‖Y‖_F                       # Frobenius 匹配

ExactPolar (operators.py:36-58):  Y = scale_to(U @ Vh, ‖native‖)        # 正交极因子, 无谱形
Head (operators.py:61-75):  w_i = 1 + λ·(σ_i/σ_1)        → Y = (U·diag(w))@Vh → scale_to  # 头部增强
Bulk (operators.py:76-89):  w_i = 1 + λ·(1 − σ_i/σ_1)    → 同上                                # 体部增强
Power (operators.py:91-):   w_i = (σ_i/σ_1)^p, p≥0       → 同上                                # 谱幂对照
```

P0 谱审计量：top-k 能量份额 Σ_{i≤k}σᵢ²/Σσᵢ²、top-decile 分位、合并变异系数 CV（P0-B 断言口径）。

## 4. 修改前后差异与动机

native：O=NS5(H)（谱形由 NS5 定点隐式决定，头部分量被压制到近均匀）。修改：显式重加权谱形（头/体倾斜 λ=0.25；幂 p=0.5；ExactPolar 端点对照）。动机：P0 测得 NS5 的归一化 gap 0.133/缺陷 0.342 来自 quintic 定点误差而非迭代不足（H2）——即谱形存在「未优化余量」，Head/Bulk 是最粗粒度的两个方向性探针。

## 5. 已验证 vs 仍假设

已验证：H1 动量谱 top-8 份额 0.389 vs 均匀 0.018（21×）；H2 NS5-polar gap 归因 quintic 定点；H4 形状 CONFIRMED（AdamW top-decile 0.86-0.96）但**合并 CV 断言 REFUTED**（0.227>0.2）；P1 Head/Bulk/Power/ExactPolar 全部负裁决；P0-C DATA_GAP→P2 永久封锁。
仍假设：谱形在**其他作用域**（如更新 A、梯度 G）是否有增量——本线未测且已关闭（裁决「不值得继续」）；AMO/SAMuon 类论文方法（SOURCE_BLOCKED）。
