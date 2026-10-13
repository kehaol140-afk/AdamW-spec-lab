# MUON_MOMENTUM_EQUIVALENCE — 新方法数学真值审计（任务一）

日期：2026-10-12。方法：逐行源码推导 + 位级/数值测试锁定（`tests/test_group_muon.py` 25/25 PASS，含本轮新增 2 项）。外部真值 = torch 2.11.0 `torch/optim/_muon.py`（sha256 `38e95918…`，与本仓 vendored `ns5.py` 逐操作一致）。

---

## 1. B_t 是什么：EMA 动量缓冲，不是经典动量，也不是 Nesterov 有效输入

torch Muon 代码真值（`_single_tensor_muon`，vendored 等价物 `group_muon.py`）：

```
B_t ← B_{t-1}.lerp(g_t, 1−μ) = μ·B_{t-1} + (1−μ)·g_t     # EMA，非经典 μ·B+g
B̃_t = g_t.lerp(B_t, μ)    = (1−μ)·g_t + μ·B_t            # nesterov=True 时的 NS5 输入
O_t = NS5(B̃_t)；θ ← θ·(1−lr·wd)；θ ← θ − adj_lr(lr)·O_t
```

- **torch 文档字符串写 `B_t ← μB_{t-1}+g_t`、`B̃=g+μB`，与实现不符**（lerp 语义；Keller Jordan 参考实现同 lerp）。本轮再次逐行确认。
- **勘误记录**：本仓 `100M_METHOD_DESIGN.md` 伪代码此前写 `buf ← μ·buf + g`（错误），`group_muon.py` docstring 残留 `lam0=1/(1+μ)` 旧推导（源自文档字符串而非代码）。两处已于 2026-10-12 修正并加日期注记；**代码本身从未错**（位级等价测试 `test_native_bitwise_vs_torch_muon` 一直通过）。
- 新增锁定测试 `test_momentum_buffer_is_ema_not_classic`：一步更新后 buffer 与 `μB+(1−μ)g` 一致（1 ULP 内）、与经典 `μB+g` 显著不同。

## 2. Group-Muon-A 与普通低动量 Muon 的精确关系

**定义**（作用集内每个矩阵；μ=0.95 固定，动量态与 native 位级同轨——`test_momentum_state_independent_of_lam`）：

```
M_t = λ·g_t + (1−λ)·B_t        B_t 为上述 EMA 缓冲
O_mix = NS5(M_t)，逐矩阵范数匹配至 ‖O_nat‖（fp64，实测 1.87e-16）
```

**锚点**：λ₀ = 1−μ = 0.05 时 M ≡ B̃（native Nesterov 输入），优化器对该值路由到同一 lerp 表达式 → **位级等价 native**（`test_lam_anchor_direction_and_norm_match`：参数轨迹差 <1e-5，cos=1）。**λ 的定义域因此是 (0.05, 1]，离开锚点即离开 native。**

**一阶等价量：有效梯度年龄**（稳态，梯度历史权重加权的平均滞后步数；EMA_μ 滞后 = μ/(1−μ)）：

| 配方 | 当前梯度权重 | 历史衰减率 | 有效年龄（步） |
|---|---|---|---|
| native μ=0.95 | 0.05 | 0.95 | μ²/(1−μ) = **18.05** |
| low-μ′=0.9 | 0.10 | 0.90 | 8.10 |
| low-μ′=0.8 | 0.20 | 0.80 | 3.20 |
| **mix λ=0.65, μ=0.95** | **0.65** | 0.95 | (1−λ)·μ/(1−μ) = **6.65**（≈μ′0.88） |
| **mix λ=0.85, μ=0.95** | **0.85** | 0.95 | **2.85**（≈μ′0.78） |

**结论 1（λ=0.65/0.85 的实际作用）**：λ 是 NS5 输入中**当前梯度的权重**；由于 NS5 正尺度齐次 + 范数匹配，λ 的唯一因果通道是**输入方向**——λ=0.65 把 FFN 矩阵的有效梯度年龄从 18.05 步缩到 6.65 步，λ=0.85 缩到 2.85 步，且历史成分保留 0.95 长衰减尾（"锐利当前 + 长平滑记忆"的双峰结构）。

**结论 2（与 low-μ 无精确等价）**：mix 家族与任何 μ′ 的 native 在一般梯度序列下方向不同（历史衰减剖面不同；两步历史下权重比解析式：mix w₁/w₂ = (1−λ)μ(1−μ)/(λ+(1−λ)(1−μ))，native(μ′) 为 μ′²/(1+μ′)，λ=0.85 时 0.0083 vs μ′=0.8 时 0.356，相差 43 倍）。新增测试 `test_mix_family_has_no_low_momentum_equivalent`：μ′ 全网格扫描（0.005..0.995）最优 cos < 0.9999。
**因此 low-μ{0.9, 0.8} 臂是"一阶年龄匹配对照"（λ=0.65↔μ′≈0.88、λ=0.85↔μ′≈0.78，两臂恰好包夹），不是等价复制**——这正是排除"收益只是换动量系数"解释的正确构造。

**结论 3（动量解释的检验逻辑）**：若 C_A 收益被 low-μ 复现同量级 → `MOMENTUM_EXPLAINED`（一阶新鲜度足够）；若 C_A 收益显著超过年龄匹配的 low-μ → 归因于混合方向的二阶结构（长记忆 × 当前方向的组合几何），这是候选的科学主张。

## 3. 混合方向之后的完整更新链核验（逐项，代码行号）

| 设计声明 | 代码证据 | 状态 |
|---|---|---|
| NS5 原版（系数 3.4445/−4.775/2.0315，5 步，bf16，转置规则，eps 1e-7） | `ns5.py:25-55` 与 torch `_zeropower_via_newtonschulz` 逐操作一致；`test_native_bitwise_vs_torch_muon` 位级 | ✅ |
| NS5 输入 = M（混合方向），非正交化后混合 | `group_muon.py` mix 分支：`o_mix = ns5(mix)` 先于范数匹配 | ✅ |
| 范数匹配：fp64 下 ‖Δ_mix‖ = ‖Δ_nat‖（同 lr 同步） | `scale = n_nat/(n_mix)`，`delta = o_mix_d·(−adj_lr·scale)`；实测 norm_rel_err_max **1.87e-16**（100 步 GPU 诊断） | ✅ |
| 学习率调整对 mix 矩阵同样生效（original sqrt(max(1,A/B)) 或 match_rms 0.2·sqrt(max(A,B))） | mix 分支调用同一 `adjust_lr(lr, group["adjust_lr_fn"], p.shape)` | ✅ |
| weight decay 用**原始 lr**（非形状调整后），解耦式，更新前施加 | `p.mul_(1 − lr·wd)` 先于 `p.add_(delta)`，与 torch 顺序一致 | ✅ |
| 动量态演化与 λ 无关 | `buf.lerp_(grad, 1−momentum)` 在 mode 分支之前，无条件执行 | ✅ |
| aux 参数（embed/norms）AdamW 同 LR 同 schedule | runner 接线（`run_pilot_arm.py` build_optimizers）；M 线语义 | ✅ |
| 锚点路由位级 | `abs(lam−(1−momentum))<1e-12` 时走 `grad.lerp(buf, momentum)` | ✅ |

## 4. NS5 数值性质（与 λ 语义相关）

- 正尺度齐次：NS5(cX)=NS5(X)（bf16 舍入内）→ 混合比例只进方向不进幅度（`test_homogeneity_positive_scale`）。
- 1e-7 相对输入扰动在 fp64 cos 度量下 <1e-4 影响；fp32 求和在 ~2M 元素上有 ~1e-4 误差（会伪造/淹没方向信号）→ 一切 cos 诊断必须 fp64（本轮重申并已在代码中固定）。
- 谱平坦化：gram 对角 ∈[0.2,2.3]（NS5 输出奇异值 ~[0.5,1.5]，torch 注释确认 S′~U(0.5,1.5) 非精确 UV^T——**"正交化"是近似谱白化**，方向诊断读数应按此理解）。

## 5. 裁决

1. B_t = **EMA 动量缓冲**（μB+(1−μ)g）；NS5 的输入是 Nesterov 有效输入 B̃=(1−μ)g+μB；mix 家族的锚点在 λ₀=1−μ=0.05（非文档推导的 0.513）。
2. λ=0.65/0.85 = NS5 输入中当前梯度权重 65%/85%，等效把 FFN 矩阵梯度年龄 18.05→6.65/2.85 步且保留长记忆尾；**不存在精确的 low-μ 等价物**（解析 + 数值双锁定），low-μ 臂为一阶年龄匹配对照。
3. 更新链（NS5→范数匹配→adj_lr→解耦 wd→aux AdamW）逐项与声明一致；范数匹配实测达机器精度。
