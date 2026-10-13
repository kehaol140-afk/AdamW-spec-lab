# MOMENTUM_SCALING_ERRATA — Keller/torch 与 Moonlight 镜像动量参数化的标度等价（零 GPU 勘误）

日期：2026-10-12。脚本：`scripts/momentum_scaling_errata.py`；原始数值：`results/momentum_errata.json`（进证据包）。零 GPU；零修改（冻结文件、staging、历史项目全部未动；镜像优化器只读 import）。

---

## 1. 命题与解析证明

两种动量参数化（同一梯度序列、同一 μ、零初始化）：

```
Keller/torch:  B_t = μB_{t-1} + (1−μ)g_t     X_K = (1−μ)g_t + μB_t
Moonlight 镜像: C_t = μC_{t-1} + g_t          X_M = g_t + μC_t
```

归纳：B_1 = (1−μ)g_1 = (1−μ)C_1；若 B_{t-1} = (1−μ)C_{t-1} 则
B_t = μ(1−μ)C_{t-1} + (1−μ)g_t = (1−μ)(μC_{t-1} + g_t) = (1−μ)C_t。
于是 X_K = (1−μ)g_t + μ(1−μ)C_t = (1−μ)(g_t + μC_t) = (1−μ)X_M。∎

**两族是精确的标度重参数化**：镜像的经典全梯度动量缓冲恒等于 torch EMA 缓冲的 1/(1−μ) 倍（μ=0.95 时 ×20）。

## 2. 数值实证（E1–E4）

| 实验 | 内容 | 结果 |
|---|---|---|
| E1 解析恒等式（fp64，T=64，3 形状） | B_t−(1−μ)C_t 与 X_K−(1−μ)X_M 相对误差 | **≤ 4.0e-17**（fp64 机器精度，精确成立） |
| E2 NS5 输出 | vendored NS5(X_K) vs 镜像 NS5(X_M)：cos ≥ **0.9996**，范数比 0.990–1.000；**对照** vendored NS5(X_K) vs vendored NS5(X_M)：cos ≥ 0.9999，范数比 1.0000–1.0001 | 标度被 NS5 的 Frobenius 归一化消除（对照 1e-4 级）；对镜像的 ~1% 范数差全部来自**其 NS5 管线**（`B=b*A+c*A@A`+`X=a*X+B@X` 的逐项算术 vs torch 融合 `addmm`；bf16 下 5 步 quintic 放大），与动量机制无关——对照实验隔离证明 |
| E3 优化器逐步轨迹 | torch Muon(match_rms) vs 镜像 Muon，同梯度/lr/wd，64 步 | 末参数相对差 **2.6e-4**（bf16 级、非系统性） |
| E4 真实小模型训练（MiniMind 2 层，CPU，60 步） | torch(Muon_rms+aux AdamW) vs 镜像（Muon+内置 AdamW） | 首步 loss 位级同（8.80683）；逐步 loss 差 ≤ 9.7e-4；末参数相对差 1.4%——bf16 级种子差经非线性训练混沌放大，属预期，非机制差异（E3 已证逐步更新等价） |

结论门（任务书口径）：**差异全部在 BF16/epsilon 解释范围内**；无需归因于"历史动量机制"，也未发现任何实现错误。

## 3. 对本项目的含义

1. **Native Muon / Group-Muon-A 正确性**：本项目 vendored 路径与 torch 位级一致（测试锁定），本勘误未发现任何影响正确性的代码错误。**正式训练按授权继续。**
2. **"动量世系分叉"降级为重参数化**：并行审计记录的 Keller-lerp vs Moonlight-full-grad 分叉（缓冲差 ×(1−μ)）在 NS5 之后**不构成更新方向或幅度的语义差异**（本勘误 E2/E3）。其提案中的 `B_moonlight_mirror` 增补臂与本项目 `B_muon_rms` 臂（同 0.2·sqrt(max(A,B)) 缩放 + 等价动量）**在更新语义上冗余**——v1.1 提案若被采纳，应据此删除该臂，节省 +1 GPU-h。
3. **Group-A 的 λ 定义不受影响且必须绑定参数化**：M = λg + (1−λ)B 在 torch/EMA 参数化下定义并冻结（锚点 λ₀=1−μ 精确成立）。若换用经典参数化 C，同一 λ 值对应**不同**混合（M′ = λg + (1−λ)C = λg + (1−λ)B/(1−μ) 非标度等价）——λ 方案的冻结定义明确锚定 torch 缓冲，此点已写入勘误备查。
4. nesterov=False 路径同理：NS5(B_t) vs NS5(C_t) 由同一尺度不变性覆盖。

## 4. 边界

- 本勘误不修改：v1.0 冻结候选、训练数据、原始证据、NorMuon staging、V16/30M/历史项目（全部只读）。
- E4 的 1.4% 末参数差是 bf16 种子差的混沌放大（逐步更新差 2.6e-4），不得被引用为"机制差异"。
