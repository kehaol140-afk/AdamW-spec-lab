# 100M_PILOT_PROTOCOL_DRAFT — 100M 先导协议（草案，DRAFT_NOT_FROZEN）

- 日期：2026-10-12。机器载体：`configs/pilot_draft_v0.json`。**状态：草案——冻结前置见 §7。**
- 设计原则：先审计成本（已完成，100M_COMPUTE_AND_MEMORY.md）→ 固定候选与评价规则 → 有限授权开跑。

---

## 1. 共同基底（全臂一致）

| 项 | 值 |
|---|---|
| 模型 | MiniMind 100,784,000（h768/13L/12Q-6KV/FFN2432/seq256/tied） |
| 初始化 | seed 3471（model-build seed 跨臂共享，位级同 init——smoke 已验证 CRN 机制） |
| 数据 | pretrain_t2t_mini 确定性缓存，各臂 cursor 0 起（同 seed CRN） |
| 时钟 | warmup 128 + cosine→0.1×peak，T=4096 步 |
| 批形状 | micro 8 × accum 4 × seq 256 = 8192 tok/步 |
| 精度 | bf16 autocast，fp32 主参数/优化器态 |
| pilot 预算 | **33,554,432 tokens/run（0.33 tok/param）** |

## 2. 候选矩阵（15 runs，单 seed）

| 臂族 | runs | 配置 |
|---|---|---|
| B_adamw | 3 | LR {3e-4, 1e-3, 3e-3}，wd 0.02，β(0.9,0.95) |
| B_muon | 3 | LR {5e-3, 1e-2, 2e-2}（original 缩放，wd 0.02，aux 同 LR） |
| B_muon_rms | 3 | match_rms_adamw 缩放，LR {1e-3, 3e-3, 1e-2}（Moonlight 优化器级） |
| B_lowmu | 2 | μ∈{0.9, 0.8} @ B_muon 最优 LR（动量解释对照） |
| C_A | 2 | Group-Muon-A λ∈{0.65, 0.85}，norm_match=True @ B_muon 最优 LR |
| C_A_nat | 1 | 最优 λ，norm_match=False（自然范数变体） |
| C_B | 1 | Group-Muon-B @ 最优 λ（FFN39+V/O26） |

选择规则（冻结草案）：LR/λ 最优点 development H4096 token 加权 NLL；差 <1e-4 并列取保守端；best 落网格边界 → `RANGE_UNRESOLVED` 标志（单次外扩须新授权）。

## 3. 评价（七项指标，对应任务书）

1. **同 token 预算终点 NLL**（主；development 256 docs，H∈{1024, 2048, 4096}）；
2. 相对最佳 tuned Muon 的改善（配对逐文档 bootstrap 1000 reps，seed 9071）；
3. 相对可用强基线的改善（match_rms/low-μ/AdamW；**NorMuon/SAMuon-lite 行保留但标注 SOURCE_BLOCKED，不虚填**）；
4. 达到共同损失阈值所需 token（阈值 = 最佳 Muon 臂 H4096 终点 NLL，预注册锁定）；
5. 真实 wall-clock + 优化器额外开销（opt 秒分列 NS5 native/mix、fp64 诊断）；
6. 显存 + 优化器状态内存（torch.cuda.max_memory_reserved + 状态字节计数）；
7. 参数组更新尺度 + 梯度一致性（cos(O,g)/cos(O,O_nat)/逐组 ‖Δ‖，锚点步采样）。

## 4. 门与停止（预注册草案）

**Pilot 通过（进入确认层）**：C_A 或 C_B 对最佳 B_muon 的 H4096 差 ≤ −1e-3 且 CI95 全负，**并且** low-μ 对照未能复现同量级收益（排除纯动量解释）。

**Pilot 失败（关闭线）**：
- 无候选过门 → `NO_GAIN_OVER_STRONG_MUON`，写结题，不开确认层；
- 收益被 low-μ 复现（差距 <1e-3）→ `MOMENTUM_EXPLAINED`，同上关闭；
- LR 网格穷尽且边界未解 → `RANGE_UNRESOLVED`，报告停止。

**短预算条款（任务书要求，预注册）**：0.33 tok/param 属瞬态训练区；pilot 任何正/负结果均不表述为完整预训练终点收益。确认层（100–300M tokens，≤3 方法 × 3 seeds + 封存评价）才是终点裁决层。

## 5. 确认层骨架（gate 后细化冻结）

- 臂：最佳 C 候选 + 最佳 Muon 基线 + （若不同）match_rms 变体，× seeds {3472–3474}；
- 预算：100.7M tokens/run 起（≤301.9M 视语料 epoch 纪律：3 seeds × 100.7M = 302M ≈ 0.95 epoch 无重复 ✓）；
- 评价：development 曲线 + 封存集一次开封（三选项：dv holdout_v2 协调 / 新铸 holdout_v3 / dev-only 降级披露）；
- 推断单位 = 训练 seed。

## 6. 明确不做（第一轮）

动态谱指数；门控网络；每步 SVD（源扫描强制）；MoE GPU 训练；修改历史项目；跨模型 NLL 绝对值比较；把 pilot 短预算信号当终点结论。

## 7. 冻结前置清单

| # | 项 | 状态 |
|---|---|---|
| F-1 | fp64 诊断降采样（G-1）+ 100 步复测吞吐 | 待办（小时级） |
| F-2 | V16 文档分区规则 re-vendor + 33.6M 正式缓存铸造（G-3） | 待办（CPU 小时级） |
| F-3 | 确认层评价方案三选一 | 冻结时定 |
| F-4 | 30M 线 T2 结论落地后复核 Muon 家族 LR 网格（G-4） | 等外部 |
| F-5 | 代码哈希冻结（src+tests+configs）+ 用户授权 pilot ~5.2–6.2 GPU-h | 待授权 |
