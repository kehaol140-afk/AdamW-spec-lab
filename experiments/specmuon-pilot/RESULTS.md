# RESULTS — specmuon-pilot（按时间顺序）

## R1 P0-A（10-10）：动量谱头部集中证实

`docs/P0_SPECTRAL_AUDIT.md` + `results/p0a/`：top-8 能量份额 0.389 vs 均匀 0.018（**21×**，H1 CONFIRMED）；56 矩阵逐个谱形分布（p0a_per_matrix.csv 全量随仓）。

## R2 P0-B（10-10）：AdamW 形状证实、合并断言证伪

`results/p0b/p0b_phase2_pairs.csv`（1624 行）：top-decile 0.86-0.96（H4 形状 CONFIRMED）；**合并 CV=0.227 > 0.2 → H4 合并断言 REFUTED**；P0-C：无合格梯度时间序列 → DATA_GAP，P2 永久封锁。

## R3 P1（10-11）：14 分支双恢复点全表（P1_RESULTS.csv；recompute CK SPECMUON_P1 一致）

| run_id | v_q H256 | nll_A | nll_B | Δ(native_lr1.0) | wall(s) |
|---|---:|---:|---:|---:|---:|
| s512_native_lr1.0（native 主对照） | 2.7989 | 2.6680 | 2.9297 | — | 48 |
| s512_native_lr0.5（**真杠杆**） | **2.7249** | 2.5973 | 2.8525 | **−0.0740** | 46 |
| s512_native_lr2.0 | 2.9167 | 2.7778 | 3.0556 | +0.1178 | 46 |
| s512_exact_polar | 2.7996 | 2.6687 | 2.9306 | +0.0007 | 262 |
| s512_head_lam0.25 | 2.8040 | 2.6730 | 2.9349 | +0.0051 | 264 |
| s512_bulk_lam0.25 | 2.7961 | 2.6650 | 2.9272 | **−0.0028** | 260 |
| s512_muon_p0.5 | 2.8501 | 2.7150 | 2.9852 | +0.0470 | 281 |
| s1536_native_lr1.0 | 2.4024 | 2.2806 | 2.5241 | — | 49 |
| s1536_native_lr0.5 | **2.3945** | 2.2739 | 2.5150 | **−0.0079** | 45 |
| s1536_native_lr2.0 | 2.4413 | 2.3160 | 2.5667 | +0.0389 | 45 |
| s1536_exact_polar | 2.4025 | 2.2807 | 2.5242 | +0.0001 | 280 |
| s1536_head_lam0.25 | 2.4034 | 2.2816 | 2.5251 | +0.0010 | 279 |
| s1536_bulk_lam0.25 | 2.4016 | 2.2798 | 2.5235 | −0.0008 | 251 |
| s1536_muon_p0.5 | 2.4152 | 2.2927 | 2.5376 | +0.0128 | 254 |

（Δ=v_q(臂)−v_q(native_lr1.0)，正=更差；nll_A/nll_B=配对域 A/B 终点。）

## R4 裁决（P1_DECISION.md:10-26）

- **Head λ=0.25：NO_EFFECT**（s512 +0.0051 / s1536 +0.0010，两恢复点均未过 2e-3 屏）
- **Bulk λ=0.25：SINGLE_POINT**（仅 s512 −0.0028 过屏，s1536 −0.0008 回落屏下 → 无确认效应）
- ExactPolar ≈ native（+0.0007/+0.0001）→ 无 POLAR_ONLY 信号；Power p=0.5 明确有害（+0.047/+0.013）
- **H5 三重门对照**：native lr0.5 在两恢复点 −0.0740/−0.0079——比 Head/Bulk 最好值大一个量级（P1_SPECTRAL_EFFECT.md:33-36）
- **终局：「No——不值得继续」**（P1_DECISION.md:26）；无 P2/AdamSpec/重跑

## R5 证据等级与边界

单父轨迹（3101）双恢复点；V_Q 256 docs 筛查级；无跨 seed；谱形操作限于 O=NS5(H) 域；SOURCE_BLOCKED 论文未核对（沿 muon-ns-allocation 决议）。原始数据：`results/p1/branches/<run_id>/{branch_meta,result.json,train_metrics.jsonl}`×14 随仓；P0 zip 含 3 个唯一成员保留、P1 zip 全重复省略（docs/DUPLICATE_ELISION_AUDIT.csv）。
