# STRONG_BASELINE_COMPARISON — 强基线接入与外部源审计（第二阶段）

- 日期：2026-10-12。载体：`configs/baseline_registry.json`（机器可读版，外源哈希已固定）。
- 纪律：缺权威实现的候选一律 **SOURCE_BLOCKED**，不以二手描述冒充复现（沿用 muon_spectral_allocation 2026-09-30 冻结记录）。

---

## 1. 源可达性实测（2026-10-11/12）

| 候选 | 官方源 | 获取尝试 | 结果 |
|---|---|---|---|
| MoonshotAI/Moonlight | github.com/MoonshotAI/Moonlight | webfetch ×1 | **Transport error（网络阻断）** |
| zichongli5/NorMuon | github.com/zichongli5/NorMuon | webfetch（github 域阻断）+ 本地磁盘全搜（/home/b205、site-packages） | **SOURCE_BLOCKED**（无本地副本） |
| microsoft/dion | github.com/microsoft/dion | 同上 | **SOURCE_BLOCKED** |
| SAMuon | arXiv:2608.25990 | webfetch ×1 | **Transport error → SOURCE_BLOCKED**（与 2026-09-30 记录一致） |

## 2. 各基线接入状态

### READY（4 个）

| 基线 | 实现 | 关键语义（已核实到代码级） | LR 网格（100M 候选） |
|---|---|---|---|
| Tuned AdamW | torch.optim.AdamW | M 线配方：β(0.9,0.95) eps 1e-8，wd 全参数（pilot 定 0.02） | {3e-4, 1e-3, 3e-3} + 边界规则 |
| Tuned Muon (original) | torch.optim.Muon，adjust_lr_fn=None | momentum 0.95/Nesterov/NS5 (3.4445,−4.775,2.0315)/eps 1e-7；**Nesterov 输入真值 = (1−μ)g+μB（lerp 代码语义，非其文档字符串公式）**；aux AdamW 同 LR 同 schedule wd 0.02 | {5e-3, 1e-2, 2e-2} |
| Tuned Muon (match_rms_adamw) | 同上，adjust_lr_fn="match_rms_adamw" | Moonshot RMS 对齐缩放 0.2·sqrt(max(A,B))——**torch 2.11.0 内置、文件内引用 arXiv:2502.16982（MoonshotAI），无需网络即权威可得**；这是 Moonlight 优化器级贡献的合法形态 | {1e-3, 3e-3, 1e-2}（有效 LR 放大 ~5.5×，网格相应下移） |
| Low-momentum Muon | GroupMuon(native, momentum∈{0.9,0.8}) | 对照臂：检验"方向混合收益 ≈ 降低有效动量"的解释 | 沿用 Muon 最优 LR |

**Moonlight 的分层结论**：优化器级（RMS 对齐缩放）READY；训练环级（MuonClip 的 QK-clip、权重范数目标管理）SOURCE_BLOCKED。不得把 match_rms 臂称为"完整 Moonlight 复现"。

### SOURCE_BLOCKED（3 个）

NorMuon、SAMuon-lite、dion：无本地实现 + 网络阻断。**不造数、不冒充、不引用二手描述做语义断言**。若未来网络恢复或用户提供源码包，先建复现对应表（来源哈希 → 逐字段语义 → 本地适配 diff）再升级为正式基线。

## 3. 公平性构造（防"打败未调优弱 Muon"）

1. 每个 READY 基线自带 ≥3 点 LR 网格 + 边界规则（best 落界 → `RANGE_UNRESOLVED`，单次外扩须新授权）；
2. 全臂：同 model-build seed（smoke 实测三路径初始化 fingerprint 一致 `33f93a0f`）、同缓存 cursor 0 CRN、同 schedule（warmup128+cosine→0.1）；
3. Muon 家族统一 wd 0.02 + aux AdamW 同 LR（M 线接线原样）；AdamW 臂 wd 0.02（M2 优于 0.01 的历史记录，披露）；
4. 对照臂（low-μ）与候选臂（Group-Muon-A/B）共享同一 Muon 最优 LR——候选只在与**各自最优配置**的比较中记分；
5. 30M 线（adamw_direction_scale30m）T1 已给出其 Muon 参照臂实测（读-only）；若其 Muon LR 网格边界未解，本项目 100M 网格按其 T2 结论修正（G-4）。

## 4. 与外部主张的关系边界

- 本项目不声称与 Moonlight/NorMuon/SAMuon/dion 论文数字可比（不同模型/数据/预算）；
- "击败 NorMuon/SAMuon-lite"目前**不可评价**（SOURCE_BLOCKED）；pilot 报告中相应指标行留空并标注原因，不删列。
