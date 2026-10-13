# 100M_BASELINE_AUDIT — 100M 实验底座审计（第一阶段）

- 日期：2026-10-12。项目：`context_aware_muon`。本轮零修改历史项目；GPU 仅消耗授权的有限 smoke（§5）。
- 上游输入：`audits/cross_scale_100m_prep_20261011/`（100M 准备报告族）、`legacy_fluctuation_response/outputs/generalization_g1b_100m_v01/`、`optimizer_sensitivity/docs/F4A_EXISTING_BANK_INVENTORY.md`、`adamw_direction_scale30m/`（只读）。

---

## 1. 可复现的 100M 模型结构与参数量（本机实测复核）

| 项 | 值 | 验证 |
|---|---|---|
| 结构 | MiniMind h768/13L/12Q-6KV（头维 64，GQA 2:1）/FFN 2432（冻结公式）/vocab 6400/tied | `MiniMindConfig` 实例化，参数计数 **100,784,000 断言通过**（tests/test_model.py） |
| 参数分组 | FFN 39 矩阵（3×13）+ V/O 26 + Q/K 26 = **91 Muon 矩阵** + **54 aux 张量**（embed 4.92M + final_norm + 13×4 norms） | tests/test_routing.py（互斥/全并/tied 单对象） |
| 初始化确定性 | 同 seed 两次构建位级相同（fingerprint 一致）；不同 seed 不同 | tests/test_model.py |
| 30M 回归结构 | 30,025,216（h512/8L/FFN1664）同套断言通过——本项目 30M 仅作实现回归 | 同上 |

## 2. 从随机初始化训练的真实代码（本轮已建立并验证）

- `src/context_aware_muon/`：model_100m.py（构建+指纹）/ routing.py（分组）/ ns5.py（**vendored torch NS5，逐操作一致**）/ group_muon.py（多矩阵优化器）/ data.py（token 缓存+窗口流+schedule）。
- **native 路径位级等价**：GroupMuon(native) 与 `torch.optim.Muon` 在随机矩阵 3 步更新后参数位级相同（tests/test_group_muon.py::test_native_bitwise_vs_torch_muon）。
- GPU 侧 smoke 已跑通 100M 前向/反向/优化全链（§5）。

## 3. tokenizer、语料、训练流、验证集

| 项 | 状态 |
|---|---|
| tokenizer | minimind BPE `71f32c68…`（vocab 6400），三处副本 sha 一致（30M 线已核） |
| 语料 | pretrain_t2t_mini.jsonl（服务器副本 sha `6dd6716c…`；**~320M unique tokens**，2026-10-11 抽样实测） |
| 训练流 | uint32 确定性缓存 + cursor 0 窗口流（无 DataLoader 随机性）；smoke 缓存 1.2M tokens 已铸（sha 入 meta）；**正式 pilot 冻结前必须 re-vendor V16 文档分区规则**（gap G-3） |
| 验证集 | pilot 用 V16 development 256 docs（dev 级、历史选优已披露，只读复用）；确认层评价三选项见 pilot 草案（与 dv layer-2 协调开封 holdout_v2 / 新铸 holdout_v3 / 降级 dev-only） |

## 4. GPU0 资源与环境

- RTX 5090 32G；GPU1 被外部 vLLM 长期占用（27.8GiB）——**本项目只用 GPU0**；30M 线 T1 已完成 13/13（raw/endpoint_hashes 2026-10-12 00:22），GPU0 空闲后本项目执行了 smoke。
- 环境 `fr_minimind`（torch 2.11.0+cu13.0，bf16 可用）；无外网（arxiv/github 全阻断，复测 2026-10-11/12）。

## 5. 各方法显存/状态内存/吞吐（GPU smoke 实测，非沿用旧估算）

30 步 × 8192 tok/步 × 3 路径（seed 3471，同初始化 fingerprint `33f93a0f` 三路径一致 → CRN 成立）：

| 路径 | s/步 | tok/s | VRAM 峰值 | opt 秒/步（含 NS5） |
|---|---|---|---|---|
| tuned AdamW（占位 LR 1e-3） | 0.320 | **25.7k** | 3.35 GiB | 0.010 |
| native Muon（91 矩阵）+ aux | 0.283 | **29.1k** | 3.02 GiB | 0.052（NS5 0.045） |
| Group-Muon-A（FFN39 mix λ=0.75 + 范数匹配） | 0.345 | **23.8k** | 2.98 GiB | 0.117（native NS5 0.022 + mix 双 NS5 0.037 + fp64 诊断） |

- 全路径 loss 正常下降（8.9→7.2，30 步）；范数匹配误差 **1.4e-16**；λ=0.75 时末步 cos(O_mix,g)=0.184、cos(O_mix,O_nat)=0.424（早期训练方向诊断基线，供 pilot 对照）。
- 优化器状态内存：Muon 动量 buffer 91×fp32 ≈ 403 MiB；AdamW m+v 全参数 806 MiB；合计显存 <4 GiB（32G 充裕）。
- **实测吞吐（23.8–29.1k tok/s）显著高于此前参数比外推的 9–16k**——大矩阵更高效；旧 100M 准备报告的计算口径不得沿用（任务书要求），本项目以本表为准。

## 6. G1b bank 裁定（重申并落实）

G1b 9 态（每态 1.229 GB）= 状态诊断资产（D0 4096 行、131k tokens 见世、AdamW(0.9,0.999)/wd0.01/seq128/1024 tok-步）。**本项目全部臂从随机初始化训练（seed 3471 块），零接触 G1b 状态**；代码仅引用其结构配置作为 100M 结构的已验证出处。

## 7. 就绪缺口

| # | 缺口 | 状态 |
|---|---|---|
| G-1 | fp64 方向诊断的每步开销（mix 路径 +22% 墙钟的一部分）降为锚点步采样 | 代码 TODO（pilot 冻结前完成，预期 mix 路径回到 ~0.30 s/步） |
| G-2 | 确认层评价方案三选一 | 冻结时决定 |
| G-3 | V16 文档分区规则 re-vendor（正式数据缓存） | 冻结前完成 |
| G-4 | Muon 家族 LR 网格参照 30M 线 T2 结果微调（若其 Muon 参照臂边界未解） | 等 30M 线结题 |
