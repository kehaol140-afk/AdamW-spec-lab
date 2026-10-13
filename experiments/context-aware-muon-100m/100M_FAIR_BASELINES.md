# 100M_FAIR_BASELINES — 基线/数据/评价链前置核验（任务二）

日期：2026-10-12。所有 sha 为本轮实测（证据 ZIP 含原始命令输出）。

---

## 1. 训练语料、tokenizer、数据流哈希（身份链）

| 层 | 资产 | sha256 | 状态 |
|---|---|---|---|
| 语料 | `/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl`（1,241,043,656 B） | `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c` | ✅ 与 registry 一致 |
| tokenizer | `minimind_base/model/tokenizer.json` = `adamw_spectral_pilot/minimind_repo/model/tokenizer.json`（BPE vocab 6400, BOS=1 EOS=2） | `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668` | ✅ 多副本一致 |
| 分区规则 | V16 prepare()：normalize→doc_id=sha256→bucket=id[:16]%100→cuts 70/76/82/94/97/100；仅 train(70%) 入训练流 | re-vendor 自 `adamw_spectral_pilot/prepare_data.py`（`scripts/mint_pilot_cache.py`） | ✅ **F-2 完成** |
| 正式缓存 | `artifacts/pilot_train_tokens_33p7m.u32`（33,685,504 tokens = 4096 步 × 8224 窗口 token，精确无回绕） | `f92715b4a66c2db140ab6785d8a055ccb808fd5c0fb854cb89cd11a0613a0d8e` | ✅ 前 8.5M 与 campaign_v16 缓存**字节相同**；前 17M 与 s30m 缓存**字节相同**（`pilot_cache_manifest.json`） |
| 打包 | 每 train 文档 [BOS]+enc+[EOS] 拼接；(seq+1) 窗口 stride=seq | 与 V16/s30m 同族 | ✅ |

结论：数据流身份链**从语料字节到训练窗口全闭环**，且与 V16/30M 两代前作前缀字节可证。

## 2. 独立开发评价与最终确认集

| 集 | 规模 | 身份/状态 | 用途 |
|---|---|---|---|
| development（V16） | 256 docs（45,470 tok，本轮实测） | `campaign_v16/eval_sets.json`；dev 级已披露（V16 B 曾用于选优，只作开发比较） | pilot 主指标（H∈{1024,2048,4096} token 加权 NLL，dv `loss_sum_count` 语义：单移位、显式长度掩码、fp32 CE） |
| holdout_v2 | 1024 docs | dv 线已铸**MINTED_SEALED**（sha `3e0f7653…`），未开封 | 确认层选项 (a)：与 dv 项目 + 用户两方协调开封 |
| holdout_v3（新铸） | untouched 池（本轮扫描口径：branch_eval_reserved 9,795 docs 可用） | 未铸 | 确认层选项 (b) |
| dev-only 降级 | — | — | 确认层选项 (c)：披露降级 |
| 旧 final_holdout 1024 | — | **2026-10-10 已开封（BURNED）**，禁止复用 | 仅历史引用 |

**确认层三选项维持 pilot 草案 §5，授权确认层时决定**（pilot 本身不消耗任何封存集——dev 256 是其唯一评价集，选择行为发生在 dev 上，符合 dev 级披露纪律）。

## 3. Nesterov 代码语义与实际更新一致性

- 真值 = `(1−μ)g+μB`（lerp 代码），非 torch 文档字符串公式——本轮再次逐行确认并已在 `MUON_MOMENTUM_EQUIVALENCE.md` 全文推导。
- native 位级等价（`test_native_bitwise_vs_torch_muon`）+ 两处过时文档表述已修正（METHOD_DESIGN 伪代码、group_muon docstring）。
- λ 锚点 λ₀=1−μ=0.05 位级路由；λ 不触碰动量态；范数匹配实测 1.87e-16。

## 4. Muon LR 网格与 match_rms_adamw 尺度约定（数值核验）

**尺度乘子（每矩阵有效 lr = peak × 乘子）**：

| 矩阵形状 | original（Keller） | match_rms_adamw（Moonlight） |
|---|---|---|
| gate/up 2432×768 | ×1.7795 | ×9.8631 |
| down 768×2432 | ×1.0000 | ×9.8631 |
| q/k/v/o 768×768 | ×1.0000 | ×5.5426 |

- original 是**非对称**缩放（gate/up 独享 1.78×）；match_rms 对 max(A,B) 对称。两族各自内部自洽，跨族 lr 不可直接比数字，只可比有效区间。
- 网格有效区间核对（gate/up 为最大乘子矩阵）：B_muon original {5e-3,1e-2,2e-2} → 有效 {8.9e-3, 1.78e-2, 3.56e-2}；B_muon_rms {1e-3,3e-3,1e-2} → 有效 {9.9e-3, 2.96e-2, 9.9e-2}。**两族有效区间重叠且 rms 族上探更远**（Moonshot 主张 match_rms 可复用 AdamW 型 lr——我们的 rms 网格覆盖 AdamW 网格上移区）。
- **外部证据**：30M 线参照臂（同 torch Muon、同网格、同批形 8192、wd 0.02、16.8M tokens）实测 **lr 1e-2 内点最优**（`30M_VS_MUON.md`，只读）——原 M 线 0.02 边界悬案在 30M 已解析。100M 沿用同网格有跨规模证据支撑。

## 5. 30M T2 门槛状态与版本化偏离（任务书要求项）

- 现状：30M 线 T1 已结（gate PASS）；**T2 已提案未执行**（臂 S1/S-a*/F2/P2 × seeds 3422–3424，等用户授权）。
- 本项目 F-4 原文是"Muon 家族 LR 网格参照 30M 线 T2 结果微调"。**核实：T2 提案不含任何 Muon 臂**；Muon LR 边界问题已由 T1 的 Muon 参照臂（内点最优）解决。**F-4 就此关闭，无需等 T2。**
- 30M 线的 CROSS_SCALE_DECISION_GATE 管辖其 AdamW F/P 方向线是否上 100M，**不管辖本 Muon 几何项目**（不同研究线、不同协议）。
- 按任务书要求，本偏离已**版本化预注册**于 `configs/pilot_arms_v1.json` 的 `t2_deviation_clause`（冻结文本，非口头声明）：pilot 在不等待 T2 的前提下冻结，依据与边界如上。

## 6. 强基线公平性（重申 + 本轮新增证据）

| 规则 | 证据 |
|---|---|
| 同初始化 | 四路径（本轮）/三路径（smoke）init fingerprint 全 `33f93a0f546149a0` |
| 同数据流 | 同缓存 cursor 0；首 3 步 loss 位级同值（CRN） |
| 同时钟 | warmup 128 + cosine→0.1×peak，全臂同一 T |
| 每族自带 LR 网格 | AdamW {3e-4,1e-3,3e-3} / Muon original {5e-3,1e-2,2e-2} / Muon rms {1e-3,3e-3,1e-2}（有效区间 §4） |
| 边界规则 | best 落界 → RANGE_UNRESOLVED，单次外扩须新授权 |
| 候选 LR 公平性 | C_A/C_B/lowmu 共享 B_muon 最优 lr（范数匹配使逐矩阵位移幅度 == native 同步 → 同 lr 尺度是公平默认；网格纪律披露于 lock 文档） |
| 未调优比较禁令 | 无臂使用未调参对手；NorMuon/SAMuon-lite/dion 行保留 SOURCE_BLOCKED 标注，不虚填 |
| 逐矩阵更新范数控制 | norm_match（方向效应隔离）+ natural-norm 变体臂 |
| 统一无裁剪 | 全臂不做梯度裁剪（含 AdamW），pilot_arms_v1 冻结披露 |

## 7. 裁决

数据/评价/语义/网格四链全部闭合；唯一开放项（确认层评价三选一）按设计推迟到确认层授权时。**无前置阻塞。**

## 8. 冻结后增补（2026-10-12 晚）：并行审计带来的基线状态变化

并行会话产出 `audits/context_muon_baselines_20261012/`（本项目只读，其自述"不代改 context_aware_muon"）：

| 基线 | v1.0 冻结时 | 并行审计后 | 对 pilot 的影响 |
|---|---|---|---|
| NorMuon | SOURCE_BLOCKED | **ADAPTED_FROM_LOCAL_MIRROR**：作者贡献版在 modded-nanogpt（PR #276，@lliu606，arXiv:2510.05491），staging 适配器 18/18 CPU 测试；faithful 臂待官方仓上传比对（其 B-1），**ns5+VR 受控臂现在即可用**（不宣称官方身份） | 不改 v1.0 队列；以 `configs/pilot_amendment_v1_1_PROPOSAL.json` 提交增补提案（最小 +1 臂 ≈ +0.35 GPU-h；含其 B-2 吞吐 smoke） |
| Moonlight | 仅优化器级（match_rms） | toy_train 语义本地双镜像可实例化（动量世系 = full-grad，与 Keller-lerp 分叉已测试锁定） | 同上（可选 +3 臂分解世系因素） |
| SAMuon-lite / dion | SOURCE_BLOCKED | 维持不变 | 指标行留空标注 |

**纪律说明**：按任务书"实际候选与草案不同→先差异报告与新预注册版本，不直接改队列"——v1.0 队列与 CODE_FREEZE_V1 保持原样，增补以独立提案文件登记，决策权在用户授权时。若采纳 v1.1，vendor+sha 登记 + 18 项 CPU 测试本仓复跑 + 30 步 NorMuon smoke 先行，任何冻结文件不被静默修改。
