# 100M_PILOT_EXECUTION_LOCK — 正式 Pilot 执行矩阵冻结 v1.0（任务四）

日期：2026-10-12。机器载体：`configs/pilot_arms_v1.json`（15 臂，逐臂 spec + 选择规则 + gate 全文）。代码状态：`configs/CODE_FREEZE_V1.json`（18 文件 sha256）。

---

## 1. 与草案 v0 的差异报告（任务书要求：有差异先报告，不改队列）

**结构性差异：无。** 15 runs、臂族构成、每族网格与草案 v0 完全一致（逐格核对 `pilot_draft_v0.json` vs `pilot_arms_v1.json`）。

**文本级修订（不改变任何臂的运行配置，全部来自本轮 preflight 的证据）**：

| # | 修订 | 依据 |
|---|---|---|
| A1 | fp64 方向诊断由每步改为 **diag_every=64**（范数匹配无条件保留） | F-1 关闭：每步诊断 ~0.05 s/步 × 4096 × 4 mix 臂 ≈ 14 min 纯测量开销；64 步采样保留轨迹信息 |
| A2 | 显式冻结**全臂无梯度裁剪** | 此前未成文（smoke/诊断均未裁剪）；成文以防解释歧义 |
| A3 | 并列规则具体化：B_muon 差<1e-4 取**更小 lr**、C_A 差<1e-4 取 **λ=0.65**（保守端=更近锚点） | 草案只写"保守端"未定义 |
| A4 | 新增 `interpretation_rule_rms`：若最佳 Muon 锚点是 match_rms 族且 C 臂对其过门，则确认层**必须**补 C@match_rms 变体后才可声称"击败最优约定 Muon" | 防止跨尺度约定不匹配的过门解读 |
| A5 | `t2_deviation_clause`：不等待 30M T2 的版本化偏离登记 | T2 无 Muon 臂；Muon LR 边界已由 30M_VS_MUON 解析（100M_FAIR_BASELINES §5） |
| A6 | 缓存身份从"待铸（G-3）"改为已铸 sha `f92715b4…`（V16+s30m 前缀字节证明） | F-2 关闭 |

**冻结前置清单终态**（pilot 草案 §7）：

| # | 项 | 状态 |
|---|---|---|
| F-1 | 诊断降采样 + 100 步复测吞吐 | ✅ 关闭（diag_every=64；28.4k/26k tok/s 实测） |
| F-2 | V16 分区 re-vendor + 33.6M 缓存 | ✅ 关闭（前缀字节证明） |
| F-3 | 确认层评价三选一 | 按设计推迟至确认层授权（pilot 不消耗封存集） |
| F-4 | Muon 网格的 30M 依赖 | ✅ 关闭（见 A5；T2 无 Muon 臂） |
| F-5 | 代码哈希冻结 + 用户授权 | 哈希 ✅（CODE_FREEZE_V1.json）；**授权待用户** |

## 2. 执行队列（授权后，单 GPU0 连续窗口）

| 顺序 | 臂 | 依赖 |
|---|---|---|
| 1–3 | B_adamw × 3 | 无 |
| 4–6 | B_muon × 3 | 无 |
| 7–9 | B_muon_rms × 3 | 无 |
| — | 选择步：`selection.json` 写入 B_muon_best_lr（dev H4096，规则冻结于 arms 文件） | 4–6 完成 |
| 10–11 | B_lowmu {0.9, 0.8} @ B_muon_best_lr | 选择步 |
| 12–13 | C_A λ{0.65, 0.85} @ B_muon_best_lr | 选择步 |
| — | 选择步：C_A_best_lam | 12–13 完成 |
| 14 | C_A_nat @ best λ | 两选择步 |
| 15 | C_B @ best λ | 两选择步 |

- Runner：`scripts/run_pilot_arm.py --arm <arm_id>`；逐臂 run.lock 防重跑；产物 `results/pilot/<arm>/{metrics.jsonl, endpoint_weights.pt, run.json}`。
- 选择步只读 development（dev 级，允许）；封存集零消耗。

## 3. 预算（按 preflight 稳态实测重估）

| 臂族 | runs | 单 run 估 | 小计 |
|---|---|---|---|
| AdamW | 3 | ~21 min | ~63 min |
| Muon original | 3 | ~19.5 min | ~58 min |
| Muon rms | 3 | ~19.5 min | ~58 min |
| low-μ | 2 | ~18 min | ~36 min |
| C_A ×2 + C_A_nat + C_B | 4 | ~20 min | ~80 min |
| 评价（256 docs × 3 视界，含在上面） | — | — | — |
| **合计** | **15** | | **~4.9–5.5 GPU-h**（单晚窗口） |

磁盘：端点权重 403 MB × 15 ≈ 6.0 GB + 日志。收缩规则：任一臂墙钟超估 +20% 即停并重报。

## 4. 判定（冻结文本与草案一致，见 `pilot_arms_v1.json` gates + `100M_PILOT_PROTOCOL_DRAFT.md` §4）

- 通过：C_A/C_B 对 best_muon_anchor（两种约定中较优者）dev H4096 改善 ≥1e-3 且逐文档配对 bootstrap CI95 全负，且 low-μ 未复现同量级收益。
- 失败词：NO_GAIN_OVER_STRONG_MUON / MOMENTUM_EXPLAINED / RANGE_UNRESOLVED。
- 短预算条款：0.33 tok/param 瞬态区，pilot 结果不作完整预训练终点主张；确认层（100–300M tokens ≤3 方法 × 3 seeds）为终点裁决层。

## 5. 状态

**FROZEN_PENDING_AUTHORIZATION。** 本轮到此为止（gate 停止），不自动执行 15 臂。

## 6. 冻结后增补提案 v1.1（2026-10-12 晚登记，不属于 v1.0 队列）

并行审计（`audits/context_muon_baselines_20261012/`，只读）将 NorMuon 升级为 ADAPTED_FROM_LOCAL_MIRROR（作者贡献版 + 18/18 staging 测试）、Moonlight toy_train 镜像可实例化。**v1.0 队列与 CODE_FREEZE_V1 不变**；增补选项见 `configs/pilot_amendment_v1_1_PROPOSAL.json`：

- 最小增补：B_normuon_ns5vr ×1（受控臂，不宣称官方身份）+ 吞吐 smoke ≈ **+0.35 GPU-h**；
- 完整增补：+faithful ×3（待上传门 B-1）+ moonlight 镜像 ×3 ≈ +2.6–3.4 GPU-h。

采纳与否由用户在授权时决定；采纳则先 vendor+sha 登记、18 项 CPU 测试本仓复跑、30 步 NorMuon smoke。
