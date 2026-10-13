# P0_P1_PROTOCOL_v1_1 — 执行协议（已授权执行，本文件为开跑前冻结记录）

日期：2026-09-29。授权：`references/P0_P1_EXECUTION_ADDENDUM_v1_1_CN.md`（用户批准）。
冻结配置：`configs/p0_p1_v1_1.json`。数据/排列无预生成排列本轮（候选为组级 K 向量，
非排列）。本协议在新父训练前写定；`selection_lock.json` 在全部校准与成本锁定后、
任何 Q_P1_check 计算前写出。

## 1. 与原立项稿（MUON_PROBLEM_FIRST_PROJECT_BRIEF_v1）的差异（以补充协议为准）

| 项 | 原立项稿 | 本轮执行（ADDENDUM v1.1） |
|---|---|---|
| 每输入候选数 | 15（1+7×2）+2 全局幅度 | **45**（1 + 7组×2K×3探针 + 2 全局幅度）——增加三联幅度探针 |
| 联合比较 | 未规定 | 同预算池 + 一减一增联合探针 + 交互余项（必须实测） |
| AMO | "忠实实现或命名 geometry_diagnostic" | "本地original配方下的AMO机制一步对照"；来源不落实记 COMPARATOR_UNRESOLVED |
| 评价隔离 | Q_grad/Q_select/Q_check | Q_select / Q_P1_check / Q_P2_endpoint（本轮不读后者） |

## 2. 冻结要点

- **父配方**（逐字段读取自 `m1_muon_lr2.0_wd0.02_p0.5_s2030` 的 config/optimizer
  state）：Muon 峰值 LR 0.01 / WD 0.02 / β 0.95 / Nesterov / NS5 原系数 / original
  shape scaling；辅助 AdamW LR 0.01 / (0.9,0.95) / eps 1e-8 / WD 0.02 / decoupled；
  warmup 32 + cosine→0.1 共 2048 步；数据 p=0.5、manifest seed 2030、128 窗/步。
  **新父 seed=3101**（核对：3101 未被任何旧实验使用；两恢复点 512/1536 来自同一条
  轨迹，非两个 seed）。探针 LR = 恢复点下一步应消费的原时钟值（512→lr(513)、
  1536→lr(1537)），不用峰值/终值。
- **计数器验证**：2048×32640=66,846,720 有效 targets / 2048×32768=67,108,864 slots
  （父轨迹运行时断言通过，见 parent_summary.json）。
- **探针 batch**：状态 s 的 8 个 batch = 生产流 manifest[s:s+8)；前 4 校准（仅
  Q_select）、后 4 检查（仅 Q_P1_check，锁定后开放）。两状态 batch 互斥。
- **Q 集划分**（组级，先分组后抽窗）：`sha256('P0P1_Q_v1|'+gid)[0]%3` →
  select(0)/P1_check(1)/P2_endpoint(2)；组=conversation_id 翻译线程。实测：
  select A 676 docs/B 831（各 128 窗）、check A 498/B 628（128 窗）、endpoint
  A 639/B 781（256 窗）。旧 dev 曾用于 M1/M2——结果为开发级。confirm 封存。
- **组级单位**：7 组 × 8 层 = 56 矩阵；E_l(K)=‖Y Y^T−I‖_F/√r（r=min(m,n)），
  E_g²=组内均值；u_l=−lr·s_l·NS_K(H_l)（不含 WD）；ν_g(K)、γ_g(K)=ν(K)/ν(5)。
- **三联探针**：raw / gain_only_live（NS5 方向×γ）/ direction_only（NS_K 方向÷γ）；
  WD 与辅助组保持原生；ν(5)=0 或 γ=0 退化单记。
- **成本**：解析 FLOPs 4r²s+2r³/步/矩阵；同预算窗 [0.98,1.02]（解析与实测双卡尺，
  24 交替块中位数）；成本匹配只按校准输入/无目标计时确定，先于检查锁定。
- **联合（P1b）**：42 有序组对 → 成本可行池 P（10 对，全部同形状）→ ΣB_g 提案
  top2 → 4 校准 batch 联合实测 → target_primary；几何对照（校准 E_g² 改善最大）、
  fixed_random（预登记 seed 20260929 在读任何联合目标前抽取）；target 的
  gain_only_live/direction_only/gamma_const（校准中位 γ 冻结）三对照；联合不优于
  native → no_op。并列容差 1e-6：先 native、再低成本、再 (q,k,v,o,gate,up,down) 字典序。
- **筛查线**：δ_step = max(1e-5, 10×e_repeat) nats/有效target；e_repeat=同 native
  参数点重复评价最大差（实测 0）。Q1 局部支持要求：两状态 target_primary 检查
  均值 > δ_step 且 ≥3/4 同向且成本匹配。
- **AMO（P1c）**：论文 arXiv 2605.17806v2 本机不可达、服务器无缓存、webfetch
  失败（两种 URL 均超时）→ 预申报并实际判定 **COMPARATOR_UNRESOLVED**；不生成
  PE 系数、不做同族比较、不冒充 AMO；其余 P1 结果不受影响。
- **预算**：3 GPU-hours / 4 GiB / 1 条父轨迹 / 16 个诊断 train batch；失败尝试计入。

## 3. ROUND3 审计教训的落实（T2b 缺陷 → 本轮防护）

1. 参数参考在 load 后建立（theta_plus 于生产 opt.step() 后快照）；
2. `load_anchor` 显式 clone 全部 optimizer state 张量——**实测发现 AdamW `step`
   34/34 跨恢复共享 data_ptr**（momentum/m/v 本身不共享），修复后 0 共享且 payload
   不可变（双验证脚本通过）；
3. 实际消费的 G 做 step 前后 hash（断言不变）；
4. 每状态首个校准 batch 独立二次 fwd/bwd 采集对照（确定性验证）；
5. no-op 候选即生产 `opt.step()` 本身 + 合成对照测试（bit 级）；
6. 叠加构造的额外舍入以测试记录（≤1e-6，一次加法路径差）。

## 4. 交付物清单（任务书 §12）

`outputs/p0_p1_v1_1/`：parent/（3 状态+日志+summary）、calibration_rows.csv、
calibration_meta.json、cost_pool.json、selection_lock.json、final_schemes.json、
check_rows.csv、group_probe_table.csv、joint_allocation_table.csv、
amplitude_controls.csv、amo_comparison.csv、verdict.json、cost_ledger.json、图×3；
`docs/`：本协议、RECIPE_AND_AMO_COMPARISON.md、P0_P1_REPORT_CN.md；
测试、review zip。
