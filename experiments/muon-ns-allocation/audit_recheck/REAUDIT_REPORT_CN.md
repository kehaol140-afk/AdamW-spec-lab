# REAUDIT_REPORT_CN — P0/P1 v1.1 执行一致性与中断恢复复查

日期：2026-09-29。任务：`P0_P1_REAUDIT_TASK_CN.md`。原产物零改动；本轮全部输出在
`audit_recheck/`。

---

## 0. 总体结论：**PARTIALLY_VERIFIED**

执行链路（父轨迹→校准→成本→锁定→检查）**与已批准协议一致、结果 bit 级可复现
（19/19 定向复现点 |diff|=0，A→B→A 无顺序效应）**；但原报告有 **2 处表述错误**
（I1"事后池最优为负"、I2"γ≈0.98–1.02"）需更正。更正后：**Q1/Q2/Q3 三个裁决标签
全部维持**，其中 Q2 的幅度解释被审计**加强**（新发现：组级 gain_only_live 在 512
有超 δ_step 的 4/4 正收益）。Q3 依旧 COMPARATOR_UNRESOLVED（本轮未补做 AMO）。

## 1. 收到了什么：输入与上传失败的影响

- 任务书所指 `original/` 目录**未到达**（全盘检索无）；实际执行依据为
  `references/` 内六份文件（SHA 逐份记录于时间线 CSV）。关键时序（mtime 物证）：
  协议（ADDENDUM v1.1）与中文任务书 **14:44 在位**（任何运行之前）；四份参考中
  ROUND1 于 14:48 在位，原立项稿与 ROUND2/ROUND3 审计 **16:02 由用户补传**
  （父轨迹中段、全部 P1 分析之前）。**上传失败的是历史参考材料，不是执行协议，
  也不是结果包**（P0_P1_REVIEW_BUNDLE.zip 完好在服务器，276,990 B）。
- 后补文件未倒签任何预注册：`configs/p0_p1_v1_1.json`（14:48）与
  `docs/P0_P1_PROTOCOL_v1_1.md` 的冻结时间早于全部 P1 运行。
- 历史使用证据（哪些代码被实际 import）：所有最终产物 mtime 晚于对应源码最后
  修改；GPU 复现 bit 级一致进一步证明当前源码=运行源码。**UNKNOWN 项**：第一次
  selection_lock 与前两版检查表（380 行）无字节副本（被覆盖），仅有会话记录。

## 2. 中断、失败尝试与修复时间线（物证）

| 时间 | 事件 | 产物影响 |
|---|---|---|
| 15:42 前一次父轨迹前台运行被用户中止 | 无产物 | — |
| 15:55–16:05 | 父轨迹 seed3101 完成（512/1536/2048 三状态，计数器断言通过） | 不经过 load_anchor，**不受别名问题影响**（直连 build_muon_arm） |
| ~16:12 | 校准首次运行崩于 **AdamW step 别名断言**（34/34 共享 data_ptr 实测命中）——**发生在任何数据落盘之前** | 无产物 |
| 16:12–16:16 | 修复：load_anchor 显式 clone 全部 state 张量（验证 0 共享+payload 不可变）；restore_theta 键修复；CSV 字段修复 | — |
| ~16:15 | 校准第二次运行评价完成但 CSV 崩（字段）——**未落盘** | 无产物 |
| 16:17:40 | 校准成功落盘（362 行） | **全部最终数据自此均为修复后代码产物** |
| 16:18–16:19 | 成本基准（内嵌一次解包错误未落盘→独立脚本成功；24 交替块） | cost_pool.json |
| 16:22–16:24 | 检查 v1/v2 + 联合 v1 的迭代修复（native label、no-op、补 geometry/random 进检查侧）——中间版本被覆盖 | 会话记录为准 |
| 16:25:21 | selection_lock（终版） | **先于** check 16:26:35（锁定顺序物证 ✓） |
| 16:26:35–16:29 | 检查 396 行 + 分析表 + 图 | — |

**4 次失败尝试全部记账于原 cost_ledger；无修复前产物混入最终表**（所有最终文件
mtime 晚于全部源码修复；崩崩溃版本未落盘或被覆盖）。

## 3. 协议—实现—产物对照（requirement_trace.csv，10 条）

R1–R10 全部 **VERIFIED**（R6=VERIFIED_AFTER_FIX，影响范围=修复前仅一次无产物崩溃
运行）。要点：①候选计数逐格对上——校准 362=(45×4+1)×2、检查 396=192+204，
512 检查=45KJ+{no_op, geometry, fixed_random}、1536=45KJ+{target, gain, direction,
gamma_const, geometry, fixed_random}，(scheme,batch) 格零缺失；②42→10 可行对的
成本判定（解析 FLOPs=1.000+实测时间比 0.985–1.013 双卡尺）于 16:19 完成，
**先于任何检查目标**；③Q_P2_endpoint/confirm 全程零访问（access_guard 异常保护）。

## 4. 独立复算（result_recomputed.csv + selection_replay.json，未调用原 analyze）

- V_Q 恒等式：全部行从逐域 sum/count 复算一致（差<1e-15）。
- δ_step=max(1e-5,10×e_repeat)=1e-5（e_repeat=0 = 同参数点重复评价差）✓。
- 28 个单组 raw 检查收益与 group_probe_table 零差异；**全负复确认**。
- 联合/对照/几何/随机/幅度逐项复算一致；1536 target−gamma_const 直接配对差
  =+1.2e-7（两组各自都接近零，**不解释为等价证明**）。
- 交互余项 I(1536, cal) = −2.77e-6（联合 −1.44e-6 − 单组和 +1.33e-6）。
- **校准选择重放与历史 selection_lock 逐字段一致**（两状态 proposals_match=True、
  geometry_match=True、no_op 判定一致、γ 常量一致）。
- **池最优范围澄清（任务书 §4-83，原报告表述错误处）**：
  - raw K-重分配候选（14 单组×2 状态 + 全部实测联合 raw）检查最优 **为负** ✓
  - 含 no-op 的实测池最优 = 0（no-op 即 native）
  - **排除幅度参照后的全部实测点中，512 的 o|K3|gain_only_live = +8.90e-5（4/4，
    >δ_step）**，v/q/up/gate 的 gain_only_live 亦为正（4/4 或 3/4）；1536 同类
    +9.45e-6（4/4，略低于 δ_step）。这些是**诊断对照**（NS5 方向 × γ_g(K3)，
    γ=0.75–0.98 即组级轻缩幅），不是 K-重分配候选，也不满足两状态 δ_step 标准。

## 5. 定向 GPU 复现（replay_comparison.csv）

19 点（两状态第一个检查 batch、任务书预定清单：native×2、q K3/K7 raw、q K3
gain/direction、1536 target/gamma_const/geometry、512 o|K3 gain_only_live、
global×0.5、几何去重）——**全部与历史 bit 级一致（max diff=0.000e+00）**；
A→B→A 两状态 diff=0。GPU 消耗 ~40s（预算 15 分钟）。

## 6. 逐题结论

| 题 | 执行完整性 | 可复现性 | 科学结论保留范围 |
|---|---|---|---|
| Q1 | 完整（45 点×8 batch×2 状态 + 联合实测） | bit 级复现 | **NO_LOCAL_SUPPORT 维持**：raw K=3/5/7 家族（单组+已测联合）无同预算一步收益；限定为该固定系数家族+本设置。注：仅实测 10 可行对中的 4 个联合（2 提案+几何+随机），**不推断全部 42 有序对** |
| Q2 | 完整（三联+全局幅度+gamma_const） | bit 级复现 | **SCALAR_EXPLANATION_NOT_RULED_OUT 维持并加强**：γ(K3)=0.75–0.98 的组级缩幅在 512 拿到 >δ_step 的 4/4 收益，与全局×0.5(+3.4e-4) 同向——幅度（LR 过冲）解释有效；K3 的正效应全在幅度分量。不证明全部逐层/逐矩阵倍率被排除 |
| Q3 | **未执行**（AMO/PE 无来源） | n/a | **COMPARATOR_UNRESOLVED 维持**；本轮未补做；需材料=AMO v2 Alg.1/2、Appendix C PE composer 及其实现（获取后仅阻塞 Q3，不影响 Q1/Q2） |

**不需要重新训练**：全部偏差为报告表述层，无任何 run 受污染（复现兜底）。
唯一需更正动作：原 REPORT_CN §7"连事后池最优都为负"与 §2"γ≈0.98–1.02"两处
（已在 issue_impact.csv I1/I2 给出更正文本；原文件保留不覆盖）。

## 7. 本轮成本

CPU 复算+审计 <2 分钟；GPU 复现 40s（预算 15 分钟）；新增输出 <1 MB（预算 20 MB）。
未进入 P2、未补做 AMO、未新增任何 K/LR/seed/系数/数据。
