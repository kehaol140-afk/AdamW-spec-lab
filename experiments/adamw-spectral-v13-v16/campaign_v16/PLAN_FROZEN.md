# V16 PLAN_FROZEN — ADAMW_SPECTRAL_CAMPAIGN_V16_20261002

冻结时间：2026-10-02（OpenCode，合同 PROMPT_OPENCODE_V16.md）。运行前冻结；只有合同明确允许的顺序决策（B 阶段名额规则、路由）可按预写规则执行，其余一律不变。

## 阶段与预算（新增，历史账目单列不清零）

| 阶段 | 内容 | 更新上限 |
|---|---|---|
| A0 | V15 H0 来源更正（零训练重评 + INVALID_STATE_ORIGIN 标记）、V15 台账差异补充说明、runner/队列/身份验证、回归测试 | 32 |
| A1 | step256 分叉 H=256（父全局 257..512 调度）：S/F/P × a∈{1,2,4,8} × b∈{0.5,1,2} = 36 条；development 评 H64/H256；每族按 H256 development 选 (a*,b*) | 9,216 |
| A2 | 共同目标范数轨迹对照：F_dir@q_S、P_dir@q_S、S_dir@q_F、S_dir@q_P 各 256 步（q_t=||eta_t·a_S·D_t|| 标量序列，来自 A1 同系数三条运行） | 1,024 |
| B | 从头 2048 步（warmup128→cos 0.1peak，seed3407，新 8.5M token 缓存）：S≤8 / F≤5 / P≤5 名额按合同 §6.B 预写规则；dev 记录 H512/1024/2048；冻结后 screen 评 | 36,864 |
| C | ≤4 配方 × seeds{3408,3409,3410} × 4096 步（warmup256）；全部终点完成并身份锁定后 final_holdout 一次性开封（1024 文档，H4096） | 49,152 |
| D | 机制材料：B 关键轨迹锚点(10/25/50/75%)×3 矩阵训练中流式记录 cos(g,m)/cos(g,A)/范数/头部能量份额/输入加权输出能量；SVD≤48；可选分路按合同路由 | 8,192 |

硬上限（先到即停）：72h 墙钟；24h 设备任务；120,000 次 optimizer.step 尝试；400M 前向 token-equivalents。磁盘软 3GiB/硬 4GiB（保持 ≥5GiB 空闲）；full checkpoint ≤8、weights-only ≤12。RAM 1.5 准入/1.5 告警/1.0 硬停不降；VRAM 最低保留 0.2GiB（V15 授权沿用），启动需 free ≥ max(1.0, 峰值+0.3)。资源暂停累计 2h → STOP_RESOURCE。同类非资源异常最多两次修复重试。

## 数据（先封存后运行）

- 训练缓存：父 1,052,996 token 前缀 + 流式扩展 train 角色新文档至 ≥8.5M token（uint32 memmap，≤80MB）。A1 从父游标 524,288 续（256×2048 恰好用尽父缓存）；B/C 从 0 开始。
- development=256 自然前缀文档（V12 旧 dev 128 + probe_select 新 128，字典序）；screen=随后 256 个未用 probe_select；final_holdout=1024（final_holdout_reserved 角色优先，不足用未用 probe_confirm 补，排除 V1/V12/V13/V14/V15 一切已用 ID）。全部按 doc 整篇分组、train 角色永不入评估。
- 长文自然截取：[BOS]+全文+[EOS] 再截 seq+1，不造 EOS；mask/计数与父管线一致。

## 方法语义（与 V15 一致 + 新增）

S: D=A=m̂/(√v̂+eps)；F: D=N_A(g_clip)；P: D=N_A(g_clip/(√v̂+eps))；N_A 匹配本分支当前 A 的 Frobenius 范数。目标块位移 −eta·a·D，其他参数 −eta·b·A_other，衰减 (1−eta·wd) 独立。a=0 不在本轮网格。低 beta1：真实 AdamW 状态+bias correction，从头构建；目标块 beta1={0.5,0} 与全模型 beta1=0.5 分别标注，幅度固定用 A1 的 (a_S,b_S)。A2 范数覆盖：目标位移直接归一到外部 q_t（不重复乘 eta/a），其他参数自然。

## B 名额预写规则（运行前冻结，按合同 §6.B）

- S1-3: S(1,1)×LR{0.001,0.003,0.006}；F1-3: F(a_F,1)×同 LR；P1-3: P(a_P,1)×同 LR（a_F/a_P 取 A1 族内 H256 dev 最优 a，裁剪至[1,8]）。
- S4: S(a_S,b_S)@S 最优 LR；S5: S(a_S,b_S)+目标块β1=0.5；S6: S(a_S,b_S)+目标块β1=0；S7: S(a_S,b_S)+全模型β1=0.5（均@S 最优 LR）。
- S8: S 最优 LR 在上边界→S(1,1)@0.01；下边界→@0.0003；中间→S(1,1) WD=0.2。
- F4/F5 与 P4/P5：其一为自身 LR 边界扩展（若在边界），另一为 a 单侧邻域（×0.5 或 ×2，按 A1 dev 相邻斜率，限[0.5,8]）；无可用项则少跑。
- B 训练中在锚点(10/25/50/75%)流式记录 3 矩阵机制标量 + SVD（4 试验×4 锚×3 矩阵=48 次封顶）；锚点试验集合在 B 启动前按 A1 结果定并记录（plain-S 最优 LR、F 最优 LR、P 最优 LR、S(a_S,b_S) 最优 LR）。

## 路由（B→C，冻结）

1. F/P 最优对最优 S 的 screen H2048 点收益 ≥1e-4 且无持续数值劣化 → 该族进 C。
2. 收益只在 H64、H2048 消失/反向 → HORIZON_DEPENDENT；C 只含最佳 S+普通 S+≤1 个 F/P 参照。
3. S4-S7 任一 ≥ F/P → SIMPLE_CONTROL_EXPLAINS_WITHIN_TESTED_RANGE；C 含该 S+普通 S+≤1 参照。
4. 两者皆败 → C 最小化（最佳 S+普通 S），随后 D 描述后结束。

## C 与最终判读

配方 ≤4（去重），seeds 3408/3409/3410（已核历史未训练过；如撞历史按升序换首个未用并记录）。LR 从 B 直接转移。每 seed 同初始化同数据序列。final_holdout 一次性开封后不再调参。终标：PRELIMINARY_SEED_ROBUST（3seed 同向、≥2/3 正、中位 ≥1e-3、无一 seed 反向 >1e-3）/MIXED/NO_GAIN/HORIZON_DEPENDENT；文档 95% 与 Bonferroni97.5% 区间 + seed 散布。

## 执行与状态

单写入者持久台账（尝试/完成/恢复分列）；`python -m campaign_v16.run` 按磁盘 QUEUE 串行执行，崩溃/超时后同命令续跑；STATE.json/QUEUE.jsonl/DECISIONS.md/STATUS.md 全落盘。终点成功必须核对 step/游标/指纹/评估完整性，不能只看文件存在。可再生的本轮大文件按 deletions.jsonl 登记后删除；绝不删旧实验/原始数据。终态：COMPLETE_WITH_SIGNAL / COMPLETE_WITH_SIMPLE_EXPLANATION / COMPLETE_WITH_NO_GAIN / MIXED_OR_LIMITED / STOP_RESOURCE / STOP_BUDGET / STOP_IMPLEMENTATION。
