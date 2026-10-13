# Muon 更新幅度：R1 最小勘误 + R2 三臂开发性续训执行协议 v1

日期：2026-09-29。执行者：OpenCode / Claude / Kimi，中文报告。

## 0. 目标、依据和本轮范围

用户批准按上一轮讨论继续：先修复已定位的恢复缺陷，再直接比较原生、只缩 Muon 主矩阵、主辅数据更新一起缩三种固定策略。先看真实 256 步终点，不开发新优化器，也不重新搜索倍率。

依据：`references/R0_R1_INDEPENDENT_ANALYSIS_CN.md`、原 R0/R1 协议，以及本轮三臂设计。参考材料是已有审计，不代表本轮已执行。本文将上一轮暂未批准的 R2 改为当前有界授权；旧实验的停止决定和原始文件不追溯修改。

本轮回答：
1. 一步上 a=0.5 的 Muon 数据缩幅，是否在 256 步真实续训后仍优于原生？
2. 同样把辅助 AdamW 数据更新也减半，效果相同、更好还是更差？
3. 结果是否依赖恢复点或数据域？

不回答：充分调优后的最优 LR、完整预训练优势、跨种子泛化、物理机制唯一归因、超越 OrScale/AMO/SAMuon/DynMuon。

## 1. 执行顺序

A. R1 最小勘误、局部回归测试与必要重算。
B. 从已有资产绑定 R2 配置、六个 run、数据流与目标清单。
C. 第一步实现一致性核对，然后运行六条 256 步真实续训。
D. 固定一条 native 数值重复（预算允许；不是新科学样本）。
E. 所有训练完成后，统一在预留的 Q_P2_endpoint 上计算 0/1/256 步目标并报告。
F. 交付后停止，不自动扩大实验。

A通过且B所需资产齐全时，直接进入C；不要再只交一份“等待审批”的协议。仅技术正确性、资产、隐私/访问或资源阻塞才停止。旧校准的科学结果不用于临时改变本轮三臂；即使更正后的幅度网格选出别的值，本轮仍然检验事先指定的0.5，不称它为已充分优化的最优值。

## 2. 文件与工程边界

在已有 `muon_spectral_allocation/` 内复用真实训练、数据、恢复与评价模块；实际目录不同则核对记录。不要另建通用平台或修改安装版优化器。

建议新增：
- `docs/R1_REPAIR_FOR_R2.md`；
- `configs/r2_amplitude_executed.json`（实际路径、SHA、状态、评价和预算）；
- `src/r2_data_scale.py`（每个参数组的数据更新缩放语义）；
- `scripts/run_r2_amplitude.py` 与 `scripts/analyze_r2_amplitude.py`；
- `tests/test_r2_data_scale.py`（现有测试框架）；
- `outputs/r0_r1_v1_1_corrected/` 和 `outputs/r2_amplitude_continuation_v1/`。

可用现有同等模块代替这些文件名，但必须给出映射。所有新计算另存；不覆盖旧校准、旧检查表、旧selection_lock和报告。保存实际代码SHA/diff和失败记录。已有同版本运行只能按run_id/config hash恢复，不重复训练已完成run。

本轮没有远程仓库强制要求，不增加数据库、监控服务或繁复权限系统。

## 3. 阶段A：有终点的R1修复

### 3.1 必须修复和验证的数值缺陷

- 完整保存、恢复native一步之后全部unique参数和必要model buffers，含辅助组，不再仅恢复Muon坐标。或让PRE_STATE诊断使用独立副本。
- 所有候选恢复独立模型和优化器状态；CPU `step`、m/v、momentum与原payload不共享可变张量。
- 所有参数组实际写入下一步应消费的LR，native和候选共用时钟。
- 实际锁函数和边界函数调用同一个near-best规则：原协议tie=1e-6，包含最终上下界；不改科学阈值、不新开第三轮网格。
- “实际等范数”从写入后的参数减去共同WD基点计算；intended向量另列。原已有容差保持不变。

回归最少覆盖：N→P→N、P→扩展、候选单独/批量相同、A→B→A、payload不变、恢复时钟、相同输入下native一致。先复现缺陷，再验证修复；不以旧布尔PASS代替测试。

### 3.2 原数据最小勘误

复用原manifest、batch、目标与同一扩展列表，补算受影响的176条扩展校准，记录实际键数。旧初始校准与直接检查值在身份一致且抽验一致时可保留。

重放原文档规则产生 `corrected_selection_lock`，明确是看过旧检查结果后的勘误。候选参数点不变且实际模型点一致的检查结果可复用；新增缺少检查点最多72个，只补锁规则所需点。不得利用检查结果再挑赢家，也不再次搜索a/b。

若服务器已有经相同修复并可核验的更正版，验证并复用，不重复计算。

阶段A上限30 GPU-minutes，计入全轮2 GPU-hours。达到资源上限而修复未闭合时停止，不带病进入续训。若只是历史文件缺失，准确登记；新R2所需状态恢复/更新语义未验证则必须停。无需为了历史文档美观重训父轨迹。

## 4. R2固定矩阵：两个恢复点 × 三臂 × 256步

父seed=3101；恢复点=step512、1536。预期MiniMind 30,025,216参数，其实际身份以原冻结父配置为准，不为匹配文本修改模型。

| 臂 | Muon数据更新倍率 r_M | 辅助AdamW数据更新倍率 r_A | 含义 |
|---|---:|---:|---|
| N | 1 | 1 | 原生续训 |
| MH | 0.5 | 1 | 仅主矩阵数据更新减半 |
| AH | 0.5 | 0.5 | 主辅数据更新都减半 |

六条主分支：每条256次真实optimizer updates，总1536次。
- step512恢复后，消费原时钟513…768；
- step1536恢复后，消费1537…1792。

没有新的父seed，不将两个checkpoint称为两个独立seed。

固定一条额外的step512/N完整重复，最多256 updates，仅检查数值复现；应在六条主分支之后进行，预算不足时写未完成，不减小主分支horizon。主分支加该重复最多1792 updates；修复和一步测试的计算另计台账。

全部保持原NS=5、经验系数、original shape scaling、precision、路由、momentum、Nesterov、aux betas/eps、grad clipping和数据规则。O组不特别处理；不运行NS3、零主更新或新候选。

## 5. 更新语义：缩放数据项，不缩放梯度或WD

对参数组p，令原时钟在真实全局步s+k给出的基准LR为 eta[p,k]，原WD为lambda[p]，原生优化器由当步实际剪裁梯度和历史状态得到数据方向 d[p,k]。

本轮必须执行：

    theta_next = (1 - eta * lambda) * theta - r[p] * eta * d

其中r[p]按N/MH/AH取值。所有组的WD乘积 eta*lambda 不因r变化。

关键约束：
1. 每步都根据该臂当前theta重新forward/backward，按原时序累积，最后全体unique参数共同clip。
2. 不能先把`.grad`乘0.5；那会改变momentum与AdamW二阶矩，不是本轮干预。
3. 不能把完整参数delta（含WD）整体乘0.5。
4. 每次更新momentum、m/v、真实step、scheduler/data cursor一次；不reset状态。
5. 同一梯度/同一旧状态的一步核查中，三臂的原生moment状态应相同；真实续训分叉后梯度和moment不同是正常反馈，不强行同步。
6. 原生grad=None、skip-step等语义不改，不给无梯度参数新增decay。
7. tied embedding/head只归属于一个参数对象和一个辅助组。

### 5.1 可采用的最小实现

若安装版两优化器均按该组LR执行解耦WD，可在每步写：

    effective_lr = r * base_lr
    effective_weight_decay = base_weight_decay / r

对本轮r∈{1,0.5}没有除零。这样 effective_lr*effective_weight_decay 保持原乘积。base配置独立保存，每步从原时钟计算，不能用上一步已缩放值再乘导致0.5重复累积。

这个代数等价不代替实际测试。使用安装优化器原生step，并在小张量与两个真实首步上检验。若原实现的LR/WD或fused路径不满足假设，报告并改用经过同等验证的数据项分离适配；不能静默修改moment或参数路由。

注意：此实现中配置表上的WD数值可能变为0.04，但相同基准LR下的实际衰减乘积仍为原0.02配方。报告必须同时保存base_lr、effective_lr、base_wd、effective_wd、decay_product，避免误称“加倍正则”。

### 5.2 实际首步验证

每个恢复点使用同一个真实完整batch，分别建立独立原生参考和三臂。
- N与原生产训练器同一步的模型与optimizer状态，应在相同运算路径下bit一致；不满足则停止检查原因。
- MH与AH相对共同WD基点的数据delta，按Muon/aux组分别检验目标倍率（预期1或0.5）。FP64归约；非退化组相对误差≤1e-4，绝对参数误差≤5e-8+1e-6*|reference|作为辅助检查。
- 同输入下moment buffers/Adam m/v/step相同，payload保持不变；每组的decay product相同。
- 每臂当前状态与scratch参考不共享张量；保存actual start/end hashes及错误最大值。

这些是本轮工程容差，不是效应显著性阈值。不得为了通过临时放大。首步测试只验证实现，不据其NLL决定是否运行某臂。

## 6. 数据、原时钟和配对

从父checkpoint真实cursor继续原数据序列，构建未来256个完整optimizer batches。确认父模型初始化seed与train manifest seed不同的语义；不得直接用3101代替已冻结manifest seed。

每个恢复点内三臂及native重复使用完全相同的tokenized/packed batch成员、顺序、micro-batch顺序及起始RNG。保存一个共享manifest及SHA。它们不共享训练过程中计算的梯度。

沿原2048总步数的warmup/cosine，逐真实步计算所有参数组base LR。不得重启、压缩、改成常量尾部或拿峰值/最终值替代当前LR。保留原辅助/主组LR比例；实际两组相等也要从来源核对。

不强行过滤掉曾用于诊断的训练windows以制造“新数据”；这是指定状态之后的原生续训，历史暴露如实记录。训练成员不得进入评价池。

预期每完整update：128窗口×255有效targets=32640，input slots=32768。
- 每条256步：8,355,840有效targets；
- 六条主分支：50,135,040；
- 主分支加一条256步重复：58,490,880。
这些是运行量计数，不是独立或不重复语料量。实际mask/count必须核验，不能改标签凑数。

## 7. 评价：本轮开始使用Q_P2_endpoint，但confirm不动

使用已预留、有真实manifest的 `Q_P2_endpoint`。固定两域等权：

    V_Q = 0.5*(sum_A/count_A) + 0.5*(sum_B/count_B)

数据关联组、pool SHA、token计数、历史访问记录在训练前冻结。若P2_endpoint已被历史访问，则真实记录并降为开发评价；不重新命名成独立final，不为获得更好结果更换pool。若资源缺失，不临时挪用confirm，停止目标评价并报告。

### 7.1 训练中和训练后的分离

训练中仅在64、128、256相对步上按原 `Q_select` 记录诊断曲线；评测必须恢复train模式与所有会影响后续训练的RNG/buffer。monitor数值不能触发调参、早停或更换臂。

每个run保存：
- step1：模型state（含buffers）快照；
- step256：完整模型/optimizer/RNG/cursor/scheduler状态。

全部六条训练及预定重复完成后再统一读取 `Q_P2_endpoint`，评价每恢复点初始、各臂step1与step256。两个初始点可共享；无重复时总计14个主要目标点。native重复如完成，再评价其终点。

由此，同一个目标集可直接比较新首步信号与256步结果，不把旧Q_P1_check的绝对NLL和新目标相减。Q_P2_endpoint的内容或梯度不得用于生成更新、选择参数或选中间checkpoint。

原confirm全程禁止访问。step256为唯一主要终点，不以中间最佳点替代。无“根据收益良好继续跑”的追加窗口。

## 8. 输出量、统计和解释

每个恢复点、每个域、每个时刻先列原始sum/count/NLL和相对于共同初始点的下降量。

预定主要差值（参照减候选，正值有利）：

    G_MH(t) = V_N(t) - V_MH(t)
    G_AH(t) = V_N(t) - V_AH(t)
    G_ratio(t) = V_AH(t) - V_MH(t)

报告t=1与256，分别解读，不将首步收益乘256；不将两个checkpoint均值当成跨seed推断。

建议描述性投入参考线：终点收益1e-3 nats/有效target。它是本轮先固定的资源讨论参考，不是普适成功门槛、显著性界或自动继续规则。精确值和正负方向必须完整报告，不能仅输出过门/不过门。

如果可以取得目标文档/线程/翻译关联组的loss，允许配对group bootstrap（10000次，seed=3101256）：同一次重采样在所有臂和时刻共用权重，逐域sum/count后按q=0.5合成。区间仅表征固定模型下的评价抽样不确定性，不表征训练seed变异。缺少关联组映射则不编造CI，不按token当独立样本。

同目标、同模型重复评价误差只能说明数值重复性。native重复也不是一个新随机训练seed。

逐state讨论：
- MH>N且MH>AH：只缩主矩阵在该设置有相对优势，提出主辅关系假说；三个臂不足以估计完整二维交互，更不证明唯一物理机制。
- MH和AH都>N：一般数据更新缩幅可能足够；不因差异不显著就宣布两者等价。
- AH>N而MH无益：没有证据优先采用“只缩Muon”。
- 首步改善、256步消失/反号：首步目标不足以预测当前有限预算结果。
- 两状态反号或域间权衡：完整报告，不只选好的状态。

以上结论全部是开发性续训，不代表充分调优基线、全训练收益或论文新颖性。Q3/外部方法比较仍未做。

## 9. 最小验证、成本和运行纪律

只补本轮必要测试，不继续完整平台审计：
1. 全参数/状态独立恢复，N→P→N等历史回归；
2. r=1原生等价，0.5数据缩放与decay-product保持；
3. 同梯度下三臂moment一致；实际续训不共用冻结梯度；
4. 连续运行与save/load续跑小例子相同，scheduler恢复不会重复缩放；
5. Q_eval不扰动训练RNG和model模式；
6. sum/count、符号、每域计数及缺失/重复run识别。

每恢复点内优先在同一GPU顺序跑三个arm。若并行两恢复点，分别绑定GPU并记录环境/算子模式；不能将一组臂随意跨不同backend比较。初始完整梯度可用于实现测试，但正式每步都重新计算。

预算：全轮最多2 GPU-hours，CPU数值测试最多30分钟，新输出最多4GiB；阶段A包含在总额中。双卡各1小时即2 GPU-hours。包括加载、编译、F/B、谱计算、评价、重复、失败、修复探针，不只算训练loop。

监控达到预算先保存完整可恢复状态，写PARTIAL_BUDGET_STOP。不得压缩horizon、换精度、减目标池或改batch来凑完成。loss/grad NaN/Inf立即停该run并记录；同配置技术重试最多一次，确定性发散不可通过调整参数救场，其他臂可在预算内完成。

损失不好不是技术失败，不得提前删掉或停止一个正常run。禁止未经授权kill他人任务。

## 10. 交付与停止

建议输出目录：`outputs/r2_amplitude_continuation_v1/`。

审阅材料应至少含：
- `R1_REPAIR_FOR_R2.md`及修正前后影响清单；
- `r2_executed.json`、源码SHA/diff、checkpoint与manifest清单；
- `preflight_checks.json`、所有真实测试命令及输出；
- `run_index.csv`：六主run和重复run的明确身份、完成状态；
- `training_metrics.csv`：真实step、base/effective LR与WD、targets、preclip norm、训练loss和耗时；
- `monitor.csv`：只能Q_select；
- `endpoint_lock.json`：在第一次P2_endpoint访问前锁定run/快照/目标SHA；
- `endpoint_values.csv`：step0/1/256、每域sum/count；
- `paired_effects.csv`、可用时的document-cluster不确定性；
- `cost_ledger.json`，包括失败和重复；
- `REPORT_R2_CN.md`：结果和未完成项分开，不提出新方法；
- `R2_REVIEW_BUNDLE.zip`（目标≤15MiB）。

完整checkpoint及快照留服务器，不放轻量审阅包。不能删除原始文件来修上传；上传失败先返回中文报告、配置、endpoint_values和paired_effects。

完成后停止。结果由用户与审阅者决定下一步，不自动增加倍率、窗口、种子、阶段、论文对照或优化器。
