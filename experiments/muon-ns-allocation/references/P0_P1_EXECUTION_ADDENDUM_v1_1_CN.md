# Muon 问题立项 P0/P1 执行补充协议 v1.1

日期：2026-09-29。性质：交给 OpenCode 的有界实验任务，不是已完成结果。
依据：用户已阅读的《MUON_PROBLEM_FIRST_PROJECT_BRIEF_v1_CN.md》、三轮审计，以及本轮用户关于预算、幅度、评价隔离、组级单位和 AMO 配方的补充意见。

## 0. 本轮目标和授权范围

**效果优先。直接完成 P0 与修订后 P1，不以“又生成了协议”代替执行。**

本轮只回答局部问题：
1. Q1：原系数 K=3/5/7 的有限候选中，是否有同谱计算预算、检查目标上可分辨的一步收益？
2. Q2：该收益能否由逐 operator 的标量增益解释？几何准则选错时，目标损失是否有实际量级？
3. Q3：与可验证的本地 AMO 机制对照相比，是否还有局部收益；比较是否真正具有同配方、同候选和同成本的归因条件？

**本轮不能回答**充分调优后的持续终点优势、跨种子效果、超过完整 AMO 论文方法或部署效率。Q3允许并要求报告 COMPARATOR_UNRESOLVED，而不是用弱几何规则冒充 AMO。

本轮执行：必要修复 → 实现/配方对照 → 冻结输入与配置 → 一条原生父轨迹 → P1a组级探针 → P1b同预算联合探针 → P1c本地AMO机制比较 → 锁定候选 → P1检查 → 中文报告。

P2和P3不自动运行。旧M2、T2b/T3-mini保持关闭，不追加旧分支训练。

## 1. 文件和工程边界

在现有实验根目录旁使用独立 `muon_spectral_allocation/`；已存在则核对并续接，不重复创建项目。复用现有model/data/full_state/evaluation，不修改旧冻结项目、安装环境、旧原始结果或其他人未提交文件。

将用户提供的四个markdown只读存入 references/：
- MUON_PROBLEM_FIRST_PROJECT_BRIEF_v1_CN.md
- ROUND1_AUDIT_REPORT_CN.md
- ROUND2_AUDIT_AND_RESEARCH_ASSESSMENT_CN.md
- ROUND3_AUDIT_AND_CAUSAL_ANALYSIS_CN.md

原立项稿不覆盖。本补充协议优先修订其中：单矩阵/组混用、全局幅度对照不足、P1仅改单类、P2复用检查集，以及“忠实AMO”称呼。

新代码只需小模块：recipe_registry、state_restore、group_probe、cost_meter、joint_selection、amo_local_adapter、report。先写失败测试再实现对应功能；无需数据库、远程服务或大控制器。使用现有测试工具，不安装不必要依赖。

## 2. Q1–Q3的证据等级

每题分别输出 status、比较对象、数据/状态、成本口径、效应及限制：
- LOCAL_SUPPORTED：仅所测一步/当前配方和检查数据支持。
- NO_LOCAL_SUPPORT：所测家族未达到本轮局部筛查要求。
- SCALAR_EXPLANATION_NOT_RULED_OUT：没有排除便宜的组级幅度解释。
- COST_UNRESOLVED / COMPARATOR_UNRESOLVED：公平性或来源未落实。
- INSUFFICIENT_PRECISION / PARTIAL_BUDGET_STOP：不能把无法分辨/未完成写成阴性。

若固定原系数 K=3/5/7 没有足够检查目标收益，停止这个有限候选家族；不否定所有目标敏感谱分配。单步正结果也不自动触发P2。

## 3. 配方与数据冻结

### 3.1 原生配方

沿用MiniMind 30,025,216参数、h512/L8/GQA-4、BPE6400、seq256、FP32参数与momentum、BF16训练和原生NS路径。实际数量和路由应由既有源码核对，不为符合文档改模型。

使用历史稳定且完整的 `m1_muon_lr2.0_wd0.02_p0.5_s2030` 配方作为诊断父配方：主Muon峰值LR 0.01、WD0.02、momentum0.95、Nesterov、original shape scaling、原经验quintic和NS=5。辅助AdamW的LR/betas/eps/decay mask、warmup/cosine时序从该run真实配置读取并逐字段记录，不凭聊天默认值填写。它不是充分调优的最优基线。

新父seed=3101。若已有同名本轮配置和相同SHA的部分运行，严格恢复；若3101已用于其他实验，先登记碰撞并在看新结果前确定首个未使用整数seed，不按结果选seed。

只训练一条原生父轨迹，计划2048 optimizer updates，保存512、1536、2048完整状态。两恢复点来自同一父轨迹，不是两个独立seed。

每步128个长度256的窗口，按已核对的shift/mask应为32,640有效targets、32,768 input slots；2048步分别为66,846,720 targets、67,108,864 slots。必须由计数器验证；不是为了获得整数而修改标签或mask。

每次探针使用恢复点后下一次更新按原时钟应消费的LR；不重启warmup、不套用终态LR、不使用峰值替代实际值。

### 3.2 评价集

开跑前按完整文档、线程和跨语言翻译关联组，预分配：
- Q_select：P1选择和校准，默认每域128个固定packed windows；
- Q_P1_check：P1只读检查，默认每域128个windows；
- Q_P2_endpoint：未来P2终点评价，默认每域256个windows，本轮不读内容算loss；
- 原confirm：继续封存，不用于补足上述集合。

分组先于window抽取；同一个关联组不能跨集合。使用域均值NLL的等权平均q=0.5，每域sum/count都保存。数据不足时停止评价环节并报告，不临时缩小、重叠或挪用confirm。

集合可以从有可靠partition依据的既有dev/probe资源分配，但必须标注历史暴露：旧dev曾用于M1/M2不等于新final。Q_P2_endpoint至少不参与本轮P1；P2仍是开发性验证，独立方法结论需要P3新seed和新评价。

每个状态取8个完整train batches，固定为4校准+4检查，均恢复共同状态独立求梯度。前4只访问Q_select；后4只在全部选择锁定后访问Q_P1_check。检查结果不得反过来决定新候选、系数、幅度、组、预算或阈值。

## 4. 统一组级数学单位

operator组 g∈{q,k,v,o,gate,up,down}，组内包含所有层该类矩阵L_g。主干56矩阵、tied embedding一次更新与辅助组无重叠须核对。

对每个共同状态/训练batch，先按生产顺序累积所有unique参数梯度、统一global clip，再形成G/M/H和原生完整一步 θ_0^+。所有候选共享同一梯度、相同起点、相同辅助AdamW更新及decoupled WD。

组级有限收益：
B_g(K;Q)=V_Q(θ_0^+)−V_Q(θ_{g,K}^+)。

共同候选向量k=(k_q,…,k_down)的收益：
B(k;Q)=V_Q(θ_0^+)−V_Q(θ_k^+)。

B(k)必须实测，不能等于逐组B_g之和。记录交互余项 I(k)=B(k)−Σ_changed g B_g(k_g)。

几何量：矩阵输出在乘shape scaling前，统一转为r_l×s_l，r_l=min(m,n)。
E_l(K)=||Y_lY_l^T−I||F/sqrt(r_l)。
主组级汇总 E_g²(K)=(1/|L_g|)Σ_l E_l²(K)，另报median与范围；不得把未归一化残差混加成目标权重。

## 5. 逐类幅度解释的三联探针

令u_l^(K)为候选的数据梯度更新分量，已含实际LR与shape scaling，但不含WD。定义组级幅度：
ν_g(K)=sqrt(Σ_l∈L_g ||u_l^(K)||F²)，γ_g(K)=ν_g(K)/ν_g(5)。

对每个g、K∈{3,7}评价三种参数点：
1. raw：使用u_g^(K)；
2. gain_only_live：仍使用原生NS5方向，但乘γ_g(K)，匹配raw的组级更新范数；
3. direction_only：使用u_g^(K)/γ_g(K)，匹配原生组级更新范数。

WD与辅助组均保持原生。不能对包括WD在内的总位移作norm matching。ν_g(5)=0或γ=0等退化情形单独标记，不用任意大倍率或静默clip修补。

gain_only_live依赖候选输出范数，是诊断对照，不是免费低成本算法。对最终选中联合候选，另用校准batch的median γ_g冻结常量gamma_const；在检查时无需再算额外K就能实施，作为便宜的逐组标量对照。

这只排除组级单标量解释，不能宣称排除所有逐层/逐矩阵倍率。P2必须正式纳入原生NS5+冻结逐组标量增益基线。

另保留原生Muon数据更新整体乘0.5或2的两个幅度探针，辅助和WD不动。它们不是完整LR sweep，也不能声明LR最优已被括住。

## 6. P0：成本测量与必要正确性修复

### 6.1 只修会污染新实验的问题

- 所有参数参考拷贝在load checkpoint后建立。
- candidate间m/v/momentum/step不得共享可变张量；检查data_ptr/身份及执行后原payload内容，尤其AdamW的step。
- 实际被使用的梯度缓存做前后hash，不监控无关g0。
- no-op K5候选须与生产完整一步一致；额外用独立forward/backward验证一次，不能仅两次注入相同缓存。
- 候选梯度保持同一数据路径/global clip；源H被原地污染必须被测试捕捉。
- 几何统计与目标sum/count的单位、形状、分母测试齐全。
- P1检查模块不参与选择；不得读Q_P2_endpoint/confirm计算目标。

若候选用θ_0^+叠加delta构造，需与从θ_*按实际WD和update原生算术直接构造对照，记录额外舍入。不能让拼装顺序误差伪装成收益。

### 6.2 成本定义

C_g(K)包括组内所有层，不以组数/NS次数代替成本。对r≤s的一步标准NS矩阵乘，主要FLOPs为4r²s+2r³；以实际算子路径为准，另记归一化、点乘和搬运成本。

隔离benchmark各g/K以及最终完整联合更新；预热、同步，随机交替测reference/candidate至少20个计时block。记录median、分位区间、GPU占用、编译时间。主计时不含目标评估和SVD诊断；额外诊断成本单独列账，不能宣称免费。

同预算主比较要求：
- 解析主要NS FLOPs相对reference在[0.98,1.02]；
- 完整谱更新实测时间中位数相对reference在[0.98,1.02]，测量不稳则COST_UNRESOLVED；
- 同时给两比值原值。2%是本轮匹配容差，不是严格等价定理；略超1的方案只称“2%内匹配”。

不让快方案空转、不靠sleep补齐“同时间”。显著更便宜的候选可另报Pareto结果，但不混入主同预算比较。成本匹配只按校准输入/无目标计时确定，在看检查目标之前锁定。

记录谱更新占完整训练时间f；优化器加速不能直接称为整训练加速。没有功率测量不声称焦耳收益。

## 7. P1a：组级敏感性和幅度诊断

每个状态、每个输入，KJ固定系数族共：原生1 + 7组×2K×3探针42 + 全局幅度2 =45个候选点。两状态八输入最多720个一步候选；每个输入只评价所属目标集，不同时访问select/check。

为节省计算，相同H的K3/K5/K7输出复用并只保留必要缓存，几何诊断不反复求SVD。不能把720个点当作720个独立训练重复。

必须先完成全部校准与后续联合/AMO选择，再开放检查侧。检查侧可以完整展示已预定义的组级探针，但不得事后选组进入新主比较。

## 8. P1b：真正的同预算联合分配

### 8.1 候选池

在CPU枚举有序组对(g_minus,g_plus)，g_minus≠g_plus：一组NS5→3，另一组NS5→7，其他保持5。最多42个定义，枚举不等于42条训练。

用第6节成本条件筛成固定池P。主对照原生全5始终保留。若无足够可行组对，报告NO_EQUAL_COST_PAIR，不擅自改成单矩阵、随机舍入或增加新K。

同形状/同算术的q/o、k/v、gate/up/down组对可能名义FLOPs相等，但必须以实际routing和测时确认，不能凭名字认定。

### 8.2 只用校准数据提案和选择

对每个状态：
- 目标提案：用ΣB_g的校准均值在P中挑前2个，随后在全部4个校准输入上真实联合前向评价；用联合Q_select均值选定一个target_primary，另一个仅secondary。
- 几何对照：用校准E_g²总改善在同池选一个geometry_primary，不使用Q来改它。
- 同成本固定对照：用事前登记seed从P均匀抽一个fixed_random_pair；不得按Q挑弱对照。
- 目标主候选加gain_only_live、direction_only、gamma_const三个联合对照。

最多四个独特raw联合方案；相同方案去重但保留来源标签。主目标候选若不优于native的校准值，可选择no-op，并如实登记。

并列规则：Q_select差≤预设数值容差时先取原生，再取较低测量成本，再按(q,k,v,o,gate,up,down)字典序。禁止看到检查结果后挑secondary为新primary。

### 8.3 检查

对后4个独立train batch从相同anchor恢复，所有联合方案重新形成实际G/M/H并有限前向；使用Q_P1_check。

主问题是B(target_primary)、与geometry_primary的目标差、与gamma_const和live matching的差、以及成本，而不是逐组收益预测之和。可展示全部候选，但晋级只看预先锁定的primary。

## 9. P1c：AMO机制对照与系数混杂

### 9.1 名称必须准确

AMO论文方法同时涉及算子类型观测、几何信号、PE系数生成、预算分配及过渡后锁定；论文使用的shape scaling和本地不同。

本轮名称一律：
“本地original配方下的AMO机制一步对照”。

它不是完整AMO训练复现。没有执行观察-过渡-持续锁定全程，就不能把一步结果称为AMO终点结果。

### 9.2 来源、本地化和缺项

开跑前读AMO v2 Alg.1、Alg.2、Appendix C和实验配置；固定可核对代码/PE composer来源与hash，验证标量模拟和系数生成。不能拿论文某个大模型的已打印系数当MiniMind本地观测结果，也不能复制KJ系数然后称AMO。

论文基线uniform PE，原生本地父轨迹为KJ，这是明确的适配差异；observation来自该本地轨迹。建议在每个锚点之前及该步的4个预定位置(s−96,s−64,s−32,s)记录几何，每类固定观测第2与第7层（0-based 1、6），不更新模型、保护RNG。层名不符时按routing确定对应层并在开跑前登记。

其余缺失实现参数必须由论文/作者实现定位并在看目标结果前冻结。若来源或关键公式无法可靠落实，其他P1结果仍可完成；AMO/Q3记COMPARATOR_UNRESOLVED，不用无依据补全。

### 9.3 两个AMO标签，不能混用

A. AMO_local_paperbudget_probe：保留论文计数预算与系数生成/分配机制，输出本地观测得到的k和系数；给出真实FLOPs与时间。sum(K_g)=35不自动等于native成本。

B. AMO_local_costmatched_restricted：将同一AMO投影误差准则限制到第8节同预算池P（加全5），在池内选几何最优；系数仍由同一PE composer和同一观测ell为每个(g,K)生成。此处“有限池+实际成本约束”是本轮改动，不是论文原预算求解器。

A若成本不匹配，仅作描述；B才参加同预算Q3主比较。不能隐瞒这层区别。

### 9.4 目标准则增量需要同系数族检查

不能只比较“KJ目标分配 vs PE-AMO”就把差异归为目标信息。

为Q3另设共享PE系数表的有限比较：
1. 固定本地观测ell以及各(g,K)的PE系数，geometry与target使用完全相同的系数表、normalization和original外层配方；
2. 对4个校准输入测PE族14个单组变更的有限收益，参照组为同表的全5；
3. 在同一成本池按ΣB_g提名前2个，联合前向后只锁定一个target_PE_primary；
4. 锁定AMO_local_costmatched_restricted、target_PE_primary及其校准常量group-gain对照；
5. 在后4个检查输入同目标、同成本比较。

PE池的成本单独benchmark，不能假定等于KJ。几何与目标的池必须相同；池不足或成本不匹配就报告限制，不扩K、不改scaling、不换数据。

没有这条同系数族比较，最多报告不同完整配方的局部结果，不宣称“目标信息超越几何信息”。

## 10. 预注册锁定、统计与决策

### 10.1 两次锁定

第一次在新父训练/目标探针前写P0_P1_PROTOCOL与config，冻结候选生成算法、数据、状态、单位、预算及以下门槛。
第二次在全部校准/成本测量后、任何Q_P1_check计算前写selection_lock，包含两个状态的primary/controls、所有系数表、gamma_const、可行池、成本和hash。

不因某个状态的检查结果改变另一个状态的选择或是否运行其既定比较。正常技术失败只允许同配置恢复，所有耗费计入上限。

### 10.2 本轮局部筛查线

设e_repeat为同一native一步参数点重复评价的最大差：
δ_step=max(1e−5,10 e_repeat) nats/有效target。

这是新设的局部投入筛查线，不是理论误差界或通用实用性标准；不能把δ_step乘训练步数预言终点收益。重复差为0也不证明无限精度。

每个主要比较报告：四个检查train batch的配对值、均值/中位数、符号、逐域结果和δ_step。两个状态分开，外加等权汇总；不把两状态或56矩阵当独立seed。

可做以文档/翻译关联组为单位的配对bootstrap，重算sum/count并在所有候选间共享重采样权重；train batch若重采样也保持配对，并分别报告。CI条件于当前父轨迹和小bank，不能称为训练总体CI。4个batch不足以保证显著性/功效；本轮不设伪显著性PASS。

### 10.3 问题答案

Q1局部支持：预锁定target_primary相对native的检查均值>δ_step，至少3/4同向，两个状态都满足，并满足成本匹配。只单组raw改善不算分配通过。

Q2：必须展示target vs group gain和geometry的直接配对差。若常量group gain即可达到同样好/更好结果，记SCALAR_EXPLANATION_NOT_RULED_OUT；direction_only和live matching用于解释，不能声称已排除全面LR/逐矩阵调参。

Q3：需要同PE表、同可行池、同成本的target_PE_primary vs AMO_local_costmatched_restricted。只有两个状态的配对均值>δ_step且各≥3/4同向，才写“局部存在额外目标信息”。否则记录当前比较无支持或不充分。不能推到完整AMO、充分调优或持续终点。

正结果只允许准备P2；负结果限定到本家族/本场景。缺来源/成本/精度/预算是UNRESOLVED，不是失败算法的科学证据。

## 11. P2先预留，不执行

Q_P2_endpoint不在本轮计算。未来P2至少包含：
- 原生统一NS5并给予有限、对称、事前规定的LR搜索；
- 原生NS5+冻结逐operator常量标量增益；
- 本地AMO机制对照（与论文配方差异逐项公开）；
- 已锁定目标分配；
- 必要的同成本固定/几何对照。

若全LR范围仍在边界，诚实标记未括住；两个全局幅度探针不是解决方案。完整AMO论文复现需独立配方与相应基线，不把它加入本轮并随意换本地scaling。

P2使用预留终点集，每步按当前参数重算真实梯度；它仍是开发验证。P3才用新训练种子、新评价集及适用的外部强基线确认。

## 12. 资源、阶段上限与输出

本轮新GPU消耗合计最多3 GPU-hours；两卡各1h计2h，包含父训练、谱基准、诊断、评价、失败与重试。输出≤4GiB，不复制全梯度bank或每个候选完整checkpoint。

一条父轨迹；16个唯一诊断train batch；大量候选是同梯度的一步参数点，不是新增训练轨迹。开跑前根据已测吞吐和前向数量估算是否容纳得下。不能把旧272候选的成本照抄到新增对照后。

分阶段记账：P0配方/测试/计时；父训练与观测；KJ P1；联合；AMO/PE；检查。预算紧时先完成已锁定的核心KJ/幅度/联合比较；AMO未完成明确Q3未回答，不降低原门槛、不自动扩时。不存在“结果不好就换family继续救场”。

交付最少包含：
- docs/P0_P1_PROTOCOL_v1_1.md、RECIPE_AND_AMO_COMPARISON.md；
- group_probe_table.csv：组、K、probe、状态、batch、数据目标、E_g、实际收益、幅度与成本；
- joint_allocation_table.csv：完整K向量、联合收益、逐组和、交互项、匹配状态；
- amplitude_controls.csv：动态/常量gamma、direction_only与直接配对差；
- amo_comparison.csv：准确recipe标签、同族/跨族、成本、实际目标差；
- selection_lock.json、data/state/source manifests、cost_ledger.json；
- P0_P1_REPORT_CN.md：逐项回答Q1–Q3，局部与未回答的长期问题分栏；
- 测试、关键源码、固定小复算样例与可运行命令组成的review zip。

只画三类必要图：几何改善—目标收益散点、联合候选成本—目标图、primary/controls配对收益图。原生参数、完整语料和环境不回传。

执行结束停止，用户审查后才决定P2。

## 13. 原始来源与引用边界

- 用户四份材料为本地历史证据与原计划；本文件新增预算匹配、三联幅度对照、评价划分和局部门槛，不能写成历史预注册。
- AMO v2（2026-09-27）：https://arxiv.org/html/2605.17806v2 ，Alg.1/2、Appendix C、§4.1/F.1。其预算单位、PE composer、original与论文scaling差异必须如实记录。
- SAMuon v1：https://arxiv.org/html/2608.25990v1 ，§6/G.1；主消融基线Scion，不与本地原生Mu​​on直接等同。
- PyTorch v2.11.0：https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/optim/_muon.py ，服务器实际source hash优先，差异不静默合并。

以上文献用于定义对照和来源，不把作者的训练收益搬成本项目的预期效果。
