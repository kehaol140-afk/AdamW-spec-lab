# 有限 NS 更新的目标价值：R0＋R1 执行协议 v1.1

日期：2026-09-29。性质：按用户本轮建议形成的前瞻执行任务；尚未运行本轮实验。
定位：问题筛查与机制诊断，不是新优化器发布，不是顶会效果保证。
执行者：OpenCode／Claude／Kimi。中文报告。

## 0. 本次采纳的修改与授权边界

本轮核心问题：改变有限 NS 后，一步目标差异中有多少来自幅度；相同幅度下，更新相对结构是否仍有独立价值。一步信息能否预测持续终点留给后续 R2，本轮不宣称已经回答。

采纳用户的三项修改：
1. R0 必须先统一真实恢复时钟、LR、参数状态、实际更新范数；历史选择证据不足则明示，不倒签。
2. R1 对全局幅度 a 和 O 组相对幅度 b 做一次预先规定的边界扩展，b 包含大于1的方向；两种结构始终获得相同最终候选集。
3. 数值重复误差不等于采样不确定性；R2即使以后获批，也只是同父seed、256步的开发性续训，独立方法结论需要新种子与新评价。

当前允许：必要勘误与测试 → 冻结本轮实际输入 → 执行 R1 校准 → 自动一次边界扩展 → 锁定选择 → 检查侧有限目标评价 → 报告。
R0通过、资产齐全且预算足够后直接执行R1，不再另交一份“等待启动”的文档。

当前禁止：重新训练父模型、完整续训、R2、旧M2／T3-mini重启、旧K3/5/7联合分配矩阵重开、新的operator选择、新优化器、HVP／全程超梯度、修改NS系数／缩放、AMO/OrScale等完整训练对比。

R0不等于重审整个仓库。最多处理会影响本轮因果对照的缺陷。历史并列证据无法恢复时登记UNKNOWN，不阻止采用本协议前瞻规则开展新实验；但不能把旧赢家重写为已正确预注册。

本协议是对references中的“下一轮最小实验草案”的本轮修改，明确取代其a/b网格、最大查询数与预算；不追溯修改任何旧实验。

## 1. 所需材料与文件边界

读以下材料：
- 本协议及 `OPENCODE_R0_R1_TASK_CN.md`；
- `r0_r1_contract.json`，这是实验约束模板，不含服务器实际路径/SHA；
- references中的上一轮理论分析、独立复查和旧P0/P1协议。

这些参考文件不是原始训练结果的替代品。本机交接包不包含服务器checkpoint、训练代码或语料。执行者必须从现有服务器项目取得事实依据。

在现有 `muon_spectral_allocation/` 内以新脚本/模块和 `outputs/r0_r1_v1_1/` 工作；如实际项目名不同，核对后记录路径，不另建通用框架。旧协议、原始表、审计、其他项目及安装环境只读。

先写实际版本清单，包含：代码commit/dirty diff或源码SHA、checkpoint SHA、安装Muon源文件SHA、模型及operator路由、配方、数据清单和当前已存在的输出。发现同版本已部分运行，依据run_id与配置SHA恢复，不覆盖完整结果、不混合不同代码版本。

不为报告美观删除失败或中断记录；旧文档更新用erratum，不改原文件。

## 2. 研究对象与共同一步

### 2.1 状态与模型

复用父seed3101的两个完整恢复点step512、1536，预期MiniMind 30,025,216参数、h512/L8、BPE6400、seq256。数量、路由与配方来自实际产物，不为了符合文本改模型。

两点来自同一父轨迹，不是两个独立种子。O组是所有层attention output projection的集合，预先由旧开发证据提名；不得看新结果换成V/Q或只保留有利层。

本地配方：原经验quintic、original shape scaling、原生BF16 NS、FP32权重/动量；辅助AdamW及WD规则照原配方。NS5为主参考，NS3只作为等组范数结构诊断。

### 2.2 数据与目标

每态8个完整train optimizer batches：前4校准、后4检查。每个候选从共同checkpoint独立恢复，batch之间不持续更新参数或动量。

沿原生已冻结训练流，在step512态从编号至少520、step1536态从至少1544开始确定性扫描。排除旧P0/P1已测probe的packed-window ID以及本轮已经选过的窗口；只接受成员无重叠且完整的batch。取各态前8个合格batch。先明确编号是0-index还是1-index并保存实际ID，不能用梯度或loss挑选数据。若无法核对旧probe成员，不宣称新样本，停止数据选择环节并报告。

完整batch保持原组成、有效batch与loss mask。预期128个256-slot窗口、32640有效targets；实际计数优先，发现不同先核对，不调整标签凑数。全体unique参数共同进行原生global clipping，包括辅助AdamW组；不能只clip O组。

本轮沿用有可核验清单的 `Q_select`、`Q_P1_check`，并标记其历史开发暴露。Q_select只用于校准；本轮所有选择锁定之后才计算Q_P1_check。两个域固定等权：V_Q=0.5*(sum_A/count_A)+0.5*(sum_B/count_B)。两域sum/count、每个batch的配对结果都保存。

这只是选择过程分离的开发性检查；新train batches不使旧评价池变成新final。不得调用 `Q_P2_endpoint` 或原confirm，不从它们挪样本。评价模式、计算精度和固定pool在所有候选中一致。

### 2.3 数学定义

对一次共同、完整且剪裁后的梯度G，按安装源码形成更新后的M与NS输入H。H和M不随本次候选的a/b变化。

对每个Muon矩阵l，用该参数组真实下一步LR：
`u5_l = -eta_l * shape_scale_l * Phi5(H_l)`。
O组另算 `u3_l = -eta_l * shape_scale_l * Phi3(H_l)`。

组范数包含O组所有层并用FP64归约：
`nu5 = sqrt(sum_l_in_O ||u5_l||_F^2)`，`nu3`同理。
`c = nu5 / nu3`，`u3matched_l = c*u3_l`。
这个c每个batch由共同输入计算，不利用任何目标NLL。它是机制诊断所需的昂贵归一化，不是已经可免费部署的方法。

候选(a,b,shape)：
- 所有非O Muon矩阵的数据更新：`a*u5_l`；
- O组、shape=NS5：`a*b*u5_l`；
- O组、shape=NS3_MATCHED：`a*b*c*u3_l`。

所有候选使用相同WD作用、辅助AdamW一步和真实计数器。a/b只缩放Muon数据更新，不缩放WD或辅助AdamW。实现不能简单把param_group.lr乘a而让decoupled decay也随之改变。

等范数是O组整体等范数，不是每层分别等范数；允许组内层间更新分布不同。不要将该比较称为“只旋转奇异向量”。

额外两个参考：
- NO_MUON_DATA_UPDATE：保留共同WD、辅助一步和状态形成，只把Muon数据更新置0；不可作为正式训练候选参与选优。
- PRE_STATE：完整更新前的参数模型。与前一项不同，不混淆。

零范数处理：若nu3与nu5均为0，两结构退化相同，登记degenerate；若nu3=0、nu5>0，不用任意epsilon制造匹配，标记该样本NORM_MATCH_UNRESOLVED。

## 3. R0：先闭合实际时钟与更新

### 3.1 时钟闭环

每个恢复点只读列出：cursor、已完成step、原 `meta.lr_used`、Muon/aux每个param_group实际lr、scheduler配置、下一次原生update应消费的step与LR。

加载state_dict之后，将所有参数组写入原时钟下一次应消费的LR；不硬编码两态LR，不重启warmup。计算候选、原生一步、WD与辅助一步都使用这个统一时钟。

用原生真实step路径作独立参考，不用同一增量函数自证。核对NS5、a=b=1的模型参数、所有optimizer状态、步数、RNG等价性。相同算子顺序应bit一致；若不能实现，不私自扩大容差继续，记录差异及原始路径后停止R0。

### 3.2 实际等范数与隔离

共同参考是原checkpoint完成WD后的Muon权重与完整辅助一步。权重差在FP64中求出并报告实际data delta：`u_realized = theta_candidate - theta_after_common_WD`（Muon坐标）。

对每个相同(a,b)的结构对：
- 计算中u的O组范数相对差≤1e-6；
- 最终FP32参数中实际delta范数相对差≤1e-4；
- 超出时不宣布“等幅结构作用”，不改阈值，标记NORM_MATCH_UNRESOLVED并停止受影响比较。
这两个是本轮新设的工程容差，不是理论收益门槛。即使通过也要保存真实差值，不以匹配标签代替测量。

检查辅助参数和WD在候选间完全一致。检查每个组、矩阵与全模型更新范数；不要误把总theta差（含WD/aux）称为纯Muon更新范数。

每个候选恢复独立buffer，尤其是CPU step标量。对实际消费的梯度/H做前后hash；不得监控另一个未使用副本。执行A→B→A测试，原始checkpoint payload不被修改。

### 3.3 历史问题与新规则

输出 `R0_ERRATUM_AND_IDENTITY.md`：旧raw正负数量、交互项符号、旧选择tie/median历史证据、当前LR差异及修正范围。旧主候选缺项只登记，本轮不追加gate/down补测来重写旧赢家。

新实验的规则以下文为准，先落盘再查看本轮校准结果。历史证据缺失不倒签，新定义也不冒充历史定义。

### 3.4 数值重复与不确定性

在同一个既有校准batch、同一个候选上重复评价，记录e_repeat。它只能表征测量重复性。
`delta_practical_step = 1e-5`；`numeric_resolution_limit = 1e-6`；tie_tol=1e-6。
若e_repeat>1e-6，先停在INSUFFICIENT_PRECISION，不把实际收益阈值事后放大后宣布通过。
即使e_repeat=0，也不能说采样误差为0或CI为0；真实batch和目标文档的不确定性要另报。

R0应先对新测试建立明确失败，再修实现并跑通过。旧结果只读；新输出带新配置和代码SHA。

## 4. R1：初始网格与唯一一次扩展

### 4.1 初始网格

A0={0.5,0.75,1.0}；B0={0.75,1.0,1.25}；S={NS5,NS3_MATCHED}。
每个校准batch18个主参数点，另加两个诊断参考，共20点。
两个状态×4校准batch×20=160个逻辑目标点。

其中b>1表示O相对其他Muon参数增强，而不必表示O相对原生绝对放大（绝对倍数为a*b）。两种结构在全部相同a/b下成对测量。

### 4.2 选择函数与并列

每个state分别计算4个校准batch的平均V_Q，不从单个最有利batch选择配置。
对任一家族，令 m=min(mean_V)；near_best={c:mean_V(c)≤m+1e-6}。
在near_best中选最接近native的配置，排序键依次：
1. abs(log(a))+abs(log(b))；
2. abs(log(b))；
3. abs(log(a))；
4. a的数值；5. b的数值；6.固定candidate_id。
NS5和NS3在各自家族内选择；不通过跨结构tie人为吞掉需报告的比较。
G家族固定b=1,NS5；其tie按相同规则处理。

### 4.3 一次性、校准侧、对称扩展

初始网格全部校准完成后，检查以下家族的near_best集合：
- G：NS5,b=1，只选a；
- A：NS5全a/b；
- S：NS3_MATCHED全a/b。
遍历两个state。只要任一家族near_best触及对应初始边界：
- a=0.5 → 将0.25加入全局A集合；
- a=1.0 → 将1.5加入全局A集合；
- 在A/S家族b=0.75 → 将0.5加入全局B集合；
- 在A/S家族b=1.25 → 将1.5加入全局B集合。
G中的b=1是固定条件，不触发b扩展。

触发条件全部只由初始校准结果计算，并写 `extension_lock.json`。将所有触发取并集，形成共同A_final/B_final；两个状态和两种结构都补齐同一个扩展Cartesian网格。共享已有原生/参数点可去重，但不同state不能共享数值。

只执行这一轮扩展，不继续依据扩展结果加点。最大A=5值、B=5值、结构2种，即每个校准batch最多50个主候选。加两个参考后，校准总上限416逻辑目标点。

这不是完整LR调优：a只改变Muon数据更新。表述为“有限幅度搜索”。

### 4.4 最终边界状态

对G/A/S各自记录：selected_touches_a_boundary、selected_touches_b_boundary、near_best_touches_boundary。
扩展后边界仍触及，标记AMPLITUDE_RANGE_UNRESOLVED。
若有匹配结构差异，可报告“当前幅度下的结构差异”，但不能说已排除充分幅度调优，也不自动进入R2。
不因a=0参考更好而继续搜索到0.125或选择零学习算法；它只提示一步目标的近视局限。

## 5. 锁定与检查：最多9个方案/检查batch

每个state在全部校准完成后锁定：
N: 原生a=1,b=1,NS5。
G: 全局幅度最优(a_G,1,NS5)。
A: 组幅度家族最优(a_A,b_A,NS5)。
S_at_A: (a_A,b_A,NS3_MATCHED)。
S: 结构家族最优(a_S,b_S,NS3_MATCHED)。
A_at_S: (a_S,b_S,NS5)，作为反向匹配结构对照。
G_at_A: (a_A,1,NS5)，直接隔离O组相对倍率。
Z: NO_MUON_DATA_UPDATE诊断。
P: PRE_STATE诊断。

相同方案去重执行但保留所有角色标签，不能用去重造成配置丢失。校准表SHA、扩展清单、a/b值、映射版本、路由、规范化规则、评价清单、数值容差和代码SHA写入selection_lock，再允许Q_P1_check调用。

检查上限：两个state×4batch×9=72逻辑点。
R1最坏上限416+72=488逻辑目标点（另计R0/重复性测试）。若无扩展则160+72=232。
不是“必须凑够”目标数量；缺失、失败、去重与参考复用逐项登记。

检查侧每batch的c=nu5/nu3由当前共同梯度计算，属于事先定义的、无目标标签输入的诊断算法；不能根据检查loss调c或a/b。不在检查侧看全网格重新找赢家。

## 6. 主要比较、报告和本轮决策

收益正方向全部统一为“参照NLL－候选NLL”。逐域与等权总体分栏。

预定比较：
1. 全局改善：V(N)-V(G)。
2. 组增量（公平选参）：V(G)-V(A)。
3. 组纯配对：V(G_at_A)-V(A)，同a只改变b。
4. 匹配结构增量：V(A)-V(S_at_A)，同a/b/组范数。
5. 各自等搜索机会后的结构增量：V(A)-V(S)。
6. 反向匹配结构增量：V(A_at_S)-V(S)。

不要用“S优于N”代替“结构有独立价值”。若匹配结构作用为负而其另选幅度总体为正，明确分开，不统一称为纯结构收益。

每个比较保存4个配对差、均值、中位数、样本SD、正负个数、最小最大值，以及域A/B差值。可以做配对batch bootstrap（10000次，固定seed31010408），每次对所有方案共同重采样4个batch索引；区间明确标为条件于当前state和固定评价pool的探索性区间，n=4不保证覆盖率，不作常规显著性宣称。

若有文档级loss，可另做按线程/翻译关联组的目标bootstrap；不将tokens、矩阵层、候选或两个checkpoint视为独立训练实验。不将数值重复误差替代上述不确定性。

局部投入信号建议定义：平均配对收益≥1e-5且至少3/4同向，同时技术与范数匹配有效；这只是筛查规则，不是统计证明显著、等价或普适性。

逐state报告，不凭平均掩盖阶段反号或域退化。单状态信号只能生成阶段假说。

输出决策：
- R0_BLOCKED / NORM_MATCH_UNRESOLVED / INSUFFICIENT_PRECISION：尚不可作科学阴性。
- AMPLITUDE_RANGE_UNRESOLVED：只能给有限范围结论，停止本轮，不继续扩点。
- GLOBAL_ONLY_LOCAL_SIGNAL：先视为基线校准；停止当前新谱方法立项。
- GROUP_INCREMENT_LOCAL_SIGNAL：有超过全局的局部幅度信息；不等于超过OrScale，需未来低成本信号和持续验证。
- MATCHED_STRUCTURE_LOCAL_SIGNAL：匹配结构且公平调参有检查支持；只能建议审阅后设计R2。
- NO_INCREMENT_IN_TESTED_FAMILY：停止本轮O/NS3等范数候选家族，不否定其他系数族、更多operator或所有目标敏感谱方法。
- MIXED_OR_UNCERTAIN：方向/幅度/范围证据不一致，保留全部数据，不强行分成正负故事。

主效应、控制、边界与证据级别分别输出，不用一个PASS代替它们。

## 7. 成本与实际可部署性

R0允许GPU核查累计最多15 GPU-minutes、CPU数值测试最多30分钟。
R0+R1总计最多2 GPU-hours、新输出最多2GiB。这是本轮根据网格扩展新增的上限，不是时长预测。两张卡分别运行一小时等于两GPU-hours。

计入梯度、两种NS计算、目标前向、重复检查、编译、失败尝试和CPU费用；不额外申请父训练。
缓存同batch的H、u5、u3可复用并核验只读。不需要落盘所有候选checkpoint或全参数梯度bank；保存逐域sum/count、参数点hash、更新范数、c、实际LR和必要固定示例。

若实现能共用NS中间步骤，须验证其K3/K5输出分别与安装路径相符，并按实际调用计费；否则单独调用。不能默认为每步3+5成本，也不能默认3步匹配方案一定更快。

固定a/b的NS5幅度臂与每步要取得NS3、NS5范数的结构臂，计算成本不同。本轮为机制诊断，不宣称等墙钟训练优势。未来要做方法，必须解决昂贵norm-matching的低成本替代或如实计费。

达到预算先落盘，报告PARTIAL_BUDGET_STOP。不得为了凑结论改变目标规模、精度、batch、系数、tol或幅度网格。

## 8. 必须通过的最小测试与实现顺序

不重建平台。建议新增小模块 `r1_probe.py`、`r1_selection.py`、独立 `r1_analyze.py` 和一个执行脚本；已有接口可复用，实际路径落盘。

任务1：R0恢复与身份。
- 先写会暴露lr(s)/lr(s+1)混用、CPU step别名的测试；修正后通过。
- 测a=b=1/NS5恢复native；辅助/WD不变；A→B→A；计数与实际cache不变。
- 实际checkpoint对照通过后保存R0_GATE.json，不靠合成测试替代。

任务2：组级幅度与结构。
- 测sum-square组范数，O所有层覆盖且不重叠；gamma方向不写反。
- 非单位a/b、b>1、a=0、零norm、不同尺寸和一组多层。
- intended与realized delta-norm两个验证，common WD边界明确。

任务3：前瞻选择与边界。
- selection只读取校准结果，不能访问检查目标。
- 测1e-6 near-best、native偏好、相同b>1机会。
- 触边并集扩展只一次，两个state/shape最终网格完全相同。
- 合成边界最优须输出UNRESOLVED；不偷偷增加第三轮。

任务4：执行、重算和成本。
- 保存可续接run_id和config hash，不混用版本。
- 独立analyze从sum/count复算V_Q和配对差，不只调用训练汇总。
- 缺失candidate、重复ID、空目标、负/零count不能静默跳过。
- native重复为数值字段，不能生成采样误差为0的声明。

所有真实探针完成后，参数修改仅发生在临时候选副本，不存在新的持续训练轨迹。R1不产生R2意义上的训练更新序列。

## 9. 交付与恢复

建议交付：
- docs/R0_ERRATUM_AND_IDENTITY.md；R0_GATE.json。
- configs/r1_executed.json：实际来源、输入和本协议规则，冻结SHA。
- manifests：checkpoint、train batch、评价关联组、代码与配方hash。
- calibration.csv / extension_lock.json / selection_lock.json。
- check_values.csv / paired_effects.csv / boundary_status.json。
- norm_match_checks.csv / native_identity.json / cost_ledger.json。
- REPORT_R0_R1_CN.md：按三个主要问题、两阶段、两域汇报。
- tests及真实命令输出、失败台账、源码diff。
- R0_R1_REVIEW_BUNDLE.zip，目标≤10MB，不包含checkpoint/语料/环境。

文件上传失败时先报告服务器真实路径和缺件列表；不得在缺原始配置时用聊天回忆填值。报告与轻量CSV可分开上传，不反复重跑实验以修上传。

## 10. R2以后再决定：本轮只写建议，不执行

如果只有全局信号，先用于修正强基线，不增加新谱方法。
如果O组超过全局，首先考虑与OrScale的差异及可获得的低成本信号；单个固定倍率不足以构成新方法。
如果幅度之外结构仍有增值，再提出开发性R2：最多两个同父seed状态、每条256真实updates；每步重新求梯度，干预持续执行，原时钟不重启，目标为保留的Q_P2_endpoint。它不证明完整预训练或跨seed收益。
R2的策略与预算在R1审阅之后另批。一步的符号/排序先锁定，R2检查兑现与否，不把一步收益乘256。
若结构作用系统性负，要跨系数族/模型/种子解释，才能考虑机制或负结果论文；本轮O/NS3不足以作此定位。

## 11. 文献边界（本轮为方法定位，不要求下载后才可运行）

- OrScale：已研究逐层trust-ratio幅度校准，不能将固定O倍率当作首次层级调幅。
  https://arxiv.org/abs/2605.07815v2
- AMO：operator观测—预算分配—锁定，完整NS系数机制与本地K3/5消融不同。
  https://arxiv.org/html/2605.17806v2
- SAMuon：持出谱目标探测与谱整形，主实验基线为Scion；不等于本地original配方。
  https://arxiv.org/html/2608.25990v1
- DynMuon：阶段相关谱整形已有近邻，不能单凭两checkpoint不同就声称新阶段机制。
  https://arxiv.org/html/2605.17109v1
- Muon with Finite Newton–Schulz：有限NS作用已有理论研究，其条件不自动覆盖本地经验系数。
  https://arxiv.org/abs/2608.26288
- 实际代码参考，最终仍以服务器安装源码SHA为准：
  https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/optim/_muon.py

上述方法本轮均未复现；AMO/Q3保持COMPARATOR_UNRESOLVED。新一轮R1可不依赖网络完成，但不得因来源获取失败用弱自写版本替代它们。
