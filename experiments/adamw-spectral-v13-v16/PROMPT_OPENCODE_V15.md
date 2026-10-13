# OpenCode 执行任务 V15：当前方向局部优势的时域检验

## 0. 任务与授权范围

工作根目录：`E:\大模型预训练\adamw_spectral_pilot`。
新子目录：`direction_horizon_v15/`。
建议协议ID：`ADAMW_DIRECTION_HORIZON_V15_20261002`。
本轮由OpenCode负责必要代码实现、测试、真实运行、复算和封包，不再只写计划或等待Codex运行。本文件被用户交给你执行时，即批准下述有限实验；没有批准无限加预算或强行得到正结果。

研究主问题仍是“AdamW 预条件更新的谱分配：可利用余地、可预测条件与低成本修正”。V14对当前rank-1谱候选未确认独立收益，而两个无SVD当前方向控制在固定系数下有局部收益。本轮只检验这项非谱竞争解释是否具有持续价值，不宣布新优化器，不扩大谱候选。

先读本地V14_FINDINGS、findings.json、config、screen.py、开发/确认原始数据、V13验收及本次REVIEW。报告和ChatGPT分析都不是标准答案，按实际依赖与数字核实。已有V13/V14正常，不重跑整套历史审计。可以最小修正本轮实现问题，并保存补丁及回归测试；不要修改父树。

## 1. 保护、预检与版本

V1/V12/V13/V14源码、manifest、gate、数据、checkpoint、冻结更新均只读。所有新结果、日志、状态、代码均在新子目录内；父gate保持STOP。本轮是独立授权的新实验，不删除旧门禁解锁旧流程。

使用实际绑定解释器；历史绑定为 `D:\pytorch\anaconda\envs\py39\python.exe`，先核实而不是创建新环境。禁止自动升级torch/transformers、下载语料、换tokenizer/模型/训练精度、启用未授权多GPU、关闭用户进程。

主checkpoint固定：`artifacts/checkpoints/seed3407_lr0.003/step256.pt`。读取完整model/optimizer/scheduler或可复算的调度配置/RNG/数据游标/参数组，核验实际step、父身份和目标参数。
辅助起点固定同一轨迹step128。预检确认其完整状态是否存在，写入manifest后不能看结果再选择时点；存在则按后文执行，不存在则主实验照常、时间起点复核NOT_RUN，不自动重训或更换时点。

科学设置：沿用同一4层MiniMind、tokenizer、训练缓存、sequence256、microbatch2/accum4、有效batch8及原AdamW参数/裁剪规则。仅修改下述数据更新方向和幅度。每次从checkpoint自己的真实后续数据游标开始，不用V13任意抽取window代替连续训练。不同分支从完全相同状态和游标开始；训练后的状态自然分化。

## 2. 正式处理定义（不要用名字猜测）

每步先在该分支自己的当前参数上正常前向、按有效token求mean、反向、unscale（如有）与原全局gradient clipping，然后按父beta/epsilon计算该分支自己的m_plus、v_plus和正确的per-parameter bias correction。

对主目标 `model.layers.1.mlp.down_proj.weight`，定义：

    A_t = mhat_plus / (sqrt(vhat_plus) + eps)
    N_A(X) = ||A_t||_F * X / ||X||_F
    S: D_t = A_t
    F: D_t = N_A(g_clip_t)
    P: D_t = N_A(g_clip_t / (sqrt(vhat_plus) + eps))

N_A使用稳定累加测范数，cast与原路径一致。零范数与V14一致显式回退并记录；NaN/Inf停止，不偷偷加大epsilon。F不是标准SGD；P不是从头beta1=0训练。它们保留m/v更新规则，并使用本分支A的幅度作为参考。目标块a=0时也照常构建梯度、更新m/v，只取消data displacement。

其他所有参数方向仍为自己的原AdamW A_p。每个参数组沿用相同原调度eta_t及lambda_p；令：

    base_p = (1 - eta_t * lambda_p) * theta_p
    theta_target_next = base_target - eta_t * a * D_t
    theta_other_next  = base_other  - eta_t * b * A_other

a为目标块data倍率，b为其他参数data倍率，两者独立。不能把b当优化器LR去同时改变weight decay。普通AdamW恰是S,a=1,b=1。

沿用父512步调度的真实global step：主分支256→320，辅助128→192。不把64步重压为新cosine，不cold restart，不重置moments/step/beta，不把窗口0的A/g/V固定使用64次。每步必须重新计算所有方向。Norm matching成本计入运行成本。

可通过原optimizer.step后在同一步做经过验收的数据位移替换，或实现等价adapter；必须与原生S,a=b=1对齐并记录真实step调用数，不能漏记。

## 3. 冻结网格与预算

固定a ∈ {0,0.5,1,2}，b ∈ {0.5,1}，family ∈ {S,F,P}。
a=0时三个家族的完整轨迹相同，只运行S并作为三个家族共同候选复用。每家族都有8个可选配置（其中2个a=0共享），总计20条唯一轨迹。

主起点20条，每条H=64，共1280个连续训练optimizer更新。
辅助起点最多4条（冻结S/F/P各一个+普通S1,1），重复配置去重，最多256更新。
两条one-shot对照（冻结F、P各一个），最多128更新。
真实CPU测试不算GPU步；真实GPU集成/恢复smoke最多16步。
理论有用总量≤1680步；本轮新增GPU optimizer.step尝试硬上限1792，含smoke、失败尝试、丢失的未持久化步骤和恢复重算，剩余不是扩网格预算。

新增“从随机初始化完整预训练轨迹”=0，但新增短分支训练不是0步，必须分别报实际数量。
累计本轮token-forward-equivalents上限800万，包括训练前向、验证前向、失败前向及校准；反向次数与有效train tokens另计。token-equivalents不是FLOPs或能耗。

不运行V14旧局部扩网格，不增加rank/模块/attention/seed，不加入Muon、额外beta1扫描或预测器。本轮所有网格有限；选择落边界仍标注scale_range_unresolved，不能借此自动扩网格，也不能称“全局最优标量/强AdamW已解决”。

## 4. 数据与盲确认

开发评价集：沿用V12/V13的128个new_probe_prefix自然文档，明确已经用于开发。
本轮确认集：在父消费窗口的probe_confirm角色内，排除V1/V12/V13/V14及其他已记录运行的全部已用文档ID，按doc_id字典序取首128个未用文档；不看loss、长短或结果选样。逐字/规范化文本身份排重与训练角色保持互斥；不动最终holdout和branch保留角色。已有消费窗口不足则停报，不自动下载/换域。

训练前封存本轮确认IDs、选择规则、token计数和内容哈希。确认不参与任何梯度、学习率、幅度、家族选择。自然前缀长文不注入人工EOS；按真实有效token求和再除count。

每条开发轨迹只评H0、H1、H16、H64：H0相同状态共享；H1直接测实际后续batch上的首步效应，H16解释短时变化，两者均不用于选最佳时点或提前淘汰；唯一选择终点H64。正常完成64步，除NaN/Inf/OOM/资源/预算违规外不因为成绩不好提前停。

对每家族按H64开发NLL选择唯一配置。完全同分的预设顺序：优先a=1,b=1，其后最接近1，最后稳定候选ID；a=0跨家族仍明确标成共享S。必须在打开确认损失前保存selection_frozen及其依赖哈希。

确认评估主起点冻结S/F/P及普通S1,1的H1与H64状态；同样评所需H0共同基点。H1在选择冻结后才打开确认损失，H1不能代替主H64。未选分支不在确认上重排，不用确认结果更新配置。

## 5. 两个固定的时域诊断

### A. 一步介入后恢复基准方向（主起点）
对冻结F与P各做一次：第1步使用对应F/P，从第2步至64步改回S；a,b始终保持该fresh候选的冻结系数不变，m/v一直自然演化，不重置。
对照三者：持续fresh、one-shot fresh、从第一步就S；后者必须使用相同a,b，不要拿另一组最优S的a,b作唯一因果对照。相同系数S轨迹已在20条开发运行中，直接复用。

为此允许额外打开最多两个“与F/P匹配系数的S”终点在本轮确认集上评估。它们仅用于匹配系数的预设诊断，不反过来替换已经冻结的主scalar控制。

这组测量用于判断单步领先是否消退、持续介入是否有价值。one-shot的H1与相同系数持续fresh的H1应一致，核验后复用；不能只用V14任意构造窗口的首步效应替代本分支真实H1。one-shot不是每16步重来，不根据曲线改变介入频率。若a=0或配置实际相同则复用，不能重复记成独立实验。

### B. 同一训练轨迹的另一个起点（step128）
若预检确认完整step128存在，原样执行主起点已经冻结的S/F/P配置以及普通S1,1，各64步；不重新调参。评价相同开发/确认自然文档（配置已冻结，不再用于选择）。
它是同一seed的时点迁移诊断，不是新seed确认。128和256的共同文档不能当两份独立样本。缺少完整state时如实NOT_RUN，不伪造恢复或从零补训练。

## 6. 最小数值与实现验收

只做本轮新增行为的CPU回归+≤16真实GPU步的集成smoke，不重复历史大审计：
- S,a=b=1逐步对齐父原生AdamW及有效token loss；a=0只去data项、decay保留；无共享参数重复更新。
- F/P中v_plus、bias correction、剪裁时点、范数匹配与V14语义一致；所有分支自己的状态自然演化，不偷用基线梯度或固定初始A。
- 四种方向/状态的建立不会读确认梯度；eval不污染参数、grad、moments、RNG和游标。
- 连续与checkpoint恢复一致；one-shot只在分支第1步触发，恢复后不会重复触发。
- 预算先记尝试、结果逐步持久化，Windows原子替换有有限重试；续跑检查身份并继承累计账本，不能删除attempt或清零。

对来源文件保持严格hash；浮点更新/损失按已验收数值指标比较。不得只因字节SHA不同就宣布GPU不可复现，也不得删除有意义的数值验收。

## 7. 统计、裁决与报告边界

主量：每家族冻结H64确认NLL，Delta_F = L_Sbest - L_Fbest，Delta_P = L_Sbest - L_Pbest。每个NLL均按loss_sum/valid_token_count。

预先以doc_id为重采样单位，同时保留同文档全部分支/时点配对；5000次bootstrap，固定seed340715。对两项计划fresh-vs-S主比较报告普通95%CI，并报告Bonferroni校正97.5%CI（分位数0.0125,0.9875）用于本轮继续投入裁决。只刻画文档不确定性，不代表seed方差。

工程分辨率沿用2e-6 nats/token，按本轮重复测量记录真实误差；不把它当严格误差上界。H64的投入参考点收益delta_branch=1e-4 nats/token（本轮事先设定、不是理论常数）。

结果分别输出：
- BRANCH_SIGNAL_WITHIN_GRID：至少一个预设fresh候选主起点点收益>1e-4，校正区间下界>已测数值分辨率，且辅助起点若已测不出现明确反向证据。只能建议下一轮独立seed/更强LR基线，不宣称新方法成功。
- ONE_STEP_ONLY：局部领先/一步介入短时领先，但H64未保留；必须有对应曲线/效应数据，不能由缺失数据猜测。
- NO_BRANCH_GAIN_WITHIN_GRID：两类fresh均未通过H64裁决（可与置信不足区分）。
- INCONCLUSIVE：CI/数值问题尚不足判别，或主要必需结果缺失；具体写原因。
另单独保留scale_boundary、temporal_replicate、numerical_status、parent_gate，不让一个边界状态覆盖其他已测事实。

不要把点收益未通过解释成“没有任何可能收益”；也不要把短分支有效写成从头预训练有效。普通AdamW和开发选出的标量控制分别报告。未来真实LR/beta/WD充分调参仍未完成。

one-shot、H16、跨起点、匹配系数比较均明确为预设诊断/次要结果；不从中挑一个替代主H64。

## 8. 成本、资源和停止

单GPU串行，每次一条训练轨迹，数据memmap，评估逐文档或固定小批。保留原1.5GiB启动/1.5GiB警告/1.0GiB硬停、RSS4GiB及VRAM限额。不要为了跑通再降阈值或自动关程序。

运行前估计本轮磁盘占用并检查空间；不把所有全模型状态同时驻留内存。H16只需loss和必要的在途恢复state；保存可核验终点和H1参数/缓冲区快照（H1仅供后续评估，不伪装为完整恢复checkpoint），不用同时把这些快照放内存。不要未经批准删除旧父checkpoint或其他实验。

记录训练forward/backward、optimizer/方向归一化、评估、checkpoint/数据IO、GPU计时和总wall-time；计算训练有效tokens/s，不拿评估大量文档的时间当训练吞吐。没有功率积分不宣称能耗下降。

NaN/Inf/数据身份错误/无法校准/资源/预算超限停止受影响执行并报告。可定位的小bug可在新attempt修复、留补丁和回归、继承已用预算；同一问题不无限重试。不得因负科学结果改设计。

本轮完成后无论正负均停止；不自动扩大到独立seed、更多层、预测器或服务器训练。

## 9. 必交产物

- `V15_FINDINGS.md`：先给终点主判断，再解释H16/one-shot/跨起点；注明全部范围。
- `config.json`、run_manifest、selection_frozen、父依赖、确认ID排除链、完整最终budget_ledger。
- 每条branch的起点/身份/系数/逐步train loss、有效tokens、实际lr、moments step、目标方向范数与匹配倍率；逐文档H0/H1/H16/H64 loss_sum/count；不只给均值。
- 端点主表、配对bootstrap、边界状态、one-shot matched-S比较和时间起点复核。
- 真正执行过的`RUN_VERIFIED.ps1`、测试与运行原始日志、错误attempt、成本/资源摘要。
- `v15_light.zip`包含代码、配置、原始数值、最终账本、报告、hash与列出的真实入口；不含语料正文或所有大型checkpoint。重新打开ZIP检查关键文件和账本是否在包内。

不预先假设新CLI已存在；你可以实现必要的小入口，再报告真正成功执行的准确命令。最终不能只交“代码完成，等待运行”。
