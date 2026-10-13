# R2-F：LR–WD归因与交叉续训执行协议 v1

日期：2026-09-29。性质：依据用户本轮建议制定的前瞻执行任务；本轮新实验尚未运行。

**目标：**先检验普通LR减半能否解释现有AH收益，再检验在相同后续规则下，前256步AH形成的优势是否仍被保留。不开发新优化器或预测器。

**架构：**复用现有完整训练器与父状态；在每组原始基准LR/WD之上显式施加两个倍率。完整状态随真实梯度递推；阶段切换只改变配方，不交换或重置状态。日志按run独立落盘，目标评价与训练分离。

**技术栈：**现有Python/PyTorch/MiniMind环境；不要求升级依赖。

**依据：**用户本轮“补L＋四条交叉续训”建议、原R2执行协议、R2独立审计。原审计仅提供历史证据，不替代服务器真实checkpoint、配置和数据。本文取代上一条审计建议中的“优先补W”：本轮补L，按条件补D，不同时补W。

## 0. 研究边界和本轮授权

研究问题：Muon续训中，降低数据更新幅度的收益，主要来自普通降LR、实际衰减水平，还是在训练路径和完整状态中留下了可保留差异？不同剩余预算下选择配方的问题是后续方向，本轮不训练选择规则。

本轮必须项：零训练复用门 → 两条L分支 → 冻结是否补D的决定 → 八条N/AH交叉后半程 → 若触发则两条D → 一次汇总。

主矩阵新增10条×256步=2560次实际更新；条件D新增2条×256步=512次；总上限3072次主分支实际更新。旧N/AH第一阶段不重新训练。

资源新上限：累计3 GPU-hours（两卡计和）、CPU数值工作30分钟、新增磁盘10 GiB。失败尝试、加载、梯度、NS、评价、检查、数据移动均计时。超限保存恢复点并停止；不得缩短horizon、缩小目标或删掉不利分支。主分支步数与工程检查/失败尝试步数分别计数；资源上限包含全部工作。

全轮不新增父轨迹、种子、模型、域、K、谱系数、beta、倍率搜索；不运行预测器、第四阶段文献方法比较或完整预训练。旧M2、O/NS3与联合谱分配不重启。0.5是预先指定候选，不是已证最优。

### 优先核查的五类错误

1. 把第一个group的LR/WD广播给全部组，破坏原配方。
2. 从AH末态的有效LR/WD继续乘倍率，造成0.25×LR或4×WD。
3. checkpoint恢复或监控评估改动状态/RNG，导致同输入对照不成立。
4. 中断后只按文件存在跳过run，同时覆盖旧日志、旧锁文件。
5. 把两个同父恢复点当作独立seed，或把路径保留解释成纯权重/纯动量机制。

## 1. 文献与原结果怎样使用

现有R2独立复算的256步N−AH为：step512 0.072973962；step1536 0.006578662 nats/有效target。这是开发性固定目标结果，不是完整训练或新种子确认。

本轮材料核对所查原文：
- Wu等，Understanding Short-Horizon Bias in Stochastic Meta-Optimization，2018，https://arxiv.org/abs/1803.02021 ：短视野目标可能偏好过小步长；不将一步排名反转当作首次发现。
- Apte，Scale Weight Decay and Train Better，2026，https://arxiv.org/abs/2607.23777 ：研究随LR缩放WD和相应理论/实验；Muon-SW不等于本轮固定AH，不直接套用其收益或保证。
- Everett、Qiu，Optimizer Memory Schedules for Outscaling the Overtraining Axis，2026，https://arxiv.org/abs/2609.04577 ：训练horizon影响优化器配置和排名；本轮同一原schedule的中间读出不是每个horizon充分调优结果。
- Liu等，Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining，2026，https://arxiv.org/abs/2608.24814 ：LR/参数范数的经验联系；本轮记录norm与实际更新比作诊断，不能把任意update/weight ratio等同论文所有ELR定义。

用户原大纲要求“条件性预测→真实路径→终点和成本”，仍保留这一原则；不为保留热力学名称忽略普通调参解释。

## 2. 阶段0：只读复用门与最小工程修复

### 2.1 所需状态（主分支开跑前必须核准）

父seed3101：原step512和1536完整状态。旧R2的N、AH在256步后的四个完整状态：
- s512_N、s512_AH：global_step=768；
- s1536_N、s1536_AH：global_step=1792。

核对模型参数/必要buffers、56个主矩阵momentum（实际路由为准）、辅助m/v/step、RNG、数据cursor、scheduler状态；不将权重文件冒充full_state。

输出 `asset_certificate.json`：实际路径、文件大小/SHA、model内容hash、完整状态键与计数器、原配方版本、参数对象/组映射、数据/目标manifest SHA。核查旧N重复的模型内容hash，仅用于重复性。

要求四个旧末态能在同一目标pool上重现原V_Q及逐域值；固定原评价代码/精度/归约，差异绝对容差1e-7作为本轮身份复用检查，不作为训练效应阈值。超限先定位，不换评价池。若存在已知后端差异，登记后停止该身份确认，不事后放宽阈值。

旧逐步日志缺失登记MISSING；由公式恢复的字段登记RECONSTRUCTED，不伪造loss/grad实测。只因日志缺失但完整状态/数据/配方身份齐全，不要求重训。关键完整状态或数据无法确认则BLOCKED，不重做旧第一阶段来凑齐预算。

### 2.2 逐group配方必须忠实

从父训练实际config、builder和checkpoint建立不可变 `base_group_manifest.json`。每个组绑定optimizer类型、参数名集合/摘要、基准LR函数、基准WD、decay mask、betas/eps、NS/scaling、其他行为flags。

禁止把一个group的LR或WD应用于所有group。若原配方恰好全组相同，用清单证明后才允许共享计算。先核查旧R2的广播行为与实际原配方是否一致：
- 确实一致：继续复用；
- 存在实质差异：旧N/AH不能与修正后的L混作同配方归因，停止本轮，列出具体组和影响；不静默修正、同时保留旧baseline。

新配方永远从不可变原始base重建，不能从AH末态保存的effective WD倒推后再乘。

### 2.3 本轮必要工程修复

- 每run独立目录和train.jsonl；每条记录含run_id、source_checkpoint_sha、global_step、config_sha。中断恢复只能追加缺少的步，聚合表从独立日志重建，不能每次重启w覆盖历史总表。
- checkpoint写临时文件、完成后校验可加载及schema/计数器，再原子替换正式文件；COMPLETE包含输出hash，不以“文件存在”判断完成。
- 每次分支深拷贝模型和optimizer可变状态，包括CPU step；恢复参数参考在load后建立。
- 首步验证必须在主训练前断言，不允许先训练再补PASS。
- 如修正R1聚合，按(state,batch,point_id)用176条新记录替换旧记录，不平均新旧；不新增R1 GPU评价、不根据修正结果改变本轮配方。

## 3. 配方：数据更新强度 r 与实际衰减乘积强度 d

对组g和原全局步t，基准为 eta_g(t)、lambda_g；方向v来自该臂当前theta/历史状态和真实剪裁梯度。

    theta_next = (1 - d * eta_g(t) * lambda_g) * theta
                 - r * eta_g(t) * v

可通过原生优化器实现：

    effective_lr[g,t] = r * eta_g(t)
    effective_wd[g,t] = (d / r) * lambda_g

本轮r非零。lambda_g=0的组仍为0；无梯度参数遵守原生skip规则。eta/WD两个乘子均应用于Muon与辅助AdamW所有对应组，但每组base不相同也必须保留。

|配方|r 数据项|d 衰减乘积|LR倍率|配置WD倍率|
|---|---:|---:|---:|---:|
|N|1|1|1|1|
|AH|0.5|1|0.5|2|
|L|0.5|0.5|0.5|1|
|D（条件）|1|0.5|1|0.5|

AH−L在相同LR下改变衰减乘积；N−D在相同原数据步长下改变衰减乘积。N/AH/L/D构成(r,d)的2×2，不是LR×配置WD的矩形。W=(LR1,WD2)不在本轮。

禁止缩放.grad、reset/mixing moments、对含WD的整个delta直接乘0.5。改变配方后，真实梯度随轨迹反馈变化是正常现象，不强制各臂moments相同。

### 3.1 必须先通过的测试

使用异质参数组（不同LR、WD且含WD=0）、tied权重、CPU step：
- 每组r/d与预期完全对应，base不变，连续两步不会重复乘倍率；
- 同一输入/旧state时四配方的moment更新一致，数据项与decay乘积按定义改变；
- 同一输入的独立原生N一步模型和optimizer状态一致；
- N→AH、AH→N、AH→AH切换后有效LR/WD符合原base，不从旧effective值累乘；
- uninterrupted与save/load续跑的小例子一致，数据游标/原时钟不重复或跳步；
- monitor不改变训练RNG、mode和必要buffers；
- N→P→N与A→B→A恢复，实际最终delta对照，错误会触发停止。

真实首步检查在两个原父起点、以及四个后半程来源态上进行，复用其固定下一batch；不以NLL选择是否通过。相同后端的N对独立原生一步要求bit一致；数据delta相对误差≤1e-4、参数绝对误差≤5e-8+1e-6*|reference|；退化delta单列不以除零比值判断。工程检查不计为科学分支，但计算费用全部记账。

## 4. 阶段1：先补两条L

每条从原父状态开始，完整256个真实更新：

|run_id|起点|消费原全局步|比较的旧终点|
|---|---:|---|---|
|s512_L256|512|513…768|s512_N/AH|
|s1536_L256|1536|1537…1792|s1536_N/AH|

使用与旧R2相同前256步tokenized/packed batch成员、顺序、起始RNG和全局clip；各臂在当前theta重新forward/backward。没有冻结梯度回放。

两条L完成后统一评价终点（及已固定诊断快照），输出：
- B_L=V_N−V_L；
- B_decay=V_L−V_AH（正值表示固定数据步长下，保留原衰减乘积更好）；
- 全部逐域值和配对差。

### 4.1 条件D：把“必要”变成事先规则

新增预算分配线 tau=1e-3 nats/有效target，两个起点分别计算。

只有两个起点的 B_decay 都严格 >tau，且状态/目标身份通过时，设 `D_required=true`，冻结 `D_TRIGGER.json`。该阈值只决定是否值得花512步补完整四格，不是跨seed显著性声明。

其他情形（相近、反向或阶段不一致）不自动补D，完整报告原因。不能因交叉续训结果好坏再更改D触发；边界值等于tau不触发。无论是否触发，八条交叉主分支仍按固定设计执行（技术阻塞除外）。为优先完成必做10条分支，D在交叉分支完成后、剩余预算允许时执行；已触发但预算不足写PARTIAL_BUDGET_STOP。

若补D：s512_D256和s1536_D256，各消费与L相同的原窗口。禁止同时补W或做网格。

归因四格的差异之差：

    I_r,d = [V_N - V_AH] - [V_D - V_L]

这衡量该设计下数据缩幅的效果是否依赖衰减乘积；不是自动的物理协同或连续最优。

“AH与L接近”若仅由点差小支持，表述为本轮未分辨实用增量；不由p>0.05或CI含0证明等价。若完整评价关联组存在，可补条件于模型的区间，仍不是训练seed等价结论。

## 5. 阶段2：八条交叉后半程，识别优势保留

AH在显示表中简写A，代码和文件仍写AH。

|origin|来源run|后半配方|新run_id|global_step|
|---:|---|---|---|---|
|512|s512_N end768|N|s512_N_to_N|769…1024|
|512|s512_N end768|AH|s512_N_to_AH|769…1024|
|512|s512_AH end768|N|s512_AH_to_N|769…1024|
|512|s512_AH end768|AH|s512_AH_to_AH|769…1024|
|1536|s1536_N end1792|N|s1536_N_to_N|1793…2048|
|1536|s1536_N end1792|AH|s1536_N_to_AH|1793…2048|
|1536|s1536_AH end1792|N|s1536_AH_to_N|1793…2048|
|1536|s1536_AH end1792|AH|s1536_AH_to_AH|1793…2048|

每条新分支只追加256步，前256步来自旧完整状态，不能重新从原父模型开始假装交叉。

### 5.1 切换语义

- 保留每条来源态自己的theta、buffers、momentum、m/v/step、RNG、cursor；不同路径的状态不交换、不平均、不reset。
- 只把下一步配方从原base和新策略重新写入。
- 原2048总步数的schedule继续，不重启、不压缩、不改常量、不超2048。
- 同一origin四条尾部batch成员与顺序相同；N→N和AH→N是规则相同，不是梯度或完整状态相同。
- 对相同来源态的两个后半策略，初始RNG完全相同。不同来源态保留各自原RNG；若二者RNG不一致必须报告，不偷偷用另一路RNG覆盖以制造CRN。
- 检查原R2 N与AH的cursor确实对应同一未来数据流；存在成员不一致则停止该origin。

### 5.2 主比较（总续训512步的目标）

令J_s(P,Q)表示从origin=s出发，前256步用P、后256步用Q，最后目标。

    R_N(s)  = J_s(N,N)  - J_s(AH,N)
    R_AH(s) = J_s(N,AH) - J_s(AH,AH)

它们识别在相同尾部规则下的前段路径效应。正值表示早用AH仍有领先；不是纯参数、纯动量或唯一物理机制。

另列：

    T_N(s)  = J_s(N,N)  - J_s(N,AH)
    T_AH(s) = J_s(AH,N) - J_s(AH,AH)
    I_path(s) = J_s(AH,AH)-J_s(AH,N)-J_s(N,AH)+J_s(N,N)
              = R_N(s)-R_AH(s)

初始领先B0=V_N(256)−V_AH(256)取同目标身份复核值。若B0>tau，可描述R_N/B0、R_AH/B0，允许负值或大于1，不截断为[0,1]，也不按记忆衰减系数解释。

解释规则：
- R_N、R_AH均大于tau且两origin同向：有限窗口内有实用量级的路径领先保留线索，不称跨seed稳定或已发现机制。
- N→AH与AH→AH点差很小：晚切换在该终点可能已足够；不自动宣称统计等价。
- AH→N领先消失：优势依赖后续规则；仍可能含切换暂态和有效LR/norm差，不能断言之前未学习。
- 两origin模式不同：提出阶段/预算条件假说，不据两个样本拟合预测器。
- 同一个固定配方始终占优：优先更强基线，不能为了预测器强行声称状态选择必要。

## 6. 评价、诊断和证据级别

Q_P2_endpoint继续使用原pool和q=0.5，已经历史暴露，明确称开发目标。本轮不访问旧confirm，不生成伪装的新final。

训练中仅在新增相对步1/16/64/128/256记录旧Q_select，禁止依据曲线改臂或延长。保存模型state_dict（含buffers）在1/16/64，256保存full_state；加载后模型模式和RNG恢复正确。

每阶段的规定训练完成后，先写含run/快照/目标/数据/config/source hashes的评价锁，再统一对规定快照算Q_P2_endpoint。不能覆盖旧锁；技术重试复用相同锁，记录失败原因。

- L/D的主要终点为新增256步；
- 交叉主要终点为总512步（新增256步）；
- 新增1/16/64只诊断切换暂态，不选最佳中间点、不选择D、不改变剩余分支。

逐域保存CE sum/count、NLL、总体V、重复误差。若原pool关联组映射完整，可在一次前向内导出每个关联组loss：按跨语言线程/翻译关联组做配对cluster bootstrap（10000次，seed=3101512），每次抽样权重在同一配对各臂共用，逐域sum/count后q=0.5组合。区间仅表征固定模型的评价样本不确定性。关联组缺失写CI_UNAVAILABLE，不把token/windows默认当独立样本。

同父seed两个origin不是独立训练seed；0.001是本轮实用讨论/预算门，不是统计显著性边界。重复值完全一致不估计采样不确定性。跨臂差值用同一个时刻、同一个pool，不能把Q_select与Q_endpoint的绝对loss相减称模型误差。

可记录参数组/大块范数、实际更新范数与权重之比、preclip norm；每16步即可，优先输出真实测量，不加SVD/HVP或完整FDR推导。不要把任意norm ratio命名为唯一ELR或温度。

## 7. 执行顺序、文件与简明实现任务

推荐复用已有 muon_spectral_allocation/，新输出 `outputs/r2f_lrwd_retention_v1/`。旧源码修复可新增适配模块并保留diff，不修改安装版优化器与旧原始结果。

建议文件（现有等价模块可复用，但需映射）：
- `src/r2f_recipe.py`：group_id到原base绑定、策略r/d到effective LR/WD；
- `scripts/run_r2f.py`：L、交叉、条件D，独立run日志与恢复；
- `scripts/analyze_r2f.py`：独立sum/count、四格与路径差值；
- `tests/test_r2f.py`：本协议的非等组、切换、保存恢复及代数恒等式；
- `docs/R2F_REPORT_CN.md`、`configs/r2f_executed.json`。

任务1：先为旧广播/double-scaling写失败测试→实现按group的base配方→用异质组及CPU测试验证→生成组清单。
任务2：为切换恢复和日志续写写测试→最小修runner→验证完整状态/时钟/原生首步→冻结资产与run_matrix。
任务3：完成两条L→封存评价→按预设tau写D_TRIGGER，不拟合新规则。
任务4：完成八条交叉→封存评价；若D_TRIGGER且余量允许完成两条D→封存评价。
任务5：用独立sum/count脚本生成逐域和配对表；核对行数/run_id/global_step唯一性、2×2恒等式、实际预算；打包报告后停止。

主分支日志每step保存：run_id、origin、prefix、suffix、local/global_step、train targets、train loss、preclip norm、每group base/effective LR/WD/decay product、累计用时。较重norm诊断另表。恢复可用已有每64步full_state或本轮增加，老临时状态不要不断重复占磁盘；旧科学结果不删。

不允许因loss差而提前停正常run；NaN/Inf视为明确技术/数值结果，保存，不改参数救场。技术重试至多一次，全部消耗记账。不得kill其他用户进程。

## 8. 交付

1. `reuse_gate.json`与`base_group_manifest.json`：逐组忠实性、旧完整状态身份与缺失字段。
2. config/source/data/target SHA、run_matrix、D_TRIGGER（含未触发原因）。
3. 每run日志、完整状态清单、COMPLETE标识、所有失败尝试和恢复说明。
4. `lrwd_attribution.csv`、`cross_path_effects.csv`、逐域sum/count、可用时group-bootstrap。
5. `cost_ledger.json`：main唯一更新2560/可选512、工程/失败更新另计；复用旧计算另栏，不能将复用视为原实验零成本。
6. 中文报告分别回答来源归因、路径领先是否保留、阶段/域一致性、未回答的问题。
7. 轻量 `R2F_REVIEW_BUNDLE.zip`，目标≤15MiB；不含checkpoint、语料、环境、密钥。完整weights/buffers留服务器并列hash。

理论结论必须与数据分开：AH/L点差不是全局超参最优证明；交叉保留不是纯weights/momentum的拆分；同一schedule不同观察步不是不同预算调优结果；未做的Muon-SW、ELR解释、WSD/cooldown、AdamW等一律列未比较。

## 9. 本轮之后的路线（不自动执行）

只有归因与路径结果提供有用差异，才申请新父seed筛查、两阶段四配方至512步矩阵。若最佳固定配方跨条件已足够，就不开发状态预测器。

预测阶段用前16/64步观测事前预测指定H的配方，评价regret=V_selected(H)−min_a V_a(H)；全候选最优是事后参考，不让选择器接触未来。按整条父轨迹划分，不随机拆同轨迹行；比较最佳固定、短loss选择、简单外推和同预算搜索；所有探测费用计入，并让固定基线用同样墙钟继续训练。

数学/物理服务于“何时排名变化、何时路径效应保留”的可检验预测，不把已有滤波、有效LR、能量类比当新贡献。只有独立状态/种子上有终点与成本增量，才讨论规模、数据、优化器迁移和完整训练主实验。
