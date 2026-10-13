# OpenCode任务：补L＋交叉续训，条件补D

先完整读取 `R2F_EXECUTION_PROTOCOL_v1_CN.md`、`r2f_contract.json`，再读references中的R2审计、旧协议及原始轻量结果包。本轮协议才是新增执行边界。全程中文报告。

用户已批准本轮有界筛查：先做实际证据核对和必要修复，通过后直接完成实验，不只交一份“等待审批”的新文档。存在具体资产/配方/数据/预算阻塞才停止。

## 1. 开跑前，只做必要的零训练复用检查

父seed3101：原step512、1536；旧R2 N/AH完整末态：768、1792。

- 导出文件SHA、model内容hash、完整state与计数器、数据cursor/RNG、原始逐group LR/WD及参数名映射。
- 验证旧N/AH来源与原配方；重点查统一广播LR、广播第一组WD的问题。如果原组本来不同且旧执行已违背，停止，不能拿修正后L与错误旧baseline混比。
- 同pool核对旧N/AH终点值；缺旧逐步loss日志如实写MISSING，不编造或为补日志重训。
- AH保存的是effective参数，切换时必须从原始base重建，不得0.5再乘0.5、WD再翻倍。
- 修复每run日志持久化、checkpoint完整性判断和真实首步门。中断不覆盖其他run日志，完整文件经可加载和来源核对才可跳过。
- R1均值修正若处理，按键替换旧176条记录，不能混合平均；只做CPU/CSV勘误，不加旧R1 GPU探针。

## 2. 四配方逐group定义

每个组自己的原schedule eta_g(t)、lambda_g不可变。
执行 theta+=(1-d*eta_g*lambda_g)*theta-r*eta_g*direction。
用原优化器时，lr_eff=r*eta_g；wd_eff=(d/r)*lambda_g。

N: r=1, d=1；
AH: r=0.5, d=1；
L: r=0.5, d=0.5；
D: r=1, d=0.5（条件运行）。

不缩grad，不重置moments，不丢弃辅助AdamW，不改NS5/coeff/scaling/betas/precision，不添加W。
逐组保持原本不同的base LR、WD和零WD mask。
必须通过异质组、N→AH、AH→N、AH→AH、保存恢复、payload不共享、eval不扰动训练的测试。真实首步测试在完整续训前断言。

## 3. 先补L

两条新分支：
- s512_L256：原step512开始，消费513…768；
- s1536_L256：原step1536开始，消费1537…1792。

与旧R2同batch成员、顺序和起始RNG。真实每步重新求梯度。
完成两条后统一评价，同Q_P2_endpoint计算B_L=V_N−V_L、B_decay=V_L−V_AH。

只有两个origin的B_decay都>0.001 nats/有效target，才冻结D_required=true。
该规则只决定预算，不称为跨seed统计显著。相近或一正一负不自动补D。
先写D_TRIGGER；不得看交叉结果后重新改触发。

## 4. 四条交叉，两个origin共八条新分支

从旧R2 N/AH的256步完整末态出发，分别接256步N或AH：
N→N、N→AH、AH→N、AH→AH。

- origin512：从global768开始，消费769…1024；
- origin1536：从global1792开始，消费1793…2048。

只改后半程规则。每条保留自己的theta、buffers、Muon momentum、aux m/v/step、RNG和cursor；不拼接/交换/reset状态。
同一origin四条尾部使用相同未来batches和原2048总时钟，不重启、不压缩、不改常量。
“尾部相同”仅指规则与数据，不声称梯度、有效步幅或状态相同。

主要比较：
R_N=V(N→N)−V(AH→N)；
R_AH=V(N→AH)−V(AH→AH)。
同时报告相同前缀的尾部规则差、路径交互、逐域值和初始领先的保留比例。保留比例不截断，不归因为纯动量或纯权重。

## 5. 条件D

先保证L＋八条交叉完成；D_TRIGGER=true且预算剩余时，从两个原父起点各补256步D，共512步。用与L相同原窗口。
D未触发或资源不足准确列状态，不用其他配方替代，不补W，不扩网格。

## 6. 评价与预算

全部训练用原真实训练循环、全参global clip、独立branch状态。
训练仅用Q_select诊断；Q_P2_endpoint在每阶段完成并写评价锁后统一评价。它是已经暴露的开发目标，旧confirm不访问。
保存新增step1/16/64模型快照，256完整state；早期读出不用于选赢家或改配方。L/D主要终点为256，交叉主要终点为累计512。
有完整目标关联组映射才做配对cluster CI；没有则CI_UNAVAILABLE。不按token当独立样本，两个同父checkpoint不是独立seed。

新增必做2560 updates，条件D后上限3072；工程和失败更新另计并受总费用约束。
全轮累计最多3 GPU-hours、10GiB新输出、CPU数值30分钟；包括所有加载/验证/评价/失败。到限保存停止，不删不利分支或缩目标。

## 7. 交付后停止

新目录outputs/r2f_lrwd_retention_v1/，保存：
- reuse_gate与完整逐group manifest；
- executed config、run/data/target/source hashes、D_TRIGGER；
- 每run不可互相覆盖的日志与完整状态清单；
- sum/count、lrwd_attribution、cross_path_effects、可用时CI；
- 失败/恢复/成本台账，中文结论及复跑命令。

压缩R2F_REVIEW_BUNDLE.zip，目标≤15MiB，不放weights/语料/环境。
结论分开回答：普通LR解释多少；前段优势在相同尾部规则下保留多少；阶段/域是否一致；哪些信息尚不能归因。

不开发预测器、不添加新父seed、不跑完整模型比较。后续第三至第五阶段只能在这轮审阅后另行批准。
