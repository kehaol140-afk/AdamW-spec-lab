# OpenCode任务：最小勘误后执行三臂256步开发性续训

请先完整读取 `R2_EXECUTION_PROTOCOL_CN.md`、`r2_contract.json` 和 `references/` 中的独立审计及旧协议。全程中文汇报。

本轮已经获得用户同意：先修复已定位的恢复错误，再运行预定三臂，等真实结果后再讨论方法和理论。不再另交一份“等待启动”的协议。仅当技术正确性、资产或预算存在实质性阻塞时停止。

## 1. 本轮不再找新方法

复用现有muon_spectral_allocation项目、父seed3101的step512/1536完整checkpoint。不新训父模型，不换数据、模型、NS、momentum或系数。不运行O组/NS3、AMO、OrScale等新臂，不扩大幅度网格，不重启旧M2和顺序实验。

## 2. 先完成最小勘误

独立审计定位的核心问题是：PRE_STATE恢复了所有参数，但回到native时只恢复Muon参数，辅助AdamW参数遗漏，随后扩展校准复用了错误基点。

必须先：
- 完整恢复native一步后的全部unique参数和必要buffers；
- 修复并测试N→P→N、P→扩展、候选单独/批量一致、payload/CPU step隔离；
- 各组实际写回原时钟下一步LR；
- 最终锁与边界处理使用同一个既定near-best规则；
- 实际等范数从最终参数减共同WD基点计算。

原配置、原batch、原目标只补176条受影响扩展校准，按旧规则生成corrected锁；缺少的必要检查点最多72个。旧完整结果不覆盖。已有验证过的更正版可核对复用，不重复运行。

修复与勘误不用于改变本轮0.5倍率。新续训不是“选出了最优倍率”，而是验证已经事先指定的简单处理。历史规则证据不足如实记录，不倒签。

阶段A上限30 GPU-minutes，包含在全轮预算。正确性未闭合不得开始续训；不要重复完整平台审计。

## 3. 绑定并执行固定三臂

|臂|Muon数据项|辅助AdamW数据项|
|---|---:|---:|
|N 原生|1|1|
|MH 只缩主矩阵|0.5|1|
|AH 主辅数据一起缩|0.5|0.5|

每臂每状态运行256次真实updates，共六条主轨迹、1536次更新：
- step512 → 原时钟513…768；
- step1536 → 原时钟1537…1792。

六主run完成后，预算允许时做一条预定step512/N完整重复，最多再256步，只用于数值复现，不算独立训练seed。

正式每步都在本臂当前theta重新forward/backward、原生累积、全体unique参数共同clip。沿checkpoint的原数据cursor和原schedule，不重启/压缩/改成恒定LR。每状态各臂同完整batch流和初始RNG，后续梯度自然不同。

## 4. 缩放的是数据更新，不是梯度，也不是整个参数差

对每参数组执行：

    theta_next = (1-base_lr*base_wd)*theta - r*base_lr*native_data_direction

共同WD衰减乘积不随r改变；m/v/momentum/step仍按未缩放的实际剪裁梯度原生更新。不reset，不把.grad乘0.5，不把含WD的总delta整体缩小。

完整协议允许经实际验证的原生实现方式：

    effective_lr = r*base_lr
    effective_wd = base_wd/r

每步从独立base配置和原时钟重新计算，不得把上一步已缩放值再乘0.5。实际decay乘积保持相同；所有base/effective值均记录。不能简单改LR而忘记WD补偿。

先用合成状态及两个真实首步核对：N与生产native等价；同输入时moments与step一致；MH/AH实际数据delta分别为规定倍率；WD作用不变；分支完整状态不共享。

## 5. 评价与锁定

运行前固定六run配置、checkpoint SHA、train manifest、Q_P2_endpoint清单。

训练中只在64/128/256记录Q_select诊断，不允许据此改参数或提前结束正常run。评测后恢复训练模式/RNG。

每run保存step1模型快照、step256完整state。
所有训练完成后再统一读取已经预留的Q_P2_endpoint，评价原起点、step1和step256。两域分别sum/count，再固定等权平均。旧confirm禁止使用。

若该P2池曾被使用，如实标记开发暴露；缺少池或关联组清单不得临时从confirm补样本。

固定比较：
- V_N−V_MH；
- V_N−V_AH；
- V_AH−V_MH。

正值有利。两个状态分别列出，不用平均掩盖反号。首步和256步分开，不把旧一步收益乘256。1e-3 nats只作为预先声明的终点投入参考线，不是显著性或自动继续门。

有文档/线程/翻译关联组时可做配对目标bootstrap；不能把token、两个同父checkpoint或native重复当独立训练seed。无法计算可靠区间就给完整点估计，不编造。

## 6. 预算与交付

全轮总计最多2 GPU-hours，CPU数值测试30分钟，新输出4GiB；包括修复、加载、编译、F/B、NS、评价和失败费用。达到上限保存可恢复状态，写PARTIAL_BUDGET_STOP，不改horizon或评价规模。

保留一run一配置、LR/WD轨迹、计数器、训练loss、目标sum/count、完整状态hash、失败记录。最终中文报告回答：
1. 一步缩幅收益有没有延续到256步？
2. 只缩Muon与主辅一起缩的结果有何不同？
3. 两阶段、两域是否一致？
4. 是否仍仅是基线校准信号？有哪些未完成项？

交付R2_REVIEW_BUNDLE.zip（目标15MiB内）：修复说明、代码/测试/配置、目标表、配对表、身份与成本清单。模型checkpoint、step1快照、语料留服务器，不塞入审阅包。上传失败先发报告和关键CSV，不为上传重跑实验。

完成后停止。无论结果好坏，不自动增加倍率、种子、窗口、结构臂或新优化器。
