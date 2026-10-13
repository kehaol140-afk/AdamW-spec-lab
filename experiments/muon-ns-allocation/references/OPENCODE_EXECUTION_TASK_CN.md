# OpenCode执行任务：完成P0/P1并回答前三个立项问题

> 执行方法：用户已选择OpenCode。具备superpowers时按executing-plans逐任务执行、先测试再实现；没有该技能也按以下明确检查执行，不因工具差异重建系统。

**Goal：** 在本地Muon配方下，实测组级谱计算的目标价值、同预算分配、幅度替代解释及本地AMO机制对照。
**Architecture：** 复用旧训练/数据/状态/评价基础设施；新项目只添加配方、一步探针、成本匹配、锁定选择和报告。P1校准/检查完全分离，P2/P3不执行。
**Tech Stack：** 服务器已有Python/PyTorch环境，固定原生Muon及辅助AdamW；无新增服务或大型依赖。
**Spec：** `P0_P1_EXECUTION_ADDENDUM_v1_1_CN.md`。必须先完整读取。

## 全局约束

- 旧项目、安装环境和历史原始结果只读；M2、T3-mini仍关闭。
- 本次直接完成有界P0/P1，不停在只写文档；实质性来源/资产/预算阻塞才报告。
- 新父轨迹最多一条2048 updates；16个诊断train batches；总上限3 GPU-hours、4GiB输出。
- 本轮新增选择/门槛在新结果前登记；不能冒称它们此前已冻结。
- 所有对外文字中文。未完成或缺失证据明确UNRESOLVED，不用PASS掩盖。

## 审查重点

1. 状态step/momentum别名会导致不同候选并非同起点。
2. 把WD计入范数匹配会把原来的幅度对照变成另一个干预。
3. 两组一减一增不自动成本相等，联合收益也不等于单组之和。
4. AMO的PE系数、计数预算和scaling不能在移植时被隐去。
5. 检查侧选主候选、共享P2评价或把同父轨迹两状态当两seed，会夸大证据。

## Task 1：来源、状态和目标隔离

文件：`docs/RECIPE_AND_AMO_COMPARISON.md`、`src/state_restore.py`、`src/recipe_registry.py`、`tests/test_state_and_routing.py`。

- [ ] 读取四个历史markdown和补充协议，记录已有资产，不做全面旧实验重审。
- [ ] 写测试：load后建立参数参考；多个候选无共享可变m/v/step；实际被消费的G/H不被改写；tied weights恰好一次。
- [ ] 最小实现独立恢复和配方映射，原生K5通过完整一步no-op对照。
- [ ] 按协议生成Q_select/Q_P1_check/Q_P2_endpoint关联组清单，写访问保护测试。
- [ ] 固定新父seed/模型/原生时钟、AMO源码和本地化差异；关键来源不全只阻塞AMO分支。

## Task 2：成本和参数点构造

文件：`src/cost_meter.py`、`src/group_probe.py`、`tests/test_group_probe.py`、`tests/test_cost_matching.py`。

- [ ] 写测试：组内包含所有层；矩形残差min维定向；global clip包含辅助参数；zero/no-grad语义正确。
- [ ] 写测试：raw与gain_only_live组梯度更新范数相同；direction_only与native相同；WD/辅助完全不动。
- [ ] 写测试：组成本不等时同NS次数不会被错判同成本；同预算joint选择不调用Q_check。
- [ ] 实现候选有限一步构造，使用同一真实G和完整辅助更新，不累积多个候选对参数的更改。
- [ ] 用实际形状测C_g(K)，报告完整谱模块成本，不复用噪声大的单次计时作为结论。

## Task 3：父轨迹和校准输入

文件：`scripts/run_parent.py`、`scripts/run_p1_calibration.py`、`configs/p0_p1_v1_1.json`。

- [ ] 冻结配置并记录hash。已有同配置运行能核对则复用/续跑，不重跑。
- [ ] 训练唯一父轨迹，记录有效target/slots/实际LR，保存512、1536、2048。
- [ ] 如AMO来源已明确，按预定位置只读记录几何；不改变父轨迹/RNG。
- [ ] 两状态各4个校准输入，执行固定KJ原系数45种一步探针；目标只用Q_select。
- [ ] 保存校准原始sum/count、geometry和cost；不保存全部候选checkpoint。

## Task 4：同预算联合与本地AMO机制

文件：`src/joint_selection.py`、`src/amo_local_adapter.py`、`scripts/freeze_p1_selections.py`、`tests/test_selection_and_amo.py`。

- [ ] 写测试：最多42有序组对只按校准成本筛；加法仅提案；最终选择用联合目标实测。
- [ ] 枚举成本可行池，按协议选至多两个目标提案、一个几何方案、一个固定随机方案。
- [ ] 联合有限前向，锁定target_primary，计算并锁定gamma_const。
- [ ] 有可靠AMO实现时完成paperbudget一步probe与costmatched restricted机制比较，准确命名。
- [ ] 目标与几何的Q3比较共用PE系数表和可行池；不同系数族不能直接归因于信息增量。
- [ ] 单测PE composer/标量预测、预算、选择无Q_check依赖；来源不完整就显式留空Q3。
- [ ] 在所有检查评价之前生成唯一selection_lock，固定所有primary、controls和阈值。

## Task 5：检查侧实测和条件统计

文件：`scripts/run_p1_check.py`、`src/report.py`、`tests/test_reporting.py`。

- [ ] 写测试：gain=V_reference−V_candidate、same target、sum/count正确；所有表可从原始数值重算。
- [ ] 后4个检查train batches仅从共同anchor独立恢复；读取已锁定候选，不再次搜索。
- [ ] 原生、单组三联、联合、幅度与可用AMO主比较按配置完成；Q_P2_endpoint/confirm访问保持0。
- [ ] 输出逐batch/逐状态配对差、逐域值、成本、参考选择与候选向量。
- [ ] 检查集不能选新的赢家；两个状态不是两个seed；bootstrap不能用token伪造训练样本。

## Task 6：报告与交接

文件：`docs/P0_P1_REPORT_CN.md`、`outputs/p0_p1_v1_1/`、`P0_P1_REVIEW_BUNDLE.zip`。

- [ ] 逐题回答Q1/Q2/Q3并列明证据级别，不只说“全部通过”。
- [ ] Q1无局部信号，停止固定原系数K=3/5/7有限家族；不得扩大到否定全部谱分配。
- [ ] 便宜gamma_const能解释收益，则明确这不足以支持NS分配方法。
- [ ] AMO来源/成本/同族比较缺失，明确Q3尚未回答，不能写“超过AMO”。
- [ ] 将P2逐组标量基线、LR范围和预留终点集写进下一阶段草案，但不运行。
- [ ] 运行本轮实际相关测试，附新输出而非旧PASS布尔值。生成轻量review包，不含模型/语料/环境。

## 最终中文答复格式

先给Q1–Q3三行状态表，然后：
1. 实际执行与未执行内容；
2. 组级敏感性表；
3. 同预算联合比较与交互余项；
4. 动态/常量幅度对照；
5. AMO准确配方标签及同族结果；
6. 评价暴露、训练随机性和成本限制；
7. 是否仅值得准备P2，或应停止本家族。

完成后停止，不自动补更多seed、K、LR、scaling或新优化器。
