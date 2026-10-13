# RECIPE_AND_AMO_COMPARISON — 配方对照与 AMO 状态

日期：2026-09-29。

## 1. 本地原生配方（逐字段，来源=锚点 run 的 config.json + optimizer state_dict）

| 组件 | 值 | 来源 |
|---|---|---|
| 模型 | MiniMind 30,025,216 参数（56 Muon 矩阵 + 34 aux 张量，tied embedding 一次） | routing_report（旧核对）+ 本轮父轨迹恢复 |
| Muon | peak LR 0.01、WD 0.02、momentum 0.95、Nesterov（更新后 buffer）、NS5、(3.4445,−4.775,2.0315)、eps 1e-7、adjust_lr_fn=None（original: lr·√max(1,A/B) 并入 lr） | checkpoint param_groups + 安装 _muon.py（SHA 38e95918…） |
| AdamW aux | LR 0.01（同 Muon 时钟）、(0.9,0.95)、eps 1e-8、WD 0.02、decoupled | checkpoint param_groups |
| 时钟 | warmup 32 步 + cosine→10%，2048 步；探针 LR=恢复点下一步原时钟值（512→lr(513)=0.008798…按公式；1536→lr(1537)） | m1_runner.schedule_factor 复刻，测试锁定 |
| 数据 | p=0.5（每 step 64A+64B）、manifest seed 2030、128×256-slot 窗、32640 有效 targets/步 | 生产 manifest 复刻，父轨迹计数器断言 |
| 前向 | bf16 autocast、fp32 参数/梯度/动量、全参 global clip 1.0 | 生产循环复刻 |

**父轨迹**：seed 3101（首次使用，无碰撞），2048 步，loss 7.54→2.29，保存
512/1536/2048 三完整状态（mte_state_v1）。两恢复点同轨迹，非两个 seed。

## 2. AMO（arXiv 2605.17806v2）状态：COMPARATOR_UNRESOLVED

- **获取尝试**：`https://arxiv.org/html/2605.17806v2` 与 abs 页 webfetch 均传输失败；
  服务器无外网；全盘检索无本地 AMO 代码/论文缓存（framework-audit、baselines 等）。
- **缺失要素**：Alg.1/2（观测-分配-锁定）、Appendix C（PE 系数 composer）、论文
  shape scaling 与本地 original scaling 的差异表——均无法核对。
- **按补充协议 §9.2 处理**：AMO/Q3 记 `COMPARATOR_UNRESOLVED`；不生成 PE 系数表、
  不做"同系数族"比较（无来源即无同族）、不拿自写残差分配冒充 AMO、不把任何一步
  结果称为 AMO 论文终点复现。**Q1/Q2 的 KJ 家族结果不受影响**（它们不依赖 AMO）。
- 若未来论文可及：先补 §9.3 的 A（paperbudget probe）与 B（costmatched restricted）
  两个标签下的同族比较，再谈 Q3。

## 3. KJ 固定系数家族与 AMO 的关系边界

本轮 K=3/5/7 候选使用**原 KJ quintic 系数**仅改变 NS 步数——这是消融家族，
不是任何论文方法（AMO 的 PE composer 会按 K 重新生成系数；SAMuon 主基线是
Scion 而非本地 original+AdamW 混合配方）。按任务书命名规则，本轮全部候选为
"固定原系数 K=3/5/7 有限候选家族"，其阴性结论不外推到目标敏感谱分配全体。
