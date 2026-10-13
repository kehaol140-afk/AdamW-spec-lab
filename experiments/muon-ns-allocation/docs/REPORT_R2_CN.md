# REPORT_R2_CN — 三臂 256 步真实续训结果（R1 最小勘误后）

日期：2026-09-29。协议：`references_r2/R2_EXECUTION_PROTOCOL_CN.md`；修复见
`docs/R1_REPAIR_FOR_R2.md`。六主 run + 1 native 重复全部完成（1536+256=1792 次真实
updates；两恢复点来自**同一父 seed3101**，非两个独立 seed）。

---

## 四问总答

**1. 一步缩幅收益是否延续到 256 步？——是，且大幅放大。**
Q_P2_endpoint 终点（step256，参照减候选，正=候选好）：

| 恢复点 | G_MH=V_N−V_MH | G_AH=V_N−V_AH | 参考线 |
|---|---|---|---|
| 512 | **+3.83e-2**（V 2.9249→2.8866） | **+7.30e-2**（→2.8519） | 1e-3 |
| 1536 | **+3.13e-3**（2.5330→2.5299） | **+6.58e-3**（→2.5265） | 1e-3 |

四组比较全部超过 1e-3 参考线（512 达 31–73 倍）；A/B 两域**全部同向**（无域权衡）。
收益不是一步收益 ×256 的线性外推：512 的一步 G_MH=−1.5e-5（**负**）而 256 步 +3.8e-2
——一步符号不能预测有限预算终点（协议 §8 预见分支）。

**2. 只缩 Muon 与主辅同缩哪种更好？——主辅一起缩（AH）更好，两状态一致。**
G_ratio=V_AH−V_MH（正=MH 好）：512 **−3.47e-2**、1536 **−3.45e-3**——均为负且 A/B 域
一致 → AH 优于 MH。即：数据更新缩幅时，辅助 AdamW 一起缩幅在同一设置下更优
（三个臂不足以估计完整二维交互；不据此提出唯一机制）。

**3. 两恢复点、两数据域是否一致？——方向完全一致。**
两恢复点、两域、两候选的全部 8 组终点比较同号；幅度上 512 > 1536 约 10 倍
（512 恢复点续训段 LR 处 cosine 高段 0.0088→0.0062，1536 段 0.0024→0.0015——
高 LR 段缩幅收益更大，与"幅度过冲"解释同向；属同一父轨迹的阶段差异，不外推为
普遍阶段规律）。

**4. 技术与证据限制。**
- 单父 seed 两恢复点、每臂单 run：无训练种子层变异估计；native 重复 bit 级一致
  （V=2.924853 与首条 N 完全相同）只证明**数值重复性**，非采样不确定性为零。
- 0.5 为事先指定值非搜索所得；LR/幅度未重新调优（不称 0.5 为最优倍率；
  "AH 更好"亦限于 r∈{1,0.5} 两点）。
- 一步→256 步预测失效实例（512 MH）明确在案；不把 256 步收益外推到完整预训练。
- AMO/OrScale/SAMuon/DynMuon 未复现、未比较（Q3/文献对照仍 UNRESOLVED）。
- 运行事故：外部进程（minimind-exp 环境，~123GB RSS）两次触发 global OOM 击杀本轮
  进程并截断一个 checkpoint（已删除重跑、加载验证 OK）；全程未触碰该进程。

## 逐步数据（endpoint_values.csv / paired_effects.csv）

**step1（一步）**：G_MH/G_AH 量级 1e-4~2e-4，512 MH 为负（−1.5e-5）；一步与 256 步
排序在 512 的 MH 上不一致——一步筛查（R1 结论 GLOBAL_ONLY_LOCAL_SIGNAL）不能替代
有限续训检验。

**monitor（Q_select，训练中诊断，未用于任何决策）**：512 V256: N 2.9495 / MH 2.9164 /
AH 2.8844；1536: N 2.5643 / MH 2.5611 / AH 2.5584——与 endpoint 排序一致（AH<MH<N）。

## 工程验证（preflight_*.json）

两恢复点：N 首步 vs 独立原生参考 **maxdiff=0.0（bit 级）**；MH 数据 delta 比
muon=0.500000/aux=1.000000；AH muon=0.500000/aux=0.500000（FP64，~1e-8 精度）；
decay 乘积 eff_lr×eff_wd=base_lr×base_wd 恒等（每步从 base 时钟重算，无累乘）；
momentum/m/v/step 按未缩放剪裁梯度原生递推；tied embedding 单对象单组。

## 结论与边界

- 256 步开发性续训**支持**"Muon+辅助数据更新整体缩幅 0.5"在该设置下优于原生续训
  （两恢复点、两域一致，超过 1e-3 参考线）；主辅同缩优于只缩 Muon。
- 这是**开发性**结果：同父 seed、固定 0.5、无 LR 对称挑战、未与外部方法比较。
  按协议不自动扩大：不新增倍率、种子、窗口或优化器；是否进入带对称 LR 挑战的
  下一阶段由用户与审阅者决定。
- R1 阶段的 O/NS3 阴性与本轮结果相互独立；本轮不为谱结构方法翻案。

## 交付物（outputs/r2_amplitude_continuation_v1/ + r0_r1_v1_1_corrected/）

run_index.json、training_metrics.csv（1536+256 步 base/effective LR/WD/preclip/loss）、
monitor.csv、preflight_s{512,1536}.json、14 个 checkpoint（服务器留存）、
endpoint_lock.json（首次 P2_endpoint 访问前锁定）、endpoint_values.csv（逐域 sum/count）、
paired_effects.csv、cost_ledger.json（含 OOM 事故）、R1_REPAIR_FOR_R2.md、本报告；
测试：test_r2_data_scale.py 6 项 + 旧套件全绿。

## 复跑

```bash
cd /home/b205/projects/training_dynamics_lab/muon_spectral_allocation
python tests/test_r2_data_scale.py
CUDA_VISIBLE_DEVICES=<free> python scripts/r1_repair_recompute.py     # 阶段A ~70s
CUDA_VISIBLE_DEVICES=<free> python scripts/run_r2_amplitude.py        # 断点续跑 ~6min
CUDA_VISIBLE_DEVICES=<free> python scripts/analyze_r2_amplitude.py    # endpoint ~90s
```
