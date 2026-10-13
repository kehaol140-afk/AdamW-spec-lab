# P0_P1_REPORT_CN — Muon 问题立项 P0/P1 v1.1 实测报告

日期：2026-09-29。协议：`docs/P0_P1_PROTOCOL_v1_1.md`；配置 `configs/p0_p1_v1_1.json`。
父轨迹 seed 3101（2048 步，loss 7.54→2.29，计数器 66,846,720 targets / 67,108,864
slots 断言通过；512/1536/2048 三状态保存）。Q_select（A676/B831 docs，各 128 窗）/
Q_P1_check（A498/B628，128 窗）/Q_P2_endpoint（256 窗，本轮零访问）组级隔离划分。
confirm 零访问。全部 14+14 项预写测试通过；e_repeat=0 → δ_step=1e-5 nats/target。

---

## 三行状态表

| 问题 | 状态 | 一句话证据 |
|---|---|---|
| **Q1** 同谱计算预算下检查目标一步收益？ | **NO_LOCAL_SUPPORT**（K=3/5/7 固定原系数家族、本设置） | 14 个单组变更×2 状态检查收益**全部为负**（0/4 同向，最大 |B|=3.8e-5）；同预算联合 10 池中 512 锁 no-op、1536 target(q3/o7) B=−1.17e-6（0/4）；事后最优候选也是负的——无可选择的空间 |
| **Q2** 逐组标量增益可否解释？几何错失多少？ | **SCALAR_EXPLANATION_NOT_RULED_OUT**（1536）；512 INSUFFICIENT（no-op 无从谈） | gamma_const 联合 B=−1.05e-6 ≈ target −1.17e-6（分配信息≈0）；**全局幅度×0.5 的 B=+3.4e-4（3/4 同向）比一切谱候选大一个量级**；几何准则联合 B=−3.8e-5（512，劣于 native 更多）——几何"错失"的收益不存在（为负） |
| **Q3** 本地 AMO 机制对照后是否仍有目标信息增量？ | **COMPARATOR_UNRESOLVED** | AMO v2（arXiv 2605.17806v2）网络不可达、无本地缓存、PE composer/系数无法可靠来源化 → 按协议不造数、不做同族比较；Q1/Q2 的 KJ 家族结果不依赖 AMO |

## 1. 实际执行 / 未执行

**执行**：P0 全部（配方逐字段冻结、别名/参考/缓存修复与验证、成本 FLOPs+24 交替块
计时）；一条父轨迹（2048 真实 updates）；2 状态 × 4 校准 batch × 45 KJ 探针点
（Q_select）；同预算池 10/42 对（解析 FLOPs=1.000、实测时间比 0.985–1.013）；
联合提案→实测→锁定（`selection_lock.json` 先于检查）；2 状态 × 4 检查 batch ×
（45 KJ 点 + 锁定方案）on Q_P1_check；三联幅度探针与全局 ×0.5/×2 对照；
e_repeat 双评价；独立二次采集确定性验证；三张图与全部表格。

**未执行**：P2/P3、Q_P2_endpoint 与 confirm 目标计算（0 次）、AMO/PE 同族比较
（来源不落实）、任何 LR/β/NS 系数/数据修改、更多 seed。

**技术修复（ROUND3 教训实测命中）**：`load_anchor` 深拷贝后 AdamW `step` 张量
仍 34/34 跨恢复共享 data_ptr（momentum/m/v 不共享）——修复后 0 共享、payload 不可变
（脚本验证）；参数参考在 load 后建立；实际消费 G 的 step 前后 hash 断言通过；
独立二次 fwd/bwd 采集一致。失败尝试 4 次（CSV 字段、成本解包、scheme 名、检查侧
补 geometry/random 对照）全部记账。

## 2. 组级敏感性（目标—几何—成本表摘要；全表 group_probe_table.csv）

检查侧 raw 单组收益 B_g（4 batch 均值，同向数）——**全部为负**：

| 状态 | 最差（|B| 最大） | 次差 | 最好（仍为负） |
|---|---|---|---|
| 512 | down K3 −3.78e-5 (0/4) | v K3 −2.84e-5 (0/4) | … o K3 −1.72e-5 (0/4) |
| 1536 | v K7 −2.68e-6 (0/4) | o K3 −2.32e-6 (0/4) | … down K3 −1.29e-6 (1/4) |

- 512（LR 高段）对谱步数变更的敏感度比 1536 大一个量级，但**方向全部为负**：
  增减任何一组的 NS 次数都损害该步目标。E_g²(K=3)>E_g²(K=5)>E_g²(K=7)（几何随 K
  单调改善）与目标方向**相反**——几何改善不带来目标收益，反而伴随损害。
- γ_g(K)（K3 相对 K5 的组更新幅度比）全部接近 1（约 0.98–1.02，详表），
  谱步数对更新幅度影响很小，损害来自方向/内容而非幅度。

## 3. 同预算联合比较与交互余项（joint_allocation_table.csv）

- 512：两提案联合校准 3.164058 vs native 3.164050（不优）→ **锁 no-op**（检查
  B≡0）。ΣB_g 校准提案值与联合实测的交互余项 <1e-6（数据见 selection_lock）。
- 1536：target (q3|o7) 校准联合 −native = +2e-6（勉强提案），检查 **−1.17e-6（0/4）**
  ——校准微弱信号不迁移；与 ΣB_g 之和的交互余项 ~1e-6 量级（无可辨识结构）。
- 几何对照（512 down3/gate7）检查 B=−3.76e-5；fixed_random 同对或类似，全部 ≤0。

## 4. 幅度对照（amplitude_controls.csv）

| 对照 | 512 | 1536 |
|---|---|---|
| 全局数据更新 ×0.5 | **+3.42e-4（3/4）** | +5.75e-5（3/4） |
| 全局数据更新 ×2.0 | −5.12e-3（0/4） | −4.60e-4（0/4） |

- ×0.5 的正向一步收益超过所有谱分配候选 |B| 一个数量级——**当前 LR 下一步更新
  幅度过冲**，最便宜的标量干预远强于任何 K 重分配。两个幅度点不是 LR 搜索
  （按协议只作对照，不外推最优幅度）。
- target vs gamma_const 配对差（1536）：−1.17e-6 vs −1.05e-6，差 ~1.2e-7，
  远低于 δ_step——**逐组 NS 分配不含超出冻结组级标量的信息**。

## 5. AMO（amo_comparison.csv）

`本地original配方下的AMO机制一步对照`：**COMPARATOR_UNRESOLVED**（论文不可达、
无本地实现、PE 系数无来源——详见 RECIPE_AND_AMO_COMPARISON.md §2）。未生成任何
AMO 数字；"KJ 目标方案 vs PE-AMO"的准则差异归因在本轮**无数据可谈**。

## 6. 评价暴露 / 随机性 / 成本限制

- Q_select/Q_P1_check 组级互斥（翻译线程），但旧 dev 曾用于 M1/M2 调参——本轮是
  **开发级**结果；一个父轨迹的两个恢复点不是两个 seed；4 batch/侧不做种子级显著性。
- 成本匹配：解析 FLOPs 与实测时间双卡尺均 ∈[0.98,1.02]（10 对池，匹配先于检查锁定）。
  谱模块占完整训练时间比例未测（不影响本轮相对比较）。
- 预算：GPU 1512s（0.42h/3h）；输出 800MB/4GiB；失败尝试 4 次已记账。

## 7. 结论与建议

**阴性质裁（限固定原系数 K=3/5/7 家族 + 本设置）**：该家族在同谱预算下没有任何
可分辨的一步目标收益（连事后池最优都为负），且便宜的组级/全局标量增益解释不可
排除、实际更强。按任务书：**停止该有限候选家族**；不否定所有目标敏感谱分配
（AMO 的 PE 系数族、按 K 再生成系数的方案、其他 K 值未测且 Q3 未回答）。

**值得准备 P2 的证据：不存在**（Q1 未通过）。若用户仍要继续该问题线，下一步不是
P2，而是先解决：(a) AMO 来源获取后做真正的同族比较（Q3 现为 UNRESOLVED）；
(b) 解释 ×0.5 幅度正收益（LR 过冲）——那是 LR 标定问题，不是谱分配问题。

## 复跑

```bash
cd /home/b205/projects/training_dynamics_lab/muon_spectral_allocation
python tests/test_state_and_routing.py && python tests/test_group_probe.py
CUDA_VISIBLE_DEVICES=<free> python scripts/run_parent.py            # ~16 min
CUDA_VISIBLE_DEVICES=<free> python scripts/run_p1_calibration.py   # ~2 min
CUDA_VISIBLE_DEVICES=<free> python scripts/run_cost_benchmark.py   # ~30 s
CUDA_VISIBLE_DEVICES=<free> python scripts/run_p1_joint_and_lock.py  # ~30 s
CUDA_VISIBLE_DEVICES=<free> python scripts/run_p1_check.py         # ~70 s
python scripts/analyze_p0p1.py
```
