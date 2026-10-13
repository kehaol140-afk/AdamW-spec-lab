# REPORT_R0_R1_CN — 有限NS更新的目标价值（R0+R1 v1.1）

日期：2026-09-29。协议：`references_r0r1/R0_R1_EXECUTION_PROTOCOL_v1_1_CN.md`。
R0 GATE PASS（`outputs/r0_r1_v1_1/R0_GATE.json`）；R1 单进程完成
（校准 337 行含 1 次 native 重复、检查 52 行、e_repeat=0、GPU ~4 分钟/预算 2 小时）。

---

## 三行状态表（先数值，后解释）

| 问题 | 状态 | 关键证据（检查侧，4 batch 配对） |
|---|---|---|
| **统一时钟后幅度收益是否存在** | **存在且增强（两状态过线）** | V(N)−V(G)：512 **+9.96e-4**（3/4，A 域 +7.2e-4/B 域 +1.27e-3，CI[+2.6e-4,+1.7e-3]）；1536 **+1.30e-4**（4/4，CI[+9.5e-5,+1.7e-4]）。G=a0.5（O 数据更新全体减半，WD/aux 不动） |
| **O 组倍率是否超过全局幅度** | **否** | 512：V(G)−V(A)=−7.9e-6（2/4，微负不达线，A 域正/B 域负）；1536：A 与 G 同方案去重（b 调整不改变最优）——组相对倍率无增量 |
| **等范数 NS3 结构是否有独立增量** | **否** | 匹配结构增量 V(A)−V(S_at_A)：512 **−3.79e-5**（1/4）、1536 **−5.68e-6**（0/4）；等搜索机会 1536 仅 +8.1e-7（3/4，≪1e-5）；反向匹配两状态均负。等范数 realized ≤2.2e-7 已验证 |

**本轮裁决：`GLOBAL_ONLY_LOCAL_SIGNAL`**（幅度范围已解决：`AMPLITUDE_RANGE_UNRESOLVED=false`——扩展后 A_final={0.25,0.5,0.75,1.0}、B_final={0.5,0.75,1.0,1.25,1.5}，全部家族最优在网格内部）。

## 1. 数值正确性（R0）

- **时钟闭环**：恢复组 LR=lr(s)（0.008799/0.002358），写回 lr(s+1)（0.008794/0.002353）至 Muon+aux 全部组；统一时钟下原生 step 与手动组装一致（≤3e-8）；**旧轮 lr(s)/lr(s+1) 混用已证实并登记**（`R0_ERRATUM_AND_IDENTITY.md` §1，旧结论维持 PARTIALLY_VERIFIED，不倒签）。
- 别名/A→B→A/grad-hash/参考参数 load 后/payload 不可变：全部通过。
- **等范数**：O 组整体 FP64 归约，c=1.24–1.33 逐 batch 实测；realized data-delta 相对差 ≤2.2e-7（门槛 1e-4），逐点保存于 `norm_match_checks.csv`。
- V 从逐域 sum/count 独立重算：389 行 max diff=0；每 batch 65,280 有效 targets（32,640/域）。
- e_repeat=0：仅表数值重复性，**不代表采样不确定性为零**。

## 2. 幅度范围是否解决

**在本轮网格内已解决**。初始校准触边（a_low/b_low/b_high 并集）→ 按 extension_lock 一次性补齐 0.25/0.5/1.5 后，G/A/S 家族选择均落在网格内部（G=a0.5、A=a0.5/b1.25 或 b1.0、S=a0.5/b0.75–1.25）。不再扩点。注意两点诚实限定：(a) a 只缩放 Muon 数据更新，**不是完整 LR 搜索**；(b) 诊断参考 Z（Muon 数据更新置零）在 1536 为 +1.48e-4（4/4）——单步目标的最优幅度在 0 与 0.5 之间，这是一步目标的近视局限（协议 §4.4 明确不据此称零学习算法或继续向 0.125 搜索）。

## 3. 检查目标增量（分比较，均值 nats，正=参照更差候选更好）

| 比较 | 512 | 1536 |
|---|---|---|
| 1 全局收益 V(N)−V(G) | **+9.96e-4** (3/4) | **+1.30e-4** (4/4) |
| 2 组增量 V(G)−V(A) | −7.9e-6 (2/4) | 去重（A≡G） |
| 3 组纯配对 V(G_at_A)−V(A) | −7.9e-6 (2/4) | 去重 |
| 4 匹配结构 V(A)−V(S_at_A) | −3.79e-5 (1/4) | −5.68e-6 (0/4) |
| 5 等搜索结构 V(A)−V(S) | −3.79e-5 (1/4) | +8.1e-7 (3/4) |
| 6 反向匹配 V(A_at_S)−V(S) | −3.79e-5 (1/4) | −4.05e-6 (0/4) |
| Z 诊断 V(N)−V(Z) | +1.72e-4 (2/4) | +1.48e-4 (4/4) |
| P 诊断 V(N)−V(P) | −1.47e-3 (1/4) | −1.99e-4 (0/4) |

（逐域分项与 bootstrap 探索性 CI 见 `paired_effects.csv`；CI 条件于当前父轨迹与固定评价池，n=4 无覆盖率保证。）域分解值得注意：组/结构比较中 512 的 A 域常为正、B 域常为负——**逐域反向**，总体等权后相互抵消，不构成域级方法线索（仅登记）。

## 4. 证据级别与哪些比较未完成

- 两状态同父 seed3101（**非两个独立训练 seed**）；4 检查 batch 为配对样本非独立训练；1e-5 为本轮局部投入线、1e-6 为数值分辨/tie 容差——均非置信区间或普适门槛。
- **未完成的比较**：AMO/OrScale/SAMuon/DynMuon 全部未复现（Q3 之外的文献对照保持 COMPARATOR_UNRESOLVED/未比较，不声称已超越）；仅 O 组被测（其他组未提名）；NS3 之外的结构（其他 K/系数族）未测；持续终点（R2）未运行。
- OrScale 未复现——本轮"组无增量"不构成对逐层 trust-ratio 方法的否定。

## 5. 决策

按协议 §6：**GLOBAL_ONLY_LOCAL_SIGNAL → 先视为基线校准，停止当前新谱方法立项**。全局幅度信号（512 +1.0e-3、1536 +1.3e-4，方向=数据更新减半有益）应优先并入未来任何强基线的幅度/LR 处理；O 组固定倍率与 NS3 等范数结构在本轮家族内**无信号，停止该家族**（不否定其他系数族、其他 operator 或所有目标敏感谱方法）。若审阅后考虑 R2（256 步持续干预），按协议 §10 仅限已锁定的全局幅度方向（如 a=0.5 的 Muon-only 数据缩放）并如实计费——本轮一步收益**不乘 256 预测终点**。

## 交付物（outputs/r0_r1_v1_1/）

R0_GATE.json、R0_ERRATUM_AND_IDENTITY.md、calibration.csv、extension_lock.json、
selection_lock.json、check_values.csv、paired_effects.csv、boundary_status.json、
norm_match_checks.csv、batch_meta.json、verdict.json、run_meta.json、cost_ledger.json、
测试（tests/test_r1.py 7 项+旧 21 项全绿）、源码（src/r1_probe.py、src/r1_selection.py、
scripts/run_r0_gate.py、run_r1.py、r1_analyze.py）、R0_R1_REVIEW_BUNDLE.zip。

## 复跑

```bash
cd /home/b205/projects/training_dynamics_lab/muon_spectral_allocation
python tests/test_r1.py
CUDA_VISIBLE_DEVICES=<free> python scripts/run_r0_gate.py    # ~2 min
CUDA_VISIBLE_DEVICES=<free> python scripts/run_r1.py         # ~2 min
python scripts/r1_analyze.py
```
