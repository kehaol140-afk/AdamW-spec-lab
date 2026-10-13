# R2F_REPORT_CN — LR–WD 归因与交叉续训结果

日期：2026-09-29。协议：`references_r2f/R2F_EXECUTION_PROTOCOL_v1_CN.md`。
复用门 PASS（6 状态证书、逐组配方忠实性证明、旧 4 终点复评 **diff=0.000**）；
10 条主分支全部完成（L×2 + 交叉×8 = 2560 次真实更新；**D 未触发**）；GPU ~18 分钟/预算 3 小时。

---

## 四问回答

### 1. 普通降 LR 解释了多少？——几乎全部

| 起点 | V_N | V_L（lr=0.5×, decay=0.5×） | V_AH（lr=0.5×, decay=1×） | B_L=V_N−V_L | 旧 B_AH | 解释比例 |
|---|---|---|---|---|---|---|
| 512 | 2.924853 | **2.850637** | 2.851879 | **+7.422e-2** | +7.297e-2 | **101.7%** |
| 1536 | 2.533048 | **2.526438** | 2.526469 | **+6.610e-3** | +6.579e-3 | **100.5%** |

- 普通"数据更新幅度减半"（等价于 LR 减半的数据项）**解释 AH 收益的 ~100%**；
  B_L 略大于 B_AH → decay 乘积减半有**轻微正贡献**（B_decay=V_L−V_AH：512
  **−1.24e-3**、1536 −3.17e-5，负=L 更好）。
- **D_TRIGGER 未触发**（两起点 B_decay 均为负、远低于 τ=1e-3），按预冻规则跳过
  D，无事后补做。
- 逐域：L 与 AH 的 A/B 域差异均 <2e-3（512 A: 2.7509 vs 2.7520；B: 2.9504 vs 2.9518）。

### 2. 早期 AH 领先在相同后续规则下保留多少？——不保留，且反转为劣势

交叉终点（总 512 步，Q_P2_endpoint，正=参照更差）：

| 起点 | J(N,N) | J(N,AH) | J(AH,N) | J(AH,AH) |
|---|---|---|---|---|
| 512 | 2.780697 | **2.721940** | 2.827313 | 2.748768 |
| 1536 | 2.493638 | **2.491751** | 2.502294 | 2.499720 |

- **R_N=J(N,N)−J(AH,N)：512 −4.66e-2、1536 −8.66e-3（两起点均为负）**——前 256 步
  AH 的领先（B0=+7.30e-2/+6.58e-3）在切回 N 后**反转为劣势**（R_N/B0=−0.64/−1.32）。
  优势不在"状态"里——每一步的配方才是收益来源。
- T_N/T_AH 全为正（512 +5.88e-2/+7.85e-2；1536 +1.89e-3/+2.57e-3）：无论前段，
  **后半程用 AH 都更好**；J(N,AH) 两起点全局最优——**晚切换已足够**。
- J(AH,N) 两起点全局最差：AH 状态配回原生步长存在切换代价（与 moments/有效步长
  失配一致的方向性解释；不据此断言唯一机制）。
- I_path=−1.98e-2/−6.86e-4（负交互：AH 前段与 AH 后段非可加）。

### 3. 结果是否随起点或域变化？——方向完全一致

- 两起点：归因方向（L≈AH>N）、交叉排序（N,AH 最优 / AH,N 最差）、D 不触发全部一致；
  幅度 512 ≈ 1536 的 10 倍（与前几轮的 LR 段效应一致，同父轨迹不外推为阶段规律）。
- 两域：L/AH 差异与交叉排序在 A/B 域**逐项同向**（见 lrwd_attribution/cross_path_effects）。

### 4. 哪些结论仍然不能作出

- 机制唯一归因：R_N 反转可能来自参数范数×有效 LR、moments 失配或切换暂态的混合；
  本设计不拆分 weights/momentum 贡献。
- 0.5 非最优倍率（无搜索）；W 配方（lr=1,wd×2）未做；"L 比 AH 略好"的 −1.2e-3
  是单配对点差，**不构成等价或优越的统计结论**。
- 同父 seed 两起点非独立 seed；无跨种子/模型/数据外推；无完整预训练收益；
  AMO/OrScale/SAMuon/DynMuon/Muon-SW/WSD-cooldown 均未比较。

## 工程验证与执行记录

- 首步门：两起点 N vs 独立原生参考 **maxdiff=0.0（bit 级）**；L/D 数据 delta 比
  0.5013/1.0013（含 decay 项混合，与代数预期一致——纯数据项精度已由 R2 轮 AH 的
  0.5000000 验证路径覆盖）；配方每步从不可变 base 重建（切换无累乘，单测覆盖
  AH→N 拒绝从 effective 值续乘）。
- 每 run 独立 train.jsonl（run_id/source_sha/global_step/config_sha 逐条）+
  COMPLETE 证书（可加载验证后原子替换）；断点续跑不覆盖其他 run。
- 旧 R2 N/AH 分支零重训复用为交叉来源（cursor 768/1792、各自 RNG 保留；
  同 origin 四条尾部 batch 流相同）。
- 评价锁先于全部 endpoint 评价；Q_P2_endpoint 历史开发暴露已声明；confirm 零访问；
  诊断快照（1/16/64）仅记录切换暂态。

## 交付（outputs/r2f_lrwd_retention_v1/）

asset_certificate.json、base_group_manifest.json、first_step_gates.json、D_TRIGGER.json
（含未触发原因）、evaluation_lock.json、10 个 run 的 train.jsonl 与 COMPLETE 证书、
endpoint_values.csv（逐域 sum/count 含诊断快照）、lrwd_attribution.csv、
cross_path_effects.csv、cost_ledger.json、本报告；测试 test_r2f.py 10 项全绿。

## 复跑

```bash
cd /home/b205/projects/training_dynamics_lab/muon_spectral_allocation
python tests/test_r2f.py
CUDA_VISIBLE_DEVICES=<free> python scripts/run_r2f_gate.py      # ~90s
CUDA_VISIBLE_DEVICES=<free> python scripts/run_r2f.py          # ~13min
CUDA_VISIBLE_DEVICES=<free> python scripts/analyze_r2f.py      # ~150s
```

**本轮停止**：不新增种子/窗口/预测器/方法。若继续该线，协议 §9 的下一候选是
带对称 LR 挑战的固定配方比较或预测阶段设计——由审阅决定。
