# R2F_CORRECTED_REPORT_CN — CONFIRMED_WRONG_DECAY 修复与四条 AH 来源分支重跑

日期：2026-09-29。缺陷档案：`outputs/r2f_lrwd_retention_v1/defect_archive/`（错误版源码
`run_r2f.py.WRONG_DECAY_VERSION` + `source_diff.patch`，旧错误结果原样保留于
`outputs/r2f_lrwd_retention_v1/`，未覆盖任何旧 run/锁/表）。

## 1. 修复内容

`load_full` 不再从来源 checkpoint 的 `param_groups` 读取 base WD；新增
`immutable_base_wds()`：从 `base_group_manifest.json` 按优化器身份绑定（muon=0.02、
aux=0.02），并在每次绑定时对 parent512/1536 原件断言一致。来源 checkpoint 仅恢复
状态（model/buffers/moments/m/v/step/RNG/cursor）；LR 沿原时钟逐全局步计算。

## 2. 真实调用链测试（tests/test_r2f_fix.py，6 项全绿，GPU 真实 step）

真实 AH 来源 checkpoint + 真实生产 batch 一步：四配置 eff 值/decay 乘积恒等
（=base_lr×0.02）；AH 来源保存的 0.04 不污染 base；配方写回前后权重/moments/
m/v/step hash 不变；同梯度首步 N 与 AH 的 56 个 momentum buffer **bit 级一致**；
连续两步/save 无重复缩放；正确 WD 基点的数据 delta 比 AH/N=0.5（|偏差|<1e-3）；
payload 不可变、CPU step 无跨分支共享。

## 3. 修正 run 首末 LR/WD（逐步日志 256 步全程断言 decay_ratio=1.000000）

| run | 首步 LR / WD | 末步 LR / WD | decay 乘积倍率 |
|---|---|---|---|
| s512_AH_to_N_corrected | 0.00734440 / **0.02** | 0.00561219 / **0.02** | 1.000000 |
| s512_AH_to_AH_corrected | 0.00367220 / **0.04** | 0.00280609 / **0.04** | 1.000000 |
| s1536_AH_to_N_corrected | 0.00135064 / **0.02** | 0.00100000 / **0.02** | 1.000000 |
| s1536_AH_to_AH_corrected | 0.00067532 / **0.04** | 0.00050000 / **0.04** | 1.000000 |

## 4. 修正 2×2 终点与路径效应（Q_P2_endpoint，复用未受影响的 N 来源分支）

| origin | J(N,N) | J(N,AH) | J(AH,N)* | J(AH,AH)* | R_N | R_AH | T_N | T_AH | I_path |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 2.780697 | **2.721940** | 2.823305 | 2.746939 | −4.261e-2 | −2.500e-2 | +5.876e-2 | +7.637e-2 | −1.761e-2 |
| 1536 | 2.493638 | **2.491751** | 2.502246 | 2.499703 | −8.608e-3 | −7.952e-3 | +1.887e-3 | +2.543e-3 | −6.559e-4 |

（*=本轮修正值；逐域值见 endpoint_values_corrected.csv / cross_path_effects_corrected.csv，
A/B 域排序与总体一致。）

**旧错误 vs 修正**（wrong−corrected，正值=修正后更好）：
s512_AH_to_N +4.01e-3、s512_AH_to_AH +1.83e-3、s1536_AH_to_N +4.75e-5、
s1536_AH_to_AH +1.71e-5。修正量远小于效应量，**全部方向性结论保持**：
J(N,AH) 最优、J(AH,N) 最差、R_N/R_AH 仍负、T_N/T_AH 仍正、I_path 仍负。

## 5. 范数诊断（每 16 步，512 起点）

AH→N：muon 参数范数 222.6→237.4（增长 6.7%）、data/param 比 0.074→0.055；
AH→AH：范数 221.7→221.4（近持平）、data/param 0.038→0.030。AH→N 的更大步长
伴随参数范数增长——与"切回原生步长后有效 LR×范数失配"的方向一致；仅为观察，
不认定唯一机制。

## 6. 成本

GPU 321s（预算 3600s）：测试 45s + 四分支 216s + 评价 60s；主分支更新 1024；
失败尝试 0；新增输出 2.46GB/4GiB；confirm 零访问；旧 run/锁/表零覆盖。

## 7. 结论边界

成立：修复后 R2F 交叉结论（R_N<0 两起点、晚切换最优、I_path 负）在正确 decay
乘积下复现；归因结论（B_L≈100% 解释 AH）不受本轮影响（L 分支未被污染）。
不能作出：机制唯一归因、跨种子外推、最优倍率、W 配方、文献方法比较。
