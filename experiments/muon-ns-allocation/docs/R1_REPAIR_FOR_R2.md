# R1_REPAIR_FOR_R2 — 阶段 A 修复说明（含影响清单）

日期：2026-09-29。缺陷来源：references_r2/R0_R1_INDEPENDENT_ANALYSIS_CN.md §3。

## 1. 修复内容

| # | 缺陷 | 修复 | 验证 |
|---|---|---|---|
| 1 | `r1_probe.BatchContext.theta_plus/restore_theta` 只保存/恢复 Muon 参数——P 评估后辅助 AdamW 参数停留在更新前状态，扩展校准复用了错误 aux 基点 | theta_plus 改为**全 unique 参数**快照；restore/apply_candidate 全参数恢复 | 真实模型 N→P→N：N1==N2 **bit 级**、活模型全参数 hash==快照 hash；CPU 合成测试复现缺陷再验证修复（tests/test_r2_data_scale.py 6 项） |
| 2 | （既修沿用）AdamW step 别名 / payload 不可变 / 时钟写回 | load_anchor 深拷贝 + lr(s+1) 写回 | R0_GATE.json（上轮）+ 本轮 preflight |
| 3 | 数据缩放语义（R2 新增） | `src/r2_data_scale.py`：eff_lr=r×base_lr、eff_wd=base_wd/r（decay 乘积不变），每步从 base 时钟重算 | 单测：乘积不变、无累乘 0.5、r=1 原生等价、不含 WD 的整体减半 |
| 4 | 首步三臂语义（R2 新增） | — | **实测**：两恢复点 N vs 独立参考 maxdiff=0.0（bit 级）；MH/AH 数据 delta FP64 比率 muon=0.500000/aux=1.000000（MH）、muon=0.500000/aux=0.500000（AH） |

## 2. 补算与勘误（outputs/r0_r1_v1_1_corrected/）

- **176 条扩展校准补算**（修复后代码、原数据/manifest/目标/扩展清单）：旧 vs 新 V 差 mean −3.15e-4、max|d| 1.29e-3——错误 aux 基点的实际影响量级。
- **保留初始 161 行**：spot check（2 点×8 batch）与旧值 **0 差异**（初始网格在 P 之前运行，不受缺陷影响——独立审计 §3.3 判断获证实）。
- **corrected_selection_lock**：按原 near-best/tie 规则重放（初始保留 + 扩展更正）→ **与历史 lock 角色零变化**（G/A/S/S_at_A/A_at_S/G_at_A 全部相同；选择均落在初始网格点，扩展行变化不改变 near-best）→ **无需补任何检查点**（原 52 行检查的角色点全部仍有效）。
- 该 corrected lock 系看过旧检查结果后的勘误产物（文件内已注明），不是新预注册。

## 3. 对本轮 R2 的影响

按协议 §1：修复不改变三臂定义——本轮检验**事先指定的 0.5 倍率**，不称其为充分优化的最优值（corrected 数据同样不用于重选幅度；更正后的最优仍在 a=0.5 附近，与指定值一致属巧合性支持，不构成搜索）。

## 4. 运行事故与处置（完整保留）

1. 首次 R2 运行被**系统级 global OOM** 击杀（他人项目 `minimind-exp` 环境 `f3_5_mode_coupling.py` 峰值 123GB RSS 触发；本轮进程连带死亡；未触碰该进程）。
2. 第二次 resume 后 s1536_MH 完成、AH 启动时再次被杀；**s1536_MH_step256_full.pt 被截断损坏**（203MB vs 正常 253MB）。
3. 处置：脚本增加按文件存在断点续跑 + 损坏文件删除重跑（s1536_MH 重跑 88s 完成、加载验证 OK）；两次事故均记账于 cost_ledger。
4. run_log 曾因 "w" 模式重开清空一次（改 "a" 模式）；stdout.log 保留完整事故记录。
