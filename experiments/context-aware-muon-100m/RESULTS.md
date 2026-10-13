# RESULTS — context-aware-muon-100m（按时间顺序）

## R1 前置（10-12 午前）

preflight 100 步 + smoke（`results/preflight_100steps.json`、`smoke_100m.json`）PASS；code freeze（CODE_FREEZE_V1.json）+ 25 项 CPU 测试全 PASS（tests_freeze_state.log）。

## R2 15 臂 × 三视界全表（PILOT_100M_RESULTS.csv；H4096 由 recompute CK CAM_100M 逐文档复算 0 偏差）

| 臂（按 H4096 升序） | H1024 | H2048 | H4096 | wall(s) |
|---|---:|---:|---:|---:|
| B_adamw_lr3e-4 | 6.0860 | 6.2615 | **6.284614**（全场最优，网格下缘） | 918 |
| **B_muon_rms_lr1e-3（强 Muon 锚）** | 6.1453 | 6.3615 | **6.423106**（网格下缘） | — |
| B_adamw_lr1e-3 | 6.1889 | 6.2802 | 6.461258 | — |
| B_muon_lr5e-3 | 6.1860 | 6.4267 | 6.522696 | — |
| B_muon_rms_lr3e-3 | 6.2562 | 6.4794 | 6.529254 | — |
| B_lowmu_0p9 | 6.1543 | 6.5274 | 6.550261 | — |
| **C_A_lam0p85（最佳候选）** | 6.1744 | 6.5859 | **6.559776** | — |
| B_adamw_lr3e-3 | 6.2180 | 6.4339 | 6.563847 | — |
| B_muon_lr1e-2 | 6.2789 | 6.4757 | 6.564633 | — |
| C_A_lam0p65 | 6.1968 | 6.5280 | 6.566722 | — |
| C_A_nat | 6.1781 | 6.5576 | 6.567352 | — |
| B_muon_rms_lr1e-2 | 6.2807 | 6.3262 | 6.568473 | — |
| B_lowmu_0p8 | 6.1288 | 6.5654 | 6.584098 | — |
| B_muon_lr2e-2 | 6.3042 | 6.5896 | 6.598917 | — |
| C_B | 6.1891 | 6.6521 | **6.659516**（垫底） | — |

## R3 门与配对（analysis.json:111-118 + gate_analysis.json；recompute 复核）

| 判定 | 数值 | CI95 |
|---|---:|---|
| C_A_lam0p85 − 强 Muon 锚 | **+0.1367** | [+0.122, +0.152] |
| C_B − C_A | **+0.0997** | [+0.0820, +0.1164] |
| B_lowmu_0p9 − 锚 | +0.127 | [+0.113, +0.142]（排除动量解释） |
| RANGE_UNRESOLVED ×3 | B_adamw(3e-4)/B_muon(5e-3)/B_muon_rms(1e-3) 全在网格低缘 | selection.json:8 |

## R4 裁决（PILOT_100M_DECISION.md:9-15,44-51）

**PILOT_FAIL — NO_GAIN_OVER_STRONG_MUON**；momentum_explained=false；C_B_beats_C_A=false；按冻结边界规则不扩网格（阴性无抢救意义）；tokens 不足：14/15 臂 H1024→H2048→H4096 严格单调抬升（C_A_lam0p85 H2048→H4096 微降 0.026，仍远高于 H1024）；tokens-to-threshold 全臂退化。**确认层不开、多 seed 不做、第四阶段不启动**。30M/100M 优化器位次反转已显式登记（:49，tokens/参数配比差异，非矛盾）。

## R5 NorMuon staging（零 GPU，非结果）

18/18 CPU PASS + 动量独立恒等；G-2~G-5 阻塞；v1.1 提案待裁决——**无任何 GPU 方法结论**（引用时必须注明 STAGED）。

## R6 证据等级与原始数据

单 seed 3471、dev 级三视界、无 holdout（pilot 协议设计）；CI=doc 级 bootstrap。原始：`results/pilot/<arm>/{metrics.jsonl 4096 行, eval_docs_H4096.jsonl 256 行, run.json, run.lock.json}`×15 + analysis/selection/gate + DRIVER 台账；15×422M endpoint_weights 留服务器（可 seed+缓存确定性重建）；证据包含 47 唯一成员保留。
