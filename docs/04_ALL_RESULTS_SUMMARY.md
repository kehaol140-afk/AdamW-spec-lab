# 04_ALL_RESULTS_SUMMARY — 全结果汇总（正/负/不确定/未执行）

完整逐轮表格在各线 [RESULTS.md](../experiments/)；本页为跨线结论汇总。**裁决粒度=阶段级；全部 CI=doc 级 bootstrap 除注明 3-seed。**

## AdamW 线

| 阶段 | 裁决 | 关键数值 | 等级 |
|---|---|---|---|
| V14 | 阴性 SCALE_RANGE_UNRESOLVED | 谱vs标量 2.41e-6 CI 跨零 | confirm |
| V15 | 单 seed 信号 | F Δ_H64 +0.0014509 CI[+0.00089,+0.00203] | dev 单 seed |
| V16-A1 | RANGE_UNRESOLVED | 三族最优 (8,2) 角点；b 0.11≫方向 0.003 | dev 单 seed |
| V16-A2 | **方向归因成立** | F@qS−S=−0.00266 CI 全负；保留 ≥96%；纯幅度轨迹 1.7e-4 | dev 单 seed |
| V16-B | **F_OR_P_SUSTAINED** | P1 4.087912/F1 4.118871 vs S1 4.240565；三竞争解释排除 | dev 单 seed |
| V16-C | **PRELIMINARY_SEED_ROBUST / COMPLETE_WITH_SIGNAL** | S−F=+0.0363/+0.0940/+0.1515（3/3 同向，CI97.5 全离 0） | **3-seed holdout** |
| DV Layer-1 | **方向必要** | F8−S8=−0.2041、P8−S8=−0.2351 CI 全负；S8−S1=+0.0824 有害；复刻门 3/3 位级 | dev 单 seed |
| 30M T1 | **DIRECTION_GAIN 但量级收缩** | P2−S2=−0.0457、P2−S1=−0.0140；Muon−P2=**−0.2725** | dev 单 seed |

## Muon 线（全阴性）

| 阶段 | 裁决 | 关键数值 |
|---|---|---|
| MSA P0/P1 | 重分配 0/28 | 唯一强信号=全局×0.5 +3.42e-4（幅度类） |
| MSA R0/R1 | GLOBAL_ONLY | V(N)−V(G) +9.96e-4/+1.30e-4；组结构零增量 |
| MSA R2 | 符号翻转教训 | 一步 −1.46e-5→256 步 +0.0383 |
| MSA R2F | **AH 关闭** | B_L 解释 101.7%/100.5%；R_N<0 尾部反转 |
| MSA R2F-NL | 无实用增量 | \|Δtail\|≤8e-4<τ |
| SM P0 | H1/H2✓ H4-CV✗ | top-8 份额 21×；CV 0.227>0.2 |
| SM P1 | **NO_EFFECT / SINGLE_POINT** | Head/Bulk 双恢复点不过 2e-3 屏；lr0.5 −0.0740/−0.0079 |
| CAM 100M | **PILOT_FAIL — NO_GAIN_OVER_STRONG_MUON** | C_A−锚 +0.1367 CI[+0.122,+0.152]；C_B−C_A +0.0997；RANGE×3；14/15 单调抬升 |

## 基线与外部复现

Hyperball A1：Muon 3.27906 vs MuonH 3.27921（Δ+1.5e-4 噪声内；范数约束生效 261.928 恒定）——阴性。

## 未执行（引用禁止清单）

DV Layer-2 / 30M T2 / holdout_v2 开封 / P1(a=8) 跨 seed / a>8 / 多块联合 / 100M 确认层 / NorMuon GPU 消融 / ITF-4/5。详见各 EXPERIMENTS.md「未执行」节。
