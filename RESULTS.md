# RESULTS — 总表（正/负/不确定/未执行）

| 线 | 实验 | 裁决 | 证据级 | 详见 |
|---|---|---|---|---|
| AdamW | V13 四状态测量 | 完成 | 测量 | experiments/adamw-spectral-v13-v16/RESULTS.md |
| AdamW | V14 rank-1 谱修正 | **阴性**（SCALE_RANGE_UNRESOLVED） | confirm 128+20 | 同上 |
| AdamW | V15 方向-视界 | 单 seed 信号 | dev H64 | 同上 |
| AdamW | V16-A1/A2 | RANGE_UNRESOLVED / **方向归因成立（≥96% 保留）** | dev 单 seed | 同上 |
| AdamW | V16-B/C | **F_OR_P_SUSTAINED / PRELIMINARY_SEED_ROBUST**（+0.036~+0.151, 3/3 seed） | C=holdout 3-seed | 同上 + V16_FINAL_RECONCILED.md |
| AdamW | Direction Layer-1 | **方向必要**（F8−S8=−0.204, P8−S8=−0.235, CI 全负；纯幅度 +0.082 有害） | dev 单 seed | experiments/adamw-direction-validation/ |
| AdamW | 30M T1 | **DIRECTION_GAIN 迁移成立但量级收缩**（−0.038/−0.046）；**不敌 Muon（−0.2725）** | dev 单 seed | experiments/adamw-direction-30m/ |
| Muon | NS allocation P0/P1 | **阴性**（0/28；×0.5 +3.4e-4 唯一强信号） | 筛查 | experiments/muon-ns-allocation/ |
| Muon | R0/R1 | 全局信号 +9.96e-4，局部无 | 配对 CI | 同上 |
| Muon | R2/R2F/R2F-NL | **阴性关闭**（B_L 解释 101.7%；AH 尾部反转） | 256 步 + 归因 | 同上 |
| Muon | SpecMuon P0 | H1/H2 证实；H4-CV 证伪 | 测量 | experiments/specmuon-pilot/ |
| Muon | SpecMuon P1 Head/Bulk | **阴性关闭**（NO_EFFECT / SINGLE_POINT） | 双恢复点 | 同上 |
| Muon | 100M Context-Aware | **PILOT_FAIL — NO_GAIN_OVER_STRONG_MUON**（+0.1367/+0.0997；RANGE_UNRESOLVED×3；tokens 不足 NLL 上升） | dev 15 臂 | experiments/context-aware-muon-100m/ |
| Muon | NorMuon | STAGED（零 GPU，等价性未验证） | 设计 | 同上 §7 |
| 基线 | Hyperball A1 | 范数保持无终点收益（Δ+1.5e-4） | 单 seed | baselines/hyperball-a1-evidence/ |

**未执行（不写成结果）**：Direction layer-2 / 30M T2 / holdout_v2 开封 / P1 跨 seed / a>8 / 100M 确认层 / NorMuon 消融 / ITF-4/5。
**旧仓已归档五线**：FR/GTTR、continuation（STOP）、optimizer_sensitivity（F0-F4C）、muon_transition（R=0）、muon_temporal（T0/T2a/T2b）——见 docs/RELATED_ARCHIVES.md。
