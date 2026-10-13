# REPRODUCE — muon-ns-allocation

## 环境

torch 2.11.0+CUDA（重放需 GPU）；CPU 复算仅标准库。

## 入口与复算

- 只读复算：`python3 recompute/run_recompute.py`（CK MSA_attribution）：R2F 512 行 V_N/V_AH/V_L/B_L 与 **B_L/AH=101.70%** 从 `outputs/r2f_lrwd_retention_v1/lrwd_attribution.csv` 精确复算；`check_rows.csv`/`amplitude_controls.csv`/`paired_effects.csv`/`nl_effects.json` 直读可核（RESULTS R1-R5 全部表格值）。
- 重放（训练级）：`scripts/` 各轮 runner + 父状态恢复（需 GPU）。

## 重放资产（未随仓）

| 资产 | 身份 | 位置 |
|---|---|---|
| parent_state_step{512,1536,2048}.pt（各 266.5M） | 7563daec…/014480c7…（AH_ASSET_INDEX，抽查实算一致） | 服务器 outputs/p0_p1_v1_1/parent/；被 specmuon-pilot 只读依赖 |
| R2/R2F/R2F-NL 终态与快照（72 .pt） | AH_ASSET_INDEX + evaluation_lock 逐 SHA | 服务器；active≈5.3G 不可重建（原始证据），诊断快照≈5.4G 可重放 |
| V_Q 评测池 | muon_freeze/eval_pools（.pt） | 服务器 muon_transition outputs/m0 |
| Q 组划分 p0p1_q_sets.pt | 规则 sha256('P0P1_Q_v1\|gid')%3 | **已灭失待补建**（/tmp/opencode） |

## 不可复现项

- 一步检查与 256 步延续的**重放**需父状态+GPU（本仓只保结果数据）
- SOURCE_BLOCKED 三论文的方法对照永不重放（决议冻结）
- 结论分级：本仓可完整复算全部归因/筛查数值；训练级重放依赖上表服务器资产（有 SHA 锚定，未独立备份——见 docs/08）

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 | **PASS** | CK MSA_attribution（101.70% 精确）+全部 csv 直读 |
| R1 parent ckpt 重放 | **BLOCKED_ASSET** | parent 状态 11.65G（服务器，NOT_YET_BACKED_UP） |
| R2 从头训练 | **UNVERIFIED** | m1 配方自头重放从未执行（历史即恢复点设计） |

无便携包装器（多轮 runner 历史入口见 docs/TRAINING_ENTRYPOINTS.md）。
