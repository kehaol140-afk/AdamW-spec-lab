# REPRODUCE — specmuon-pilot



- 只读复算：`python3 recompute/run_recompute.py`（CK SPECMUON_P1）：14 行全字段直读核对（RESULTS R3 表）；逐分支明细在 `results/p1/branches/`
- 环境复算：纯标准库 CPU；重放（训练级）需 GPU+父态
- 重放资产（未随仓）：parent_state_step{512,1536}.pt（266.5M×2，muon-ns-allocation 仓 SHA 锚定）；V_Q 评测池（.pt，muon_transition m0）；/tmp/opencode/mte_domain_train_windows.pt（M0 域缓存 1.39G，可 109s 重建）
- 身份哈希：P1_EVIDENCE_BUNDLE（全重复省略，成员映射 docs/DUPLICATE_ELISION_AUDIT.csv）；P0 zip 保留（3 唯一成员）
- **不可复现项**：P1 分支 checkpoint 已按协议删除（成功后 prune，可由父态+冻结 runner 重放再生）；P0-B 依赖 v07 bank（服务器 4.6G 未独立备份）
- 分级声明：结果可复算=完全（本仓数据齐全）；训练可重跑=是（父态在服务器时）。

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 | **PASS** | CK SPECMUON_P1（14 行全表） |
| R1 | **BLOCKED_ASSET** | MSA parent(7563daec/014480c7)+V_Q 池+muon_transition m0；`portable/run_specmuon_p1.py` |
| R2 | **BLOCKED_ASSET** | 同上+GPU；P1 分支 ckpt 已按协议 prune（可重放再生） |
