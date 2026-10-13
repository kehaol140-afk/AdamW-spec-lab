# V16_A1_SELECTION_AUDIT — a1_select 选优审计

- 对象：服务器 A1 36/36 完成后由冻结 planner `plan_a1_select` 自动执行的选择
- 审计方法：独立重算（不调用 planner 代码）逐族 dev H256 最小值 + 冻结平手规则（优先 (1,1)，次按 |a−1|+|b−1|，再按 id 字典序），与 STATE.selections.a1 比对

## 1. 结果：复核一致

| 族 | planner 选择（STATE.json） | 独立重算 | dev_H256 | 平手数 | 一致 |
|---|---|---|---|---|---|
| S | a1_S_a8_b2 (a=8,b=2) | a1_S_a8_b2 | 5.4342579979 | 1（无平手） | ✅ |
| F | a1_F_a8_b2 (a=8,b=2) | a1_F_a8_b2 | 5.4314878576 | 1 | ✅ |
| P | a1_P_a8_b2 (a=8,b=2) | a1_P_a8_b2 | 5.4314460957 | 1 | ✅ |

## 2. 选择上下文与风险标注

1. **RANGE_UNRESOLVED**：三族最优均在冻结网格角点 (a=8, b=2)。a=8 与 b=2 均为边界值 → 按冻结合同（`auto_expand_grid: false`、原任务书边界规则）不得声称"充分调优"；后续 B 阶段的 S8/F4-5/P4-5 边界扩展槽位正是协议预写的处理路径。
2. **选优依据隔离**：选择只读 development 集（256 文档，probe_select 角色）；screen 集留待 B 后评估；final_holdout 1024 文档保持封存（RC5 已验证三划分零交集，holdout 未开封）。
3. **数据完整性**：36 分支 × 2 视界 × 256 文档 = 18,432 评估行全部携带父 checkpoint sha + param_fingerprint + global_step；无 EVAL_WITHOUT_STATE_ORIGIN、无重复 run、无失败尝试计入成功（首分支一次中断按协议重放恢复，step 序列连续无重复）。
4. **下游已入队（未执行）**：a1_select 已按冻结规则提取 q_series 并入队 4 条 A2（a2_Fdir_qS / a2_Pdir_qS / a2_Sdir_qF / a2_Sdir_qP，均 256 步、共同目标范数 q_t 来自 A1 同系数运行）+ b_plan。本轮在 A2 训练开始 14 步处按边界停止，等待用户裁决。
5. **Windows 参照**：Windows 31 条（只读归档）独立选优 S/F 亦为 (8,2)（P 因缺格选 (2,2)）；两平台 31 共享格排序 Spearman 0.9992 → 服务器重放忠实，选择结论对平台稳健。
