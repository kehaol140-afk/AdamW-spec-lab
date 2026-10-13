# 使用说明

本包是用户认可“有限NS更新的目标价值”研究定位后，针对R0/R1的执行交接材料。
它不是已运行结果，也不含服务器checkpoint、训练语料或生产代码。

建议给OpenCode的文件：整个ZIP即可。
读取顺序：
1. OPENCODE_R0_R1_TASK_CN.md
2. R0_R1_EXECUTION_PROTOCOL_v1_1_CN.md
3. r0_r1_contract.json
4. references中的历史草案与审计。

新协议明确覆盖旧草案中较窄的a/b网格、无统一边界扩展、少量检查角色与旧预算。
新实验先做R0，满足条件后同一任务继续R1；R2不在本轮执行范围。

新增设计数值属于本轮前瞻约定，不声称来自论文或旧预注册：
- a初始{0.5,0.75,1}；b初始{0.75,1,1.25}；
- 一次自动校准侧并集扩展，最多5×5×2主候选；
- 校准≤416个逻辑目标点、检查≤72点，合计≤488；
- R0+R1≤2 GPU-hours，新增输出≤2GiB。

在服务器运行前，执行者必须绑定真实路径、source SHA、checkpoint SHA、batch和评价清单。
本包contract的状态字段刻意保留TEMPLATE，不能把未绑定模板称为已完成服务器预注册。

文件完整性可用 MANIFEST_SHA256.json 核验。references字节保持与本次可读的既有文件一致。
