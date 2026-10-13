# 给 OpenCode 的监督运行任务

你是这轮本地实验的审核与运行代理，不是科学结论的保证者。请完整读取 `PROTOCOL_LOCAL_V1.md`、`config/local_v1.json`、`RUN_WINDOWS.md`、`reports/CODEX_HANDOFF.md`，并实际检查实现和数值日志。不能只采信 Codex 的“完成”文字。

工作区 `E:\大模型预训练\adamw_spectral_pilot`；原 MiniMind、旧实验与原数据只读。当前不连接服务器。使用绑定文件中的 Python，不自动换环境或下载依赖。最多一个训练进程；先检查本项目运行锁与 PID，不能再启动重复任务，不能关闭其他用户进程。

## 你的权限

可串行执行：preflight、prepare、数值测试、必要 smoke、discover；只有 discover gate 合格才执行 confirm；随后 report 与轻量 bundle。发现过程最多三条 seed3407 的512步基线，确认最多一条 seed3408 的512步基线。不得自动加 seed、扩大 LR/rho/rank 网格、增加训练步数、启动在线新优化器或完整竞争实验。

P0失败、数值异常、OOM、资源不足、数据角色重叠、checkpoint不一致、搜索范围不足都应停止后续阶段。NO_LOCAL_SIGNAL、NO_TRANSFER 不得当作软件 bug 来“修正”。只有非科学性的路径/日志/中断问题可记录后最多重试两次；改变运算逻辑应交回 Codex，并重新执行相关 P0，不能沿用受影响结果。

## 启动前审核

实际核对至少这些点：

1. A 使用累计并归一化、裁剪后的真实梯度与 next moments；epsilon、bias correction、per-parameter step、共享权重与 torch AdamW 对齐。
2. s 不改变 eta*wd；目标外参数为相同下一步；rho=1/s=1 能复现 baseline 权重和 loss。
3. probe 参数/梯度/moments/RNG/数据游标恢复，两个 gradient-build replicate 不串状态。
4. 文档级分区无已消费内容重叠，token mean/shift/mask正确，独立确认不参与选择。
5. 主 cell、网格、阈值已冻结，试验状态可区分 NOT_RUN、FAIL、NO_SIGNAL、UNRESOLVED 与 GO。
6. 代码只以逐候选单模型运行；显存与主存有实测上限，数据不整包载入；没有自动大模型下载、wandb或compile。

生成 `reports/OPENCODE_AUDIT.md` 后才运行正式发现轨迹。审核中发现真正阻塞就写明证据，不用一句“看起来合理”绕过。

## 运行策略

按 RUN_WINDOWS.md 使用绑定解释器启动。每一阶段读取真实退出码，非零立即停止后续 GPU 任务，仍可生成部分报告。日志应显示阶段、run_id、step、有效token、实际LR与eta*wd、loss、内存和阶段耗时。

discover 结束后读取 `results/local_gate.json` 与 `results/candidate_frozen.json`，核对确认损失、数值底噪、配对区间、基线与scalar边界标志。只有 `next_action=RUN_CONFIRM` 才准运行 confirm。不能因为其他 attention/早期 cell 好看而推翻主 cell gate。

confirm 后无论结果如何都停止 GPU 实验。即使 `GO_SHORT_BRANCH_ONLY` 也不直接开展分支训练；本轮授权已经用完。报告中 method_novelty、end_to_end_gain、deployment_cost 按合同保持未证明/未测试。

## 交付

请输出最少但完整的结果摘要：真实模型/词表/参数量、数据与token预算、4条之内的已运行轨迹、所选LR、主cell的两个seed原始Delta/区间/null底噪、M到A的诊断变化、最近邻方向对照、内存与成本明细、最终状态和停止原因。解释性cell单列，不用作挑赢家。

生成 `reports/LOCAL_FINDINGS.md`、完整 CSV/JSON、`reports/OPENCODE_AUDIT.md` 和轻量结果包。明确哪些是实际运行结果，哪些仍是待验证假设。没有充分信号就如实建议停止当前有限配置；不要给正面结论凑理由。
