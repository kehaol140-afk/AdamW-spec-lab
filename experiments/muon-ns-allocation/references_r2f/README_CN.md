# R2-F交接包

先读R2F_EXECUTION_PROTOCOL_v1_CN.md，再把OPENCODE_R2F_TASK_CN.md交给执行代理。

本包是前瞻实验设计，尚未运行服务器任务。r2f_contract.json绑定固定矩阵、阶段、条件D和预算；实际路径及SHA由服务器只读核对后补入executed配置。

references保留R2原协议、独立审计和原轻量结果包；其中缺少完整checkpoint/数据及部分旧日志，不能据本包声称完整GPU复现。旧W方案由本轮D设计替代，不同时运行。

plan_validation.json只验证计划算术、配方代数及矩阵计数，不是科研效果或训练正确性测试。原始参考文件保持不变。
