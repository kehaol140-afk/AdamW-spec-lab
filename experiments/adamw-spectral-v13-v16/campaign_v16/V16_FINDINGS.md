# V16 FINDINGS — ADAMW_SPECTRAL_CAMPAIGN_V16_20261002

终态：**STOP_RESOURCE**（非科学结论）。封包时间 2026-10-03。父 STOP 保持；V1–V15 全部只读。

## 发生了什么

用户授权 72h 自主实验。计划（`PLAN_FROZEN.md`/`campaign_config.json`）与持久化 runner（STATE/QUEUE/DECISIONS/STATUS + 断点续跑）已按合同建成并真实运行。**本机可用 RAM 在多集实验中稳定于 ~1.9–2.1 GiB**：训练进程驻留 ~1.4 GiB，运行中可用内存跌破冻结的 1.0 GiB 硬停线；跨 ≥3 个 2 小时暂停集无恢复（资源曲线在包内）。按合同"同一暂停累计2小时→STOP_RESOURCE 封包，不无限等待"执行。**这不是科学负结果**——A/B/C/D 阶段的判读问题仍未回答。

## 已完成（真实产物）

1. **A0 H0 来源更正（完成）**：真实加载 step256/step128 checkpoint 零训练重评。V15 的 H0_step256=8.952 是随机初始化值（INVALID_STATE_ORIGIN，原文件只读保留）；更正值：step256 dev=5.997226 / confirm=6.064046；step128 dev=6.544758 / confirm=6.611260（顺序合理）。每条新评估记录 loaded_checkpoint_sha/state_id/global_step/参数指纹；eval 源守卫（未加载状态不得挂 checkpoint 标签）已有回归测试。V15 台账差异补充说明（backwards 文字 6800 vs 台账、gpu_seconds 字段未用）见 `a0_h0_correction.json`。
2. **数据与评估集封存（完成）**：训练缓存 8.5M token（父 1,052,996 前缀逐字节验证一致 + 流式扩展；uint32 34MB）；development 256（V12 旧 dev 128 + 新 128）、screen 256、final_holdout 1024（final_holdout_reserved 角色，未开封）。
3. **A1 完成 31/36 条**（每条 256 步、父全局调度、development H64/H256 评估）：

| 家族 | 已测最优 (a,b) | dev H256 NLL |
|---|---|---|
| S | a=8, b=2（网格角点）| 5.434261 |
| F | a=8, b=2（网格角点）| **5.431514** |
| P | a=2, b=2 | 5.436529 |

初步读法（未到选择终点，5 条 P 分支未完成）：F 对最优 S 的点差 +2.75e-3；S 自身继续沿 a、b 双向走到新网格边界 → **SCALE_RANGE_UNRESOLVED_AT_CAP**（按合同不再扩 16/4，也不把边界 STOP 当全项目停止）。

## 预算（全部真实计数，历史不清零）

- optimizer.step 尝试 7,939 / 120,000；完成 7,939；token-forwards 19,171,782 / 400,000,000。
- 墙钟 ~10h（含资源等待；设备任务计时见台账——train/direction/eval 秒数分列，gpu_seconds 旧字段已废弃说明）。
- 4 个已归档工程 attempt（GPU 空闲位误判为节流、进程被杀后 running 悬挂/孤儿锁、步日志键名不一致致恢复崩溃循环、暂停循环不吃 runner 截止）——全部有补丁说明与归档，训练语义零改动；失败计数持久化后无任务被 3-strike。

## 恢复方法（单命令）

RAM 恢复 ≥2.6 GiB 持续后：`powershell -File campaign_v16\RUN_CAMPAIGN.ps1`（或 `tools_resume_driver.py`，含 RAM 门控与分集逻辑）。队列从剩余 5 条 A1 + a1_select 自动继续 A2→B→C→D→final_holdout→封包；已消耗预算与身份链自动继承。

## 五问回答状态

1) V15 信号在更强对照下剩多少——**未回答**（B 未跑）。2) 方向 vs 幅度反馈——**未回答**（A2 未跑）。3) 2048/4096 从头 + 3 seed——**未回答**（未达）。4) 成本——runner/资源/删除日志齐全，无能耗声明。5) 谱主问题下一假设——A1 边界现象（a、b 同时走大）提示"整体步幅再平衡"可能仍是最强简单解释，待 A2/B 判别。

已答完的问题到此为止；剩余为工程性等待，不为用满时长造任务。原始数据、身份与审计链全部在 `v16_light.zip`（重开校验通过）。
