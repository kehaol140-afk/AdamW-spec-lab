# LAYER2_GPU_GATE — Layer-2 前置门状态（停在 gate，等待授权）

日期：2026-10-11。Layer-1 已完成并通过预注册主门；本文件按 F_P_S_MATCHED_BASELINE_PLAN §3 与 frozen_v1.json §layer2 列出 Layer-2 的执行前状态。**未获用户正式授权前不启动。**

## 1. 代码级解锁状态

| 条件 | 状态 |
|---|---|
| layer1_gate.json {pass: true} | ✅ 已写入（双主门通过，反转未触发） |
| holdout_v2_manifest {status: MINTED_SEALED} | ✅ 保持封存（sha 3e0f7653… 复核未变） |
| runner layer2 双锁 | ✅ 代码内硬检查生效（gate 文件 + 封存状态，缺一拒跑） |
| **用户授权 Layer-2（~5.2 h GPU）** | ⬜ **待授权** |

## 2. Layer-2 预注册内容回顾（不可变更）

- 臂：S1 / S8 / F8 / **P8（强制包含，修复 V16 screen 事故）** × seeds **{3411, 3412, 3413}**
  （已核 3407–3410、2030、3101 无冲突）
- 预算：T=4096，warmup 256，LR 0.001；数据缓存 8.39M ≤ 8.5M token（各 seed 流从游标 0）
- holdout_v2：全部 12 端点身份锁定后由 `dv_runner.py layer2_holdout` **一次性开封**（状态机 +
  哈希双校验；已实测封锁：非 MINTED_SEALED 即拒）
- 推断单位 = 训练 seed：逐 seed 差值 + 均值/标准差/极差/同号率；文档 bootstrap 仅作评价误差
- 旧 V16 3-seed 结果只作历史引用，不与新 holdout 合并；旧 final_holdout 永久烧毁不复用

## 3. 预算（冻结估算，需在此窗口内批准）

| 项 | 值 |
|---|---|
| 训练 | 12 × 4096 步 ≈ 4.6 h（同平台本轮实测约 2×加速，实际可能 ≈ 1.5–2.5 h；上限按冻结 7 h 容错报批） |
| 评估 | dev 4 视界 × 12 + holdout_v2 × 12 ≈ 0.6 h |
| **合计报批** | **~5.2 h GPU（上限 7 h）**，建议单次连续窗口 |
| 磁盘 | 端点 12×18MB + 日志/eval ≈ 400 MB |
| VRAM/RSS | ~0.25 GiB / ~2 GiB（GPU0；GPU1 vLLM 不受影响） |

## 4. Layer-1 预期被检验的假设（预注册，Layer-2 结果不得反向修改本层记录）

- H1（方向必要）：F8/P8 − S8 < 0 于 3/3 seeds（Layer-1 点估计 −0.204/−0.235）
- H2（幅度有害）：S8 − S1 > 0 于 3/3 seeds（Layer-1 点估计 +0.082）
- H3（P≥F）：F8 − P8 ≥ 0 的 seed 数（Layer-1 点估计 +0.031）

## 5. 风险与既知事实

- Layer-1 运行事故（观测层设备 bug、checkpoints 目录、进程误杀）均已修复/留痕（RUNNER_FIX_LOG.md、
  LAYER1_DIRECTION_EFFECT.md 附录）；数学内核 dv_core.py 与协议 frozen_v1.json 哈希未变
- Layer-2 运行建议：`setsid` 脱离 + 预建 `raw/checkpoints/` + 监控短命令轮询（吸取 Layer-1 教训）
- 12 臂串行单进程；如中断，append-only `done` 机制支持确定性续跑（锚点重复行为已知并将在分析端去重校验）
