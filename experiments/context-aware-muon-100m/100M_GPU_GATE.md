# 100M_GPU_GATE — 资源与授权门（GPU0）

日期：2026-10-12。

---

## 1. 资源现状

| 资源 | 值 |
|---|---|
| GPU0 | RTX 5090 32G；本轮诊断前 441 MiB 桌面残留 / 0% util |
| GPU1 | RTX 5090 32G；外部 vLLM 长期占用 27.8 GiB —— **本项目禁用** |
| RAM | ~108G 空闲 |
| 磁盘 | ~148G 空闲（pilot 需 ~6.2G） |
| 环境 | `fr_minimind`（torch 2.11.0+cu13.0），python `/home/b205/miniforge3/envs/fr_minimind/bin/python` |

## 2. 授权台账

| 项 | 窗口 | 实耗 | 依据 |
|---|---|---|---|
| 首轮 smoke（30 步 ×3） | 已授权（前轮） | ~0.02 GPU-h | smoke_100m.json |
| **本轮 100 步诊断 ×4 + 重放 ×2** | 本任务书授权 ≤0.3 GPU-h | **~0.066 GPU-h** | preflight_100steps.json（含启动/评价/恢复全计时） |
| **正式 pilot 15 臂** | **未授权——本 gate 等待用户** | 预算 4.9–5.5 GPU-h | 100M_PILOT_EXECUTION_LOCK.md |
| 确认层 | 未授权（pilot 过门后再议） | 预算 9–32 GPU-h | 100M_COMPUTE_AND_MEMORY.md |

## 3. 运行纪律

1. 全部训练 `CUDA_VISIBLE_DEVICES=0` 显式指定；启动前 nvidia-smi 探测（GPU0 第三方占用 >2 GiB 即暂停并重排）。
2. 与 30M 线 T2（若获授权）的 GPU0 窗口互斥；单阶段连续窗口执行，不交错。
3. 逐臂 run.lock 幂等防重跑；任一臂墙钟超出预算估计 +20% 触发收缩规则（停、报告、重授权）。
4. 非有限 loss / VRAM OOM / 缓存 sha 不符 = 立即中止并报告（无自动重试）。
5. 每臂产物落 `results/pilot/<arm>/`；证据 ZIP 哈希进 `SHA256SUMS.txt`。

## 4. Gate 判定

**正式 pilot 的启动条件（全部满足）**：
- [x] 代码冻结（CODE_FREEZE_V1.json，18 文件）
- [x] 数据冻结（缓存 sha + 双前缀字节证明）
- [x] 100 步诊断全绿（有限值/恢复位级/吞吐/范数匹配）
- [x] 25/25 CPU 测试 PASS
- [ ] **用户对 ~4.9–5.5 GPU-h pilot 窗口的显式授权**

授权后按 LOCK 文档 §2 队列执行；本轮按任务书停在 gate。
