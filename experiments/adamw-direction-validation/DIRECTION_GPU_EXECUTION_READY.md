# DIRECTION_GPU_EXECUTION_READY — GPU 执行就绪清单（停在 gate）

日期：2026-10-11。状态：**代码/数据/协议全部就绪；唯一剩余阻塞 = GPU 排程 + 用户授权**。

## 1. S8/F8/P8 的目标块更新范数定义与配置一致性（任务书第 6 项核实）

**范数定义（三臂逐字相同，dv_core.apply_step 与 V16 冻结代码同源）**：

- 每步（梯度已 ÷valid_total、全局 clip 1.0 后）解析计算目标块 AdamW 增量 `A_t`（含 bias correction）
- **S8**：`D = A_t` → 目标块位移范数 `q_t = η·8·‖A_t‖`
- **F8**：`D = N_A(g_clip)`（fp64 精确匹配 `‖D‖ = ‖A_t‖`）→ `q_t = η·8·‖A_t‖`
- **P8**：`D = N_A(g_clip/(√v̂⁺+ε))`（同上匹配）→ `q_t = η·8·‖A_t‖`
- **语义学结论**：三臂在每一步的范数定义**恒等**（都等于 η·8·‖本分支当前 A_t‖，F/P 由构造精确匹配，S 平凡成立）。各臂 `‖A_t‖` 的**轨迹**是内生结果（训练分化所致），逐步记录 q_t 并报告——这是 layer-1 设计（匹配语义，非外生范数锁定；后者是 V16 A2 的职责，已完成其历史使命）
- 其他参数：三臂均为 `−η·b·A_other`（b=1 → 原生 AdamW）；目标块衰减 `(1−η·λ)` 三臂一致（校正发生在原生 step 之后，不触碰衰减）

**配置一致性核对表（dv frozen_v1 vs V16 b_F1/b_P1/b_S1，逐项）**：

| 项 | 三臂值 | 与 V16 B 对应臂 |
|---|---|---|
| LR 峰值 | 0.001（同一值，非各自调参） | 相同（b_*_lr0.001 族） |
| schedule | warmup128 线性 → cosine 至 0.1·peak，T=2048 | 相同（campaign_config.b） |
| 训练时钟 | 同一公式、同一 step 索引驱动 η；无额外时钟 | 相同 |
| 初始化 | `torch.manual_seed(3407)` 后构建模型（同 RNG 消耗顺序）→ 三臂初始权重**相同**；也与 V16 B 三臂相同（复刻门的基础） | 相同 |
| 数据 | 8.5M 缓存（`c91382dc…`）从游标 0，micro 2 × accum 4 × seq 256（CRN 同流） | 相同 |
| betas/eps/wd | (0.9,0.95) / 1e-8 / 矩阵 0.1、1D 0 | 相同 |
| clip | 全局 1.0（先 ÷valid_total） | 相同 |
| 评估 | development H512/1024/2048（同集合同口径） | 相同 |

## 2. Layer-1 GPU 执行清单（待授权）

顺序执行（单进程串行，GPU0）：

```bash
cd /home/b205/projects/training_dynamics_lab/adamw_direction_validation
CUDA_VISIBLE_DEVICES=0 nohup /home/b205/miniforge3/envs/fr_minimind/bin/python src/dv_runner.py layer1 \
    > raw/layer1_run.out 2>&1 &
```

| 序 | 臂 | 内容 | 验证点 |
|---|---|---|---|
| 0 | （自动） | 模型构建/缓存加载 | 无异常 |
| 1 | dv_S1 | S(1,1) 2048 步 | **复刻门**：step 日志 vs V16 `b_S1_plain_lr0.001`（eta/loss_sum/q_t 逐步位级一致预期） |
| 2 | dv_S8 | S(8,1)【缺失格】 | q_t ≈ 8×dv_S1 同步 q_t（首步精确 8×） |
| 3 | dv_F8 | F(8,1) | 复刻门 vs `b_F1_a8_lr0.001` |
| 4 | dv_P8 | P(8,1) | 复刻门 vs `b_P1_a8_lr0.001` |
| 5 | dv_S4 | S(4,1) 中点 | q_t ≈ 4×S1 |
| 6 | （分析） | 主门：F8−S8 与 P8−S8 的 H2048 配对差（bootstrap CI95） | 写 `layer1_gate.json{pass}` |

停止条件：复刻门失败 / 非有限值 / zero_norm_fallback / MASK_OR_SHIFT_CHANGED → 全停并报告（不重试不覆盖）。

## 3. 成本（最终）

| 项 | 值 |
|---|---|
| Layer-1 GPU | 5 臂 × ~12.5 min ≈ **63 min**（含评估，上限 90 min） |
| VRAM / RSS | ~0.25 GiB reserved / ~2 GiB |
| 磁盘 | ~120 MB（日志+端点） |
| Layer-2（gate 通过后另批） | 4 臂 × 3 seeds × 4096 步 + holdout_v2 一次开封 ≈ **5.2 h** |
| 合计潜在 | ~6.3 h GPU |

## 4. Gate 状态（本轮结束点）

- [x] 代码就绪（runner + holdout 开封阶段；8/8 CPU 测试；layer-2 双重封锁实测生效）
- [x] 数据就绪（语料验证链 V1-V6 全过；holdout_v2 MINTED_SEALED，sha `3e0f7653…`）
- [x] 协议冻结（frozen_v1.json；P8 强制 layer-2；a 网格不外扩）
- [ ] **SpecMuon P1 完成并释放 GPU0**
- [ ] **用户授权 layer-1**（~1.5h 墙钟窗口）
- [ ] （其后）layer-1 gate 结果 + 用户授权 layer-2

## 5. 禁止事项重申

旧 final_holdout 已开封（烧毁）——不得用于任何新选择/确认；holdout_v2 仅一次性开封；不复用、不修改 V16/Muon/SpecMuon 任何文件；失败不重试覆盖；预算不自行续期。
