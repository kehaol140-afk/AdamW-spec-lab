# HYPERBALL_PUBLIC_BASELINE_AUDIT_CN — A0 公开基线复现准备（零训练）

日期：2026-09-30。性质：A0 静态准备，**本轮新增训练 updates=0、模型评价=0**。
结论先行：**来源获取失败（全部 SOURCE_BLOCKED）→ 本轮无法建立任何论文/作者配方
的核对内容；环境静态审计完成（条件基本满足，一项依赖未知）。A1 不启动。**

---

## 1. 三层身份（先分栏，未混写）

| 层 | 本轮状态 |
|---|---|
| ALGORITHM_IMPLEMENTATION（Hyperball 更新公式实现） | **未开始**——论文与作者 PR 不可达，公式只能来自任务书转述（见 §3），未经原文核对，不得据此实现并称忠实 |
| PUBLIC_BENCHMARK_REPRODUCTION（modded-nanogpt Track3 一一对应） | **未开始**——Track3 README/结果 log/脚本均不可达 |
| REDUCED_SCALE_TRANSFER（MiniMind 30M 迁移） | 未运行；即便未来执行也不称论文复现 |

## 2. 原始来源获取记录（每个 URL ≤2 次尝试，全部失败）

详见 `outputs/hyperball_a0/source_fetch_log.json`。网络诊断：ICMP 出网正常，
HTTPS 到 arxiv.org / github.com / raw.githubusercontent / api.github.com /
objects.githubusercontent / huggingface.co **全部被网关拦截（HTTP 000）**；
国内镜像（tuna/gitee，HTTP 200）可达但 **gitee 无 modded-nanogpt/hyperball
镜像**（API 搜索空）；服务器本地全盘检索无任何相关缓存。

| 来源 | 尝试 | 结果 |
|---|---|---|
| A 论文 arXiv:2606.16899 | 2 | SOURCE_BLOCKED |
| B Track3 README（raw+web） | 2+2 | SOURCE_BLOCKED |
| C MuonH PR #267（web+api） | 2+2 | SOURCE_BLOCKED |
| D MuonH capture-full log | 2 | SOURCE_BLOCKED |
| E tuned Muon baseline（result #36 log/脚本） | 随 B 同源 | SOURCE_BLOCKED |
| F tuned-aux MuonH（result #37） | 随 B 同源 | SOURCE_BLOCKED |
| G NVIDIA Emerging-Optimizers | 2 | SOURCE_BLOCKED |

**不根据摘要、任务书转述或旧聊天补造源码/配方数值**——下表所有需要原始来源
的字段一律 UNKNOWN。

## 3. 任务书转述的更新式（仅登记，未核对，非实现依据）

任务书给出 Hyperball 对 hidden 2D 矩阵 W 的形式：R=‖W₀‖_F（**何时捕获：未知**）；
u_hat=u/‖u‖_F；W_tilde=W−lr·R·u_hat；W_next=R·W_tilde/‖W_tilde‖_F。
所有语义问题（R 捕获时机、路由、momentum/NS 相对位置、WD、eps、init std、
LR/cooldown、NS 次数/系数、隐藏技巧）**UNKNOWN**——这些问题恰是 PR/论文/日志
才能回答的，不允许猜测。

## 4. 复现对照表（全部字段待来源）

| 项目 | Tuned Muon | MuonH 公开run | tuned-aux MuonH |
|---|---|---|---|
| model/architecture | UNKNOWN | UNKNOWN | UNKNOWN |
| dataset | UNKNOWN | UNKNOWN | UNKNOWN |
| batch / seq len / tokens | UNKNOWN | UNKNOWN | UNKNOWN |
| optimizer routing / Muon公式 / Hyperball公式 | UNKNOWN | UNKNOWN | UNKNOWN |
| matrix/aux LR / WD / momentum / init / schedule / cooldown×2 | UNKNOWN | UNKNOWN | UNKNOWN |
| stopping/eval / target loss | UNKNOWN | UNKNOWN | UNKNOWN |
| seeds / mean±std | UNKNOWN | UNKNOWN | UNKNOWN |
| exact source / commit / log SHA | SOURCE_BLOCKED | SOURCE_BLOCKED | SOURCE_BLOCKED |

## 5. 5090 可执行性静态审计（`env_static_audit.json`，无 GPU 计算）

| 项 | 状态 |
|---|---|
| 2×RTX 5090 32GB（sm_120 Blackwell）、驱动 591.44 | ✓ |
| torch 2.11.0+cu13.0、bf16 ✓、triton 3.6.0 ✓ | ✓ |
| flash_attn | **未安装**——Track3 是否必需 UNKNOWN（来源阻塞）；若需则为兼容安装（不改科学语义） |
| Track3 要求的 torch 版本 | UNKNOWN（README 不可达） |
| A100/H100 专属 kernel | UNKNOWN（源码不可达；若存在，等价替换不改语义、但 wall-clock 不可比） |
| 数据下载（FineWeb 等） | **受阻**：HF 不可达；国内镜像是否覆盖该数据 UNKNOWN |
| 磁盘 170GB 空闲 | 对 124M 级 benchmark+数据 足够（估计） |
| 124M 模型单/双卡启动 | 静态判断：32GB×2 对 124M+Track3 典型 batch 可行（**未验证，禁止本轮 smoke**） |

**判定：CANONICAL_REPRO_BLOCKED**——非硬件阻塞，而是 **来源阻塞**：无论文/PR/
日志/脚本，无法定义"canonical"为何，也无法核对任何配方字段。硬件环境本身无
已知不满足项（除 flash_attn/HF 数据两项 UNKNOWN）。

## 6. 科学语义不变的兼容修改（预估，待来源后确认）

1. flash_attn 若缺失：以 torch SDPA 替代（若 Track3 用 FA 仅作速度）——语义等价
   需对照源码确认注意力实现一致；
2. A100/H100 专属 kernel（如有）：替换为通用实现——数值语义不变、wall-clock 不可比；
3. torch.compile/triton 版本差异：保持算子语义，接受编译时间差异；
4. 数据经国内镜像获取（若存在）：须核对数据 SHA 与官方一致。
以上任何一条在来源可达前均为**假设**，不预先实施。

## 7. A1 拟议（本轮不执行，且当前不具启动条件）

- 阶段1：1 seed correctness（tuned Muon + MuonH 各 1 条，Track3 完整设置）；
- 阶段2：合理后 2–3 独立 seed screening；
- 成功标准（按任务书）：稳定训练；轨迹在公开 seed 波动范围内；MuonH 参数范数满足
  Hyperball 约束（诊断见 §8）；代码路径与作者版本对应；
- 资源范围（粗估，非承诺）：124M 模型在 2×5090 上 Track3 级步数的单条训练为
  小时级（准确值来源阻塞不可估，不拿别的模型吞吐当成本模型）；
- **启动前置**：全部 A–G 来源可达并完成 §4 表核对。

## 8. 预埋诊断（未来 A1 获批时才记录，本轮未实现）

每 hidden matrix 每 operator 聚合：‖W‖_F、‖raw optimizer update‖_F、
真实 angular displacement ‖W_{t+1}/‖W_{t+1}‖ − W_t/‖W_t‖‖_F。无 SVD/HVP/额外
forward；不据此提出 adaptive radius/budget controller/新优化器。

## 9. UNKNOWN 清单（全部）

论文全部实验细节；Hyperball 语义九项（§3）；MuonH/Muon/tuned-aux 三配方全部
字段（§4）；Track3 公开结果数值与 seed 统计；作者代码版本/log SHA；
NVIDIA 参考实现；Track3 torch 版本要求与专属 kernel；FineWeb 数据镜像可达性；
5090 实测吞吐（禁止本轮测量）。
