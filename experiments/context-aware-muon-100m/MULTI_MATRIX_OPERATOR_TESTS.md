# MULTI_MATRIX_OPERATOR_TESTS — CPU 正确性与 GPU smoke 结果（本轮实测）

- 日期：2026-10-12。测试载体：`tests/`（4 文件 23 项）+ `scripts/smoke_100m.py`。全部通过。
- 运行方式：`python - <<'EOF'`（无 pytest 依赖，逐测试调用；见 tests/ 源码头部）。

---

## 1. CPU 正确性（23/23 PASS）

| 组 | 测试 | 关键断言 | 结果 |
|---|---|---|---|
| model | 100M 参数计数 = 100,784,000；tied 单对象；目标块 768×2432 | 计数/身份 | PASS |
| model | 30M = 30,025,216（回归规模） | 计数 | PASS |
| model | 初始化确定性（同 seed 位级同、异 seed 异） | fingerprint | PASS |
| routing | 39/26/26/54 计数、互斥全并、全 2D、形状保持 | 结构 | PASS |
| routing | A=39 / B=65 / native=0 矩阵数 | 组大小 | PASS |
| ns5 | 正标度齐次性 NS5(cX)≈NS5(X)（fp64 cos > 0.9995） | 数学性质 | PASS |
| ns5 | 谱平坦化（gram 对角 ∈ [0.2, 2.3]） | 输出性质 | PASS |
| ns5 | 2D-only / 系数数校验；adjust_lr 三模式（original/match_rms/矩形比） | 接口 | PASS |
| ns5 | **源码全扫描无 SVD**（linalg.svd/torch.svd/.svd( )） | 项目规则 | PASS |
| group_muon | **native 位级等价 torch.optim.Muon**（3 步、5 形状、位级 equal） | 最高强度回归 | PASS |
| group_muon | λ₀=0.05 锚点：cos>0.99999、范数匹配 <1e-9、参数轨迹差 <1e-5 | 锚点 | PASS |
| group_muon | 动量态与 λ 无关（buffers 位级同） | 状态一致性 | PASS |
| group_muon | 范数匹配精度 <1e-9（fp64） | 方向隔离 | PASS |
| group_muon | λ=1 服从纯梯度方向（cos>0.9999） | 极限行为 | PASS |
| group_muon | low-μ 组覆写生效（0.9≠0.95） | 对照臂 | PASS |
| group_muon | state_dict roundtrip 位级重放 | 检查点 | PASS |
| group_muon | 重复运行位级确定 | 确定性 | PASS |
| group_muon | stats 值域 + NS5 成本核算字段 | 诊断 | PASS |
| group_muon | 1e-7 输入扰动地板（fp64 cos > 0.999） | 度量学 | PASS |

## 2. 本轮发现的两个语义真相（已修复并测试锁定）

1. **torch Muon 的 Nesterov 输入是 `(1−μ)g + μB`（`grad.lerp(buf, momentum)` 代码语义），其官方文档字符串写的 `g + μB` 与实现不符**（Keller Jordan 参考实现同 lerp 语义）。后果：mix 家族的原生锚点是 **λ₀=1−μ=0.05**，而非按文档推导的 1/(1+μ)≈0.513。优化器对 λ₀ 路由到同一 lerp 表达式实现位级锚点。
2. **fp32 cos 度量在 ~2M 元素上求和误差 ~1e-4**，足以伪造"方向漂移"或淹没真实信号（本轮调试实测：位级相同的两个 NS5 输出 fp32-cos=0.9999）。所有方向诊断已改 **fp64**；此前怀疑的"NS5 混沌放大"在 fp64 度量下不出现（1e-7 扰动 → cos≈1.0000）。

## 3. GPU smoke（GPU0，授权有限窗口，~1 分钟 GPU）

设置：100M/seed 3471/30 步 × 8192 tok/步/bf16。三路径**同初始化 fingerprint `33f93a0f`**（CRN 成立）。

| 路径 | s/步 | tok/s | VRAM 峰 | opt 秒/步 | NS5 秒/步 | loss 30 步 |
|---|---|---|---|---|---|---|
| tuned AdamW | 0.320 | 25.7k | 3.35 GiB | 0.010 | — | 8.94→7.21 |
| native Muon | 0.283 | 29.1k | 3.02 GiB | 0.052 | 0.045 | 8.92→7.22 |
| Group-A mix(0.75)+范数匹配 | 0.345 | 23.8k | 2.98 GiB | 0.117 | 0.059 | 8.90→7.26 |

- 范数匹配误差实测 **1.4e-16**；λ=0.75 末步 cos(O_mix,g)=0.184 / cos(O_mix,O_nat)=0.424（早期方向基线）。
- mix 路径 +22% 墙钟 vs native Muon，主因：(a) 39 矩阵双 NS5（方向隔离必要成本），(b) 每步每矩阵 fp64 诊断。**pilot 冻结前把 (b) 降为锚点步采样**（gap G-1），预计回落 ~0.30 s/步。
- 100M 实测吞吐 23.8–29.1k tok/s **高于**此前参数比外推（9–16k）——旧口径作废，预算以 100M_COMPUTE_AND_MEMORY.md 为准。
