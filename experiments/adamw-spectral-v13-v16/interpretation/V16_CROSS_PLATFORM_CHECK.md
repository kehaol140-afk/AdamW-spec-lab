# V16_CROSS_PLATFORM_CHECK — 跨平台一致性验证

- 执行时间：2026-10-09 17:15–17:25 UTC；环境：GPU0 (RTX 5090) + fr_minimind（Python 3.11.16 / torch 2.11.0+cu130 / transformers 4.57.6 / tokenizers 0.22.2 / numpy 2.4.6 / psutil 6.1.1）
- 方法：全部检查在**影子副本** `/tmp/opencode/v16_replay/adamw_spectral_pilot` 中执行（core.apply_step 会写预算账本，真实 campaign 零接触）；Windows 参照 = 迁移包内 31 条 A1 原始 JSONL（字节校验一致）
- 脚本与原始数值：本目录 `rc_checks/rc_consistency.py`、`rc_results.json`
- 冻结容差：2e-6 nats/token（恢复指南 §6 / 迁移计划 §6 事前规定，未修改）

## 1. 结论表

| # | 检查 | 结果 |
|---|---|---|
| RC0 | 父 checkpoint/训练缓存/tokenizer/config 哈希；V16 缓存含父 1,052,996 token 前缀逐字节一致 | ✅ 全部一致 |
| RC1 | S(1,1) 4 步重放：**eta 逐位相等**（训练时钟/调度数学 bitwise） | ✅ |
| RC2 | 同上：数值域差 | ❌ 超容差：nll 中位差 1.3e-5、最大 1.76e-5（=容差的 6–9 倍）；A_norm/q_t 相对差 ~1.4e-5 |
| RC3 | S(1,1) 64 步 + H64 development 256 文档评估：doc 顺序与 token 计数**完全一致**；token 加权 NLL 差 2.81e-5 | ✅ 口径一致 / ❌ 数值超容差（14×） |
| RC4 | S/F/P step-1 恒等式：S 的 ‖D‖≡‖A‖（服务器上逐位）；F ‖D‖=‖A‖；P(a=4)/S q_t 比值 = 4.000000000；三族 step-1 loss_sum 服务器上完全相同（CRN 保留） | ✅ 语义逐字保持 |
| RC5 | development/screen/final_holdout 文档 ID 两两交集 | ✅ 全部为 0（无混用；holdout 未开封） |
| RC6 | 服务器自确定性：同恢复同代码同数据两次运行 step-1 loss_sum | ✅ 逐位相同（服务器内 CRN 配对有效） |

## 2. 超容差诊断（按用户要求定位差异来源）

| 假设 | 证据 | 裁决 |
|---|---|---|
| 浮点/平台算术差异 | eta 逐位相等但一切 CUDA 数值量有 ~1e-5 相对漂移；平台偏移在三族间完全一致（+0.2217 loss_sum）；RC6 证明服务器内逐位可复现 | **✅ 成立（唯一来源）**：bf16 autocast + fp32 归约在 Windows-GPU/torch2.6+cu124 与 RTX 5090/torch2.11+cu130 的内核/归约顺序不同 |
| 训练代码差异 | 逐字运行冻结 campaign 代码；eta/时钟逐位；梯度累积/clip/apply_step 路径为原代码本体 | ❌ 排除 |
| 优化器差异 | step=256 状态恢复、S 恒等式逐位成立、P/S 比值精确 4.0、β/wd/eps 分组结构一致 | ❌ 排除 |
| 评估方法差异 | RC3 doc 顺序、token 计数、mask/shift 校验（MASK_OR_SHIFT_CHANGED 守恒）全部一致 | ❌ 排除 |
| 数据读取差异 | 缓存哈希一致 + 父前缀逐字节一致 + 游标 524,288 推进一致 | ❌ 排除 |

## 3. 裁决

1. **跨平台逐文档 NLL 混合无效**（超出冻结容差 2e-6，实测 2.8e-5–1.1e-4）——证实方案 B 不可行，**方案 A（服务器统一重跑 36 条）是唯一被授权且被证据支持的路径**，与本轮用户决策一致。
2. 代码、数据、优化器、评估、RNG、时钟、划分全部核验等价 → 服务器重跑**忠实于冻结协议**；且 RC6 证明服务器内部逐位确定 → 服务器内 S/F/P 配对比较与选择有效。
3. 未为通过验证修改任何容差、损失、优化器或评估定义。

## 4. 环境记录（正式运行环境 = fr_minimind）

| 项 | 冻结要求 | 实际 | 说明 |
|---|---|---|---|
| Python | 3.9 | 3.11.16 | 用户指定复用 fr_minimind |
| torch | 2.6.0+cu124 | 2.11.0+cu130 | sm_120 所迫；GPU 冒烟（fp32/fp64/BF16/autograd/AdamW/ck roundtrip）全过 |
| transformers | 4.49.0 | 4.57.6 | 模型构建/前向/评估实测正常 |
| tokenizers | 0.21.1 | 0.22.2 | 运行时用封存 uint32 缓存，tokenization 不进路径 |
| numpy | 2.0.2 | 2.4.6 | memmap/统计实测正常 |
| psutil | 6.1.1 | **6.1.1（精确冻结版）** | 本轮增量安装，未动其它包 |
| GPU capability | — | sm_120（Blackwell），driver 591.44 | 仅 GPU0；GPU1 vLLM 未触碰 |

torch 2.8.0 备选环境（asp_v16，仅 python3.9 空壳）保留未用；版本漂移与环境选择已按边界记录于 DECISIONS.md（见 V16_A1_SERVER_RESULTS 附录）。
