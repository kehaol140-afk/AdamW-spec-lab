# REPRODUCE — 仓级复算指南（干净 clone 即可运行）

## 环境

Linux/macOS，Python ≥3.9，**仅标准库**（json/csv/random/hashlib/zipfile）。无 GPU、无第三方包、无模型权重依赖。

## 一键复算（8 个审计检查点）

```bash
python3 recompute/run_recompute.py
cat recompute/outputs/SUMMARY.json   # 应 8/8 OK
```

| CK | 检查点 | 预期（本次随仓输出已附） |
|---|---|---|
| V16C_3seed | C 阶段 3 seed | S−F=+0.036276/+0.094018/+0.151494（精确） |
| V16_A2 | A2 四臂 | F@qS−S=−0.002658 等（±3e-6 舍入） |
| DV_LAYER1 | Layer-1 匹配矩阵 | F8−S8=−0.204090；P8−S8=−0.235050（点估计精确） |
| S30M_T1 | 30M T1 | −0.037930/−0.045737/−0.013957；Muon−P2=−0.272500（精确） |
| MSA_attribution | B_L 归因 | B_L/(V_N−V_AH)=101.70%（精确） |
| SPECMUON_P1 | 14 分支双恢复点 | 与 P1_RESULTS.csv 全字段一致 |
| CAM_100M | 15 臂 | H4096 与 CSV 0 偏差；C_A−锚=+0.1367；C_B−C_A=+0.0997；严格单调 14/15 |
| HYPERBALL_A1 | A1 双臂 | 3.27906/3.27921；H 臂范数首末 261.928/261.928 |

## 精度与容差口径

- token 加权聚合（sum loss/sum tokens）无随机性 →「精确」= 与发布表逐位一致
- bootstrap CI 依赖重采样程序：本仓 pooled token 加权 + doc 级重采样（1000 次, seed 9071）复现点估计精确、CI 端点与冻结例程差 ≤1.5e-3（程序差异已声明，不构成 mismatch）
- 若出现真实偏差：勿改原始表；按 docs/ERRATA_AND_RECONCILIATION.md §4 流程写 REPRODUCE_MISMATCH 并溯源（数据身份/指标定义/旧勘误）

## 各实验目录内的补充复算入口

见各目录 REPRODUCE / METHOD_CODE_MAP_REPRODUCE（如 specmuon p1 逐分支 json、msa 逐轮 csv 直读核对）。

## 不能在本仓重放的（如实声明）

任何**重训**（需 GPU + token 缓存 + 上游语料，均未随仓；缓存 mint 配方与 sha 在各 manifest）；A2 q_override 重放与 anchors SVD（需 GPU+冻结 runner）；holdout 类新评估（final_holdout 已烧毁、holdout_v2 封存未开封）。
