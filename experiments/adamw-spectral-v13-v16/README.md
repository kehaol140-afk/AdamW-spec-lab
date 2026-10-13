# adamw-spectral-v13-v16 — AdamW 更新的谱/方向修正（V13→V16）

研究线目录索引：

| 文件 | 内容 |
|---|---|
| [THEORY.md](THEORY.md) | 科学问题、S/F/P 更新公式（逐行对应 `campaign_v16/core.py`）、已验证/仍假设 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | V13/V14/V15/V16(A1/A2/B/C/D) 全部正式实验设计、预算、偏差 |
| [RESULTS.md](RESULTS.md) | 按时间顺序的完整结果表（含全部 18 条 B 分支、C 3-seed 表、V14/V15 数值） |
| [CODE_MAP.md](CODE_MAP.md) | 文件→函数→阶段→哈希；上游 minimind 关系；历史版本 |
| [REPRODUCE.md](REPRODUCE.md) | 环境/入口/只读复算/重跑资产/身份哈希/不可复现项 |
| [V16_FINAL_RECONCILED.md](V16_FINAL_RECONCILED.md) | 旧报告对账、Round-1 勘误（+0.1144 口径）、判读文档指针 |
| `interpretation/` | 8 篇服务器完赛判读（复制自 audits，SHA 随附） |
| `RUN_INDEX.csv` / `MANIFEST.sha256` | 运行级索引 / 逐文件哈希 |

一句话总结：在冻结 AdamW 复刻环境中，「用当前（预条件）梯度方向替换目标块动量商方向 + 8× 幅度」在续训/从头/跨 seed 三层级一致为正（V16 `COMPLETE_WITH_SIGNAL`），不被幅度轨迹、LR 调优、低 β1 解释；证据边界=单目标块、4.66M、dev 级（C 期 holdout 3-seed 除外）。
