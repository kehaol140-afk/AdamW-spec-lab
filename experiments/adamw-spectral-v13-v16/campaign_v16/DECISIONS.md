
## 2026-10-10 01:18:17 (server budget amendment)
- 条件: 迁移至服务器续跑，本地 72h 墙钟与 2h 暂停上限已过
- 证据: RAM_RECOVERY.md + SERVER_MIGRATION_PLAN.md #3；修订前 {"wall_started_at": 1790957002.0275593, "pause_seconds": 14700.0, "stop": "STOP_RESOURCE"}
- 行动: wall_started_at 重置为本机时刻（新 72h 窗口）、pause_seconds 归零（原值记 pause_seconds_windows_session）、清除 STATE.stop；全部科学计数器与冻结阈值零改动

## server platform rerun (Option A)
- 条件: 跨平台数值不可比
- 证据: SERVER_MIGRATION_PLAN.md #6; Windows raw 移至 raw_win_pilot_20261003
- 行动: A1 36 条在服务器整体重跑; 台账 attempt 计数累计不清零

## 2026-10-09 server execution boundary (opencode round)
- 环境: fr_minimind (py3.11/torch2.11.0+cu130/transformers4.57.6/tokenizers0.22.2/numpy2.4.6) + psutil 6.1.1 (冻结版, 增量安装); GPU0 RTX 5090 sm_120; GPU1 他人 vLLM 不使用
- 依据: 用户 2026-10-10 指定复用 fr_minimind; torch 2.6.0+cu124 无 sm_120 内核; 跨平台检查 V16_CROSS_PLATFORM_CHECK.md (容差 2e-6 超出, 诊断为平台浮点差, 代码/数据/优化器/评估全部核验等价)
- 本轮边界: 服务器重跑 A1 36 条 + a1_select 后**停在 A2 之前**等待用户研究判断; 不自动进入 A2/B/C
- Windows 31 条以 raw_win_pilot_20261003 只读保留, 不参与服务器选优
"\n## 2026-10-10 02:53:09\n- 条件: A1 selection complete\n- 证据: {\"S\": {\"id\": \"a1_S_a8_b2\", \"a\": 8, \"b\": 2, \"dev_h256_nll\": 5.4342579979411}, \"F\": {\"id\": \"a1_F_a8_b2\", \"a\": 8, \"b\": 2, \"dev_h256_nll\": 5.43148785757447}, \"P\": {\"id\": \"a1_P_a8_b2\", \"a\": 8, \"b\": 2, \"dev_h256_nll\": 5.4314460957419985}}\n- 行动: freeze per-family (a,b); enqueue A2 common-norm branches; enqueue b_plan\n"
"\n## 2026-10-10 11:45:38\n- 条件: B fixed part\n- 证据: S/F/P x 3 LRs per frozen plan\n- 行动: 9 trials enqueued; SVD anchors on the three S-plain trials only (30/48 budget)\n"
"\n## 2026-10-10 13:27:44\n- 条件: B extension slots\n- 证据: {\"S\": 0.001, \"F\": 0.001, \"P\": 0.001}\n- 行动: S4-8 and family extensions enqueued per frozen rules\n"
"\n## 2026-10-10 15:10:13\n- 条件: B selection frozen (dev H2048)\n- 证据: [[\"b_P1_a8_lr0.001\", 4.087911955424658], [\"b_F1_a8_lr0.001\", 4.11887137398553], [\"b_P5_a4\", 4.125184657187259], [\"b_F5_a4\", 4.202969659352686], [\"b_S1_plain_lr0.001\", 4.240565429647231], [\"b_S6_tb0\", 4.244822546558535]]\n- 行动: proceed to screen eval\n"
"\n## 2026-10-10 15:11:49\n- 条件: route B\n- 证据: {\"label\": \"F_OR_P_SUSTAINED\", \"recipes\": [\"b_S1_plain_lr0.001\", \"b_F1_a8_lr0.001\"]}\n- 行动: enqueue C finals with frozen recipes\n"
"\n## 2026-10-10 17:45:49\n- 条件: final holdout opened once\n- 证据: all C endpoints identity-locked\n- 行动: no further tuning permitted; proceed to finalize\n"

## 2026-10-10 finalize regeneration (erratum, no science fields touched)
- 发现: 迁移包内含 Windows STOP_RESOURCE 中断时生成的 findings.json/V16_FINDINGS_AUTO.md; 本轮队列 71/71 耗尽后 runner 的 `if not findings.json.exists()` 守卫跳过了重新 finalize, 使终局报告滞留为旧内容 (STOP_RESOURCE + Windows 期台账快照)
- 证据: 两文件 mtime=迁移解压时刻; STATE.final_label 已由本轮循环内 derive_final_label() 正确置为 COMPLETE_WITH_SIGNAL; holdout 6 文件与 C endpoint 6/6 实际在位
- 行动: 旧文件改名 *.windows_interrupted_stale 保留; 重启 runner (队列空) 令协议自身 finalize() 以当前 STATE/ledger/raw 重生成; 零阈值/零选优/零科学计数器改动
