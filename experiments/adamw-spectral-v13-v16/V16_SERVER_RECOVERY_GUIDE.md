# V16 Server Recovery Guide

目标：在 `/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot` 无损恢复并继续 `ADAMW_SPECTRAL_CAMPAIGN_V16_20261002`（A1 剩余 5 条 → a1_select → A2 → B → C → D → final_holdout）。

---

## 1. 解包与校验（任何训练之前）

```bash
cd /home/b205/projects/training_dynamics_lab
unzip v16_server_transfer_20261010.zip        # 产生唯一顶层目录 adamw_spectral_pilot/
cd adamw_spectral_pilot
python3.9 MIGRATION_INFO/verify_package.py    # 或: sha256sum -c MIGRATION_INFO/SHA256SUMS.txt
```

`verify_package.py` 必须输出 `VERIFY OK`（逐文件哈希 + 关键身份 + A1 清单 31/5 全过）。

## 2. 环境

```bash
python3.9 -m pip install -r MIGRATION_INFO/requirements_frozen.txt
# torch CUDA 版本按服务器驱动选择（本地为 2.6.0+cu124）；版本偏离会放大跨平台数值差异，见 §6
python3.9 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. 绑定改写（唯一必须修改的文件）

```bash
cp MIGRATION_INFO/bindings.server_example.json bindings.local.json
# 核对三处路径真实存在: repo_root / tokenizer_path（source_data_path 仅重建缓存时需要）
python3.9 -c "
import json; b=json.load(open('bindings.local.json'))
from pathlib import Path
assert (Path(b['repo_root'])/'model'/'model_minimind.py').is_file()
assert (Path(b['tokenizer_path'])/'tokenizer.json').is_file()
print('bindings OK')"
```

合法依据：`bindings.local.json` 不在冻结 SOURCE_FILES 内、不参与 checkpoint config_hash。**其余任何 campaign_v16 源码与 `config/local_v1_lowmem.json` 逐字节不得改动**（否则 source_hash / LEDGER_MANIFEST_MISMATCH / checkpoint config_hash 校验失败）。

## 4. 预算修订（一次性，先于任何运行）

本地 72h 墙钟已过期、暂停累计超限、STATE.stop=STOP_RESOURCE。运行修订脚本（只改 wall_started_at / pause_seconds / STATE.stop 三个运营字段并写 DECISIONS.md，科学计数器与阈值零改动）：

```bash
python3.9 MIGRATION_INFO/amend_budget_server.py
```

## 5. 预检与续跑

```bash
# 回归测试（无需 GPU 亦可跑 CPU 部分）
python3.9 -m unittest campaign_v16.test_v16 -v
python3.9 -m unittest campaign_v16.test_resume_driver -v   # 注意: 该驱动含 Windows 路径, 仅测试逻辑部分

# 串行续跑（单写入者；崩溃后同命令续跑；不使用 tools_resume_driver.py, 服务器无需 Windows RAM 门控）
cd /home/b205/projects/training_dynamics_lab/adamw_spectral_pilot
nohup python3.9 -m campaign_v16.run --resume --max-seconds 259200 \
  > campaign_v16/logs/server_run.out 2>&1 &
tail -f campaign_v16/logs/server_run.out
```

队列自动从 `a1_P_a4_b1`（从中断的 step 1 重算，父 checkpoint 内 RNG/游标完整）→ 其余 4 条 P → `a1_select`（提取 q_series、冻结每族 (a*,b*)、入队 A2）→ A2 4 条 → b_plan → B → 路由 → C → holdout → finalize。

**红线**：绝不删除/改动 `run_manifest.json`、`budget_ledger.json`、`QUEUE.jsonl`、`raw/`、`eval_sets.json`；绝不重跑 `init_campaign()`（会清零台账并要求重扫语料）；`final_holdout` 保持封存直至 C 全部终点身份锁定（planner 自动开封一次）；同时只允许一个 runner 进程（run.lock/ledger.lock 自动守护）。

## 6. 跨平台可比性（先决策再训练）

Windows(31 条) 与 Linux/GPU 的逐文档 NLL **不得进入同一统计比较**（浮点归约顺序不同，2e-6 nats/token 分辨率无法保证）。两种方案：

**方案 A（推荐，~1.5h 服务器算力）：服务器重跑全部 36 条 A1，Windows 31 条转 pilot 归档**

```bash
cd campaign_v16
mv raw raw_win_pilot_20261003          # 保留 Windows 证据（移动而非删除）
mkdir raw && mkdir raw/steps raw/eval  # 供 A0 的 H0CORR 已随 raw 移走, A0 在队列中为 done 不会重跑
python3.9 - <<'EOF'
import json
from pathlib import Path
q = Path("QUEUE.jsonl")
rows = [json.loads(l) for l in q.open(encoding="utf-8")]
for r in rows:
    if r.get("stage") == "A1":
        r["status"] = "pending"; r.pop("note", None)
lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
q.write_text(lines, encoding="utf-8")
with open("DECISIONS.md", "a", encoding="utf-8") as f:
    f.write("\n## server platform rerun (Option A)\n- 条件: 跨平台数值不可比\n"
            "- 证据: SERVER_MIGRATION_PLAN.md #6; Windows raw 移至 raw_win_pilot_20261003\n"
            "- 行动: A1 36 条在服务器整体重跑; 台账 attempt 计数累计不清零\n")
EOF
cd .. && nohup python3.9 -m campaign_v16.run --resume --max-seconds 259200 \
  > campaign_v16/logs/server_run.out 2>&1 &
```

**方案 B（不推荐）：仅补 5 条与 Windows 31 条混合**。前提是先做平台校准——服务器重跑 1 条已完成分支（如 `a1_S_a1_b1`，256 步）并与 Windows 值比较；偏差超过 2e-6 nats/token（预期会超过）即判混合无效，回退方案 A。

无论 A/B：B/C 从头训练在服务器内自洽（同机同 seed 配对）；从 A1 转移的只有配置级量 (a*, b*)。

## 7. 恢复后验收清单

- [ ] `verify_package.py` = VERIFY OK
- [ ] 预算修订已写 DECISIONS.md；`python -m campaign_v16.run` 可启动（无 STOP_RESOURCE 立即退出）
- [ ] 首条任务 `a1_P_a4_b1` 从父 step256 成功续训 256 步（步日志 1–256 连续、H64/H256 各 256 行、身份字段齐全）
- [ ] A1 36/36 → `a1_select` 落盘 STATE.selections.a1 + q_series
- [ ] A2 4 条完成 → b_plan → B（≤18×2048）→ 路由 → C（≤4×3×4096）→ final_holdout 一次开封 → finalize 出 findings.json
- [ ] 台账新计数与磁盘 raw 一致；v16_light.zip 重新生成

## 8. 故障排查

| 症状 | 原因 | 处置 |
|---|---|---|
| `LEDGER_MANIFEST_MISMATCH` | run_manifest.json 被改动 | 从包重解 run_manifest.json（原哈希见 SHA256SUMS） |
| `PARENT_CHECKPOINT_MISMATCH` | step256.pt 损坏 | 重传，校验 `ed09a2c8…` |
| `EVAL_WITHOUT_STATE_ORIGIN` / 身份断言 | 误删 raw 或手工构造状态 | 恢复对应 raw 文件；禁止手工伪造评估行 |
| `CUDA_UNAVAILABLE` / VRAM 不足 | 驱动/环境 | 本地实测 reserved 峰值 ~165 MB，任何现代 GPU 均远低于上限 |
| runner 立即退出 `STOP_BUDGET` | 未运行 §4 修订，或 wall 又超 72h | 运行/重跑 amend_budget_server.py（DECISIONS.md 会累积记录） |
| `ModuleNotFoundError: model.model_minimind` | bindings.repo_root 未指向 minimind_repo | 回到 §3 |
