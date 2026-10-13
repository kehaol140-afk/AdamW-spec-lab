# ENVIRONMENT_AND_INSTALL — 从干净 Linux 到 R0/R1/R2

## R0：5 分钟 CPU 复算（零第三方依赖）

```bash
git clone <this-repo> && cd AdamW-spec-lab
python3 recompute/run_recompute.py          # 期望 8/8 OK
python3 tools/repro.py list                 # 六线能力总览
python3 tools/repro.py check --experiment direction-validation   # 资产检查（预期报 BLOCKED_ASSET 即正确）
```

## R1/R2：训练级（资产门控）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r env/requirements-training.txt   # torch==2.11.0 numpy==2.4.6 psutil==6.1.1（服务器实测版本）
cp configs/paths.example.json configs/paths.json && $EDITOR configs/paths.json
python3 tools/repro.py check                  # 全线资产+sha（<400MB 自动校验 sha16）
python3 portable/run_dv_layer1.py             # dry-run 计划；--execute 需 ADAMWSPEC_ALLOW_GPU=1（另行授权）
```

服务器历史环境（非编造，实测于 fr_minimind env）：Python 3.11.16 / torch 2.11.0（driver 591.44, 2×RTX 5090）/ numpy 2.4.6 / psutil 6.1.1。CUDA_VISIBLE_DEVICES 显式指定是历史纪律。

## GPU 与封存纪律

- 本仓任何入口不默认启 GPU；`train --yes` 且 `ADAMWSPEC_ALLOW_GPU=1` 双闸才可能进入执行路径（本轮均未实现执行）
- `holdout_v2` 封存集不存在于本仓任何自动加载路径（docs/SEALED_ASSET_EXCLUSIONS.csv）

## CI

`.github/workflows/cpu_ci.yml`：AST 全检 + 文档/manifest/链接校验 + 8/8 复算 + CLI dry-run（无 GPU、不拉 >10MB 数据、不触封存）。
