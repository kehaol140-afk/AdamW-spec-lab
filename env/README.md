# env/ — 环境与依赖

## R0：CPU 结果复算（本仓库默认能力）

- Python ≥ 3.9，**仅标准库**（json/csv/random/hashlib/zipfile/tarfile）
- 安装：无需任何第三方包。见 `requirements-recompute.txt`

## R1/R2：训练级重放/重训（资产门控，未在本轮授权）

服务器实际训练环境（取自运行时环境，非编造）：

| 组件 | 版本（实测服务器） |
|---|---|
| Python | 3.11.16（conda env `fr_minimind`） |
| PyTorch | 2.11.0（CUDA 构建；驱动 591.44，2×RTX 5090） |
| numpy | 2.4.6 |
| psutil | 6.1.1（V16 campaign 台账用） |
| GPU 纪律 | CUDA_VISIBLE_DEVICES 显式指定；bf16/fp16 autocast 按各线冻结配置 |

安装指南：见 [docs/ENVIRONMENT_AND_INSTALL.md](../docs/ENVIRONMENT_AND_INSTALL.md)。requirements 文件两档：`requirements-recompute.txt`（空=标准库）/ `requirements-training.txt`（torch/numpy/psutil，按上表版本 pin）。

## 路径配置

复制 `configs/paths.example.json` → `configs/paths.json`，或用环境变量 `ADAMWSPEC_DATA_ROOT/CKPT_ROOT/V16_ROOT/MUON_TRANSITION_ROOT/MINIMIND_ROOT/OUTPUT_ROOT` 覆盖。只读输入与 `OUTPUT_ROOT` 分离；任何 wrapper 不会写源目录。
