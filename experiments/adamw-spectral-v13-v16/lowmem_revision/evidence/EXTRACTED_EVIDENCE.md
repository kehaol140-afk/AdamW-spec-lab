# 上传文件中的原始证据摘录

以下为直接提取，不是新的GPU运行记录。

原zip SHA256: `c7c023bdf960361f8fa6afc2091bfa9b27b335b7aeb4e1a68f851531312962fe`

## logs/smoke_opencode.log

```text
{"time": 1790870184.5583773, "kind": "command_failed", "command": "smoke", "error": "RESOURCE_BLOCKED: available RAM=2100858880"}
Traceback (most recent call last):
  File "E:\大模型预训练\adamw_spectral_pilot\run.py", line 100, in main
    smoke(cfg, cfg_hash)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 728, in smoke
    res = resource_check(cfg)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 703, in resource_check
    raise RuntimeError(f"RESOURCE_BLOCKED: available RAM={available_ram}")
RuntimeError: RESOURCE_BLOCKED: available RAM=2100858880

```

## logs/smoke_opencode_retry1.log

```text
{"time": 1790870211.885508, "kind": "command_failed", "command": "smoke", "error": "RESOURCE_BLOCKED: available RAM=2045362176"}
Traceback (most recent call last):
  File "E:\大模型预训练\adamw_spectral_pilot\run.py", line 100, in main
    smoke(cfg, cfg_hash)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 728, in smoke
    res = resource_check(cfg)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 703, in resource_check
    raise RuntimeError(f"RESOURCE_BLOCKED: available RAM={available_ram}")
RuntimeError: RESOURCE_BLOCKED: available RAM=2045362176

```

## logs/smoke_opencode_retry2.log

```text
{"time": 1790870927.2350304, "kind": "command_failed", "command": "smoke", "error": "RESOURCE_BLOCKED: available RAM=1907736576"}
Traceback (most recent call last):
  File "E:\大模型预训练\adamw_spectral_pilot\run.py", line 100, in main
    smoke(cfg, cfg_hash)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 728, in smoke
    res = resource_check(cfg)
  File "E:\大模型预训练\adamw_spectral_pilot\pilot_core.py", line 703, in resource_check
    raise RuntimeError(f"RESOURCE_BLOCKED: available RAM={available_ram}")
RuntimeError: RESOURCE_BLOCKED: available RAM=1907736576

```

## pilot_core.py:698-710

```text
698: def resource_check(cfg: dict) -> dict:
699:     if not torch.cuda.is_available():
700:         raise RuntimeError("RESOURCE_BLOCKED: bound Python has no CUDA")
701:     available_ram = ram_available()
702:     if available_ram < cfg["runtime"]["system_ram_free_reserve_gib"] * 2 ** 30:
703:         raise RuntimeError(f"RESOURCE_BLOCKED: available RAM={available_ram}")
704:     free, total = torch.cuda.mem_get_info(0)
705:     limit = min(cfg["runtime"]["vram_soft_limit_gib"] * 2 ** 30,
706:                 free - cfg["runtime"]["vram_free_reserve_gib"] * 2 ** 30)
707:     if limit <= 0:
708:         raise RuntimeError("RESOURCE_BLOCKED: VRAM reserve unavailable")
709:     return {"free_vram": free, "total_vram": total, "process_vram_limit": limit,
710:             "available_ram": available_ram}
```

## pilot_core.py:726-740

```text
726: def smoke(cfg: dict, cfg_hash: str):
727:     b, data, mem = require_data(cfg, cfg_hash)
728:     res = resource_check(cfg)
729:     with run_lock("smoke"):
730:         seed = cfg["baseline"]["discovery_seed"]
731:         random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
732:         torch.set_num_threads(cfg["runtime"]["cpu_threads"])
733:         torch.backends.cuda.matmul.allow_tf32 = False
734:         torch.backends.cudnn.allow_tf32 = False
735:         peak = cfg["baseline"]["peak_lr_grid"][1]
736:         model, opt, aliases, count = model_and_optimizer(cfg, b, peak, "cuda:0")
737:         if ram_available() < cfg["runtime"]["system_ram_free_reserve_gib"] * 2 ** 30:
738:             raise RuntimeError("RESOURCE_BLOCKED: RAM reserve below 2 GiB after model initialization")
739:         fp = fingerprint(cfg_hash, data["data_hash"])
740:         atomic_json(ROOT / "artifacts" / "smoke_run_manifest.json", {"status": "RUNNING", **fp,
```

## pilot_core.py:650-663

```text
650:                "cursor": cursor, "total_effective_tokens": total_tokens,
651:                "forward_backward_seconds": t_backward - t0, "optimizer_seconds": t_step - t_backward,
652:                "rss_bytes": rss,
653:                "allocated_vram_bytes": torch.cuda.memory_allocated() if device.startswith("cuda") else 0,
654:                "reserved_vram_bytes": torch.cuda.memory_reserved() if device.startswith("cuda") else 0}
655:         append_csv(ROOT / "logs" / "runtime.csv", row)
656:         event("train_step", **row)
657:         if rss > cfg["runtime"]["process_rss_soft_limit_gib"] * 2 ** 30:
658:             raise RuntimeError("RESOURCE_BLOCKED: RSS soft limit exceeded")
659:         if ram_available() < cfg["runtime"]["system_ram_free_reserve_gib"] * 2 ** 30:
660:             raise RuntimeError("RESOURCE_BLOCKED: system RAM reserve crossed")
661:         if vram_limit is not None and device.startswith("cuda") and torch.cuda.memory_reserved() > vram_limit:
662:             raise RuntimeError("RESOURCE_BLOCKED: process VRAM soft limit crossed")
663:     return stop, cursor, total_tokens
```

## pilot_workflow.py:94-115

```text
94: 
95: 
96: def _candidate_directions(a_cpu, cfg):
97:     t0 = time.perf_counter()
98:     u, sigma, vh = torch.linalg.svd(a_cpu.double(), full_matrices=False)
99:     head = sigma[0] * torch.outer(u[:, 0], vh[0])
100:     rest = a_cpu.double() - head
101:     directions = []
102:     for rho in cfg["probe"]["rho_grid"]:
103:         for scale in cfg["probe"]["candidate_scale_grid"]:
104:             directions.append(("spectral", float(rho), float(scale), None,
105:                                scale * (rho * head + rest)))
106:     for scale in cfg["probe"]["scalar_control_scale_grid"]:
107:         directions.append(("scalar", 1.0, float(scale), None, scale * a_cpu.double()))
108:     for q in cfg["probe"]["diagnostic_nearest_neighbor"]["q_grid"]:
109:         c = q * sigma[0]
110:         filtered = torch.zeros_like(a_cpu) if c == 0 else \
111:             (u * (sigma / torch.sqrt(1 + (sigma / c).square()))) @ vh
112:         for scale in cfg["probe"]["candidate_scale_grid"]:
113:             directions.append(("spectra_diagnostic", 1.0, float(scale), float(q), scale * filtered))
114:     return directions, head, rest, sigma, time.perf_counter() - t0
115: 
```

## run.py:46-60

```text
46: 
47: def bundle_light(cfg, cfg_hash):
48:     dest = ROOT / "artifacts" / "adamw_spectral_light.zip"
49:     paths = [ROOT / "run.py", ROOT / "pilot_core.py", ROOT / "pilot_workflow.py",
50:              ROOT / "prepare_data.py", ROOT / "tests" / "test_numerics.py",
51:              ROOT / "config" / "local_v1.json", ROOT / "PROTOCOL_LOCAL_V1.md",
52:              ROOT / "RUN_WINDOWS.md", ROOT / "bindings.local.json",
53:              ROOT / "RUN_COMMANDS_RESOLVED.ps1", ROOT / "data" / "SOURCE.json",
54:              ROOT / "artifacts" / "preflight.json", ROOT / "artifacts" / "data_manifest.json",
55:              ROOT / "artifacts" / "protocol_resolved.json",
56:              ROOT / "artifacts" / "smoke_run_manifest.json"]
57:     for sub in ("reports", "results", "artifacts/numerics", "logs"):
58:         base = ROOT / sub
59:         if base.exists():
60:             paths += [p for p in base.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".csv", ".jsonl", ".log")]
```

## 本地逻辑回放的范围

resource_gate_replay.json执行的是上传resource_check函数的AST，RAM采用三条日志数值，CUDA是否可用与空闲显存为mock。它只证明原检查在旧阈值拒绝、改为1GiB时允许这三条输入，不证明CUDA初始化后内存足够，也不证明训练成功。
