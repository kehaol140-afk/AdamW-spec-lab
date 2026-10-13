"""Serial baseline and local-probe workflow. Formal runs are never invoked on import."""
from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from pilot_core import loss_sum_count

from pilot_core import (ROOT, adamw_next, append_csv, atomic_json, eval_blocks, event,
                        fingerprint, isolated_probe, load_checkpoint, loss_sum_count,
                        model_and_optimizer, read_json, require_data, require_resource, resource_check,
                        run_lock, save_checkpoint, schedule_lr, sha_file, sha_object,
                        source_hash, train_steps, training_batch)


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _nll(rows):
    return sum(r["loss_sum"] for r in rows) / sum(r["token_count"] for r in rows)


def _checkpoint_dir(seed: int, peak: float):
    return ROOT / "artifacts" / "checkpoints" / f"seed{seed}_lr{peak:g}"


def baseline_run(cfg, cfg_hash, binding, data, mem, eval_data, seed, peak, vram_limit):
    run_dir = _checkpoint_dir(seed, peak)
    run_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(cfg_hash, data["data_hash"])
    run_id = sha_object({**fp, "seed": seed, "peak_lr": peak})
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if (old.get("status") == "SUCCESS" and old.get("run_id") == run_id and
                (run_dir / "step512.pt").exists() and
                old.get("checkpoint_hashes", {}).get("512") == sha_file(run_dir / "step512.pt")):
            return old
        if old.get("run_id") != run_id:
            raise RuntimeError("existing baseline run has different manifest; preserve it and use a new protocol")
    _seed(seed)
    model, opt, aliases, count = model_and_optimizer(cfg, binding, peak, "cuda:0")
    require_resource(cfg, "after_baseline_model_init", stage=f"baseline_seed{seed}_lr{peak:g}",
                     vram_limit=vram_limit)
    report = {"status": "RUNNING", "run_id": run_id, **fp, "seed": seed,
              "peak_lr": peak, "parameter_count": count}
    atomic_json(manifest_path, report)
    step, cursor, total_tokens = 0, 0, 0
    ckpt0 = run_dir / "step0.pt"
    if not ckpt0.exists():
        require_resource(cfg, "before_checkpoint_save", stage="baseline", gpu_updates=step,
                         vram_limit=vram_limit)
        save_checkpoint(ckpt0, model, opt, step=0, cursor=0, total_tokens=0,
                        config_hash=cfg_hash, data_hash=data["data_hash"], aliases=aliases)
        require_resource(cfg, "after_checkpoint_save", stage="baseline", gpu_updates=step,
                         vram_limit=vram_limit)
    for boundary in cfg["baseline"]["checkpoint_steps"][1:]:
        ckpt = run_dir / f"step{boundary}.pt"
        if ckpt.exists():
            require_resource(cfg, "before_checkpoint_restore", stage="baseline", gpu_updates=step,
                             vram_limit=vram_limit)
            payload = load_checkpoint(ckpt, model, opt, config_hash=cfg_hash,
                                      data_hash=data["data_hash"], aliases=aliases)
            require_resource(cfg, "after_checkpoint_restore", stage="baseline", gpu_updates=step,
                             vram_limit=vram_limit)
            step, cursor, total_tokens = (payload["successful_optimizer_step"],
                                          payload["data_cursor"], payload["total_effective_tokens"])
            continue
        step, cursor, total_tokens = train_steps(model, opt, cfg, mem, peak_lr=peak,
                                                 step=step, cursor=cursor,
                                                 total_tokens=total_tokens, stop=boundary,
                                                 device="cuda:0", log_prefix=f"baseline_seed{seed}_lr{peak:g}",
                                                 vram_limit=vram_limit)
        require_resource(cfg, "before_checkpoint_save", stage="baseline", gpu_updates=step,
                         vram_limit=vram_limit)
        save_checkpoint(ckpt, model, opt, step=step, cursor=cursor, total_tokens=total_tokens,
                        config_hash=cfg_hash, data_hash=data["data_hash"], aliases=aliases)
        require_resource(cfg, "after_checkpoint_save", stage="baseline", gpu_updates=step,
                         vram_limit=vram_limit)
    rows = eval_blocks(model, eval_data["lr_select"], "cuda:0")
    for row in rows:
        append_csv(ROOT / "results" / "baseline_runs.csv",
                   {"run_id": run_id, "seed": seed, "peak_lr": peak, "step": step,
                    "split": "lr_select", **row, "nll": row["loss_sum"] / row["token_count"]})
    report.update(status="SUCCESS", step=step, cursor=cursor, effective_tokens=total_tokens,
                  lr_select_nll=_nll(rows), checkpoint_hashes={str(k): sha_file(run_dir / f"step{k}.pt")
                  for k in cfg["baseline"]["checkpoint_steps"]})
    atomic_json(manifest_path, report)
    return report


def _target_name(role):
    return "model.layers.1.mlp.down_proj.weight" if role == "ffn_down_projection" else \
           "model.layers.1.self_attn.o_proj.weight"


def _candidate_directions(a_cpu, cfg):
    t0 = time.perf_counter()
    u, sigma, vh = torch.linalg.svd(a_cpu.double(), full_matrices=False)
    head = sigma[0] * torch.outer(u[:, 0], vh[0])
    rest = a_cpu.double() - head
    directions = []
    for rho in cfg["probe"]["rho_grid"]:
        for scale in cfg["probe"]["candidate_scale_grid"]:
            directions.append(("spectral", float(rho), float(scale), None,
                               scale * (rho * head + rest)))
    for scale in cfg["probe"]["scalar_control_scale_grid"]:
        directions.append(("scalar", 1.0, float(scale), None, scale * a_cpu.double()))
    for q in cfg["probe"]["diagnostic_nearest_neighbor"]["q_grid"]:
        c = q * sigma[0]
        filtered = torch.zeros_like(a_cpu) if c == 0 else \
            (u * (sigma / torch.sqrt(1 + (sigma / c).square()))) @ vh
        for scale in cfg["probe"]["candidate_scale_grid"]:
            directions.append(("spectra_diagnostic", 1.0, float(scale), float(q), scale * filtered))
    return directions, head, rest, sigma, time.perf_counter() - t0


def _next_build(model, opt, cfg, mem, cursor, eta, target_name):
    seq = cfg["runtime"]["sequence_length"]
    mb = cfg["runtime"]["micro_batch_size"]
    accum = cfg["runtime"]["accumulation_steps"]
    model.train()
    opt.zero_grad(set_to_none=True)
    valid_total = 0
    for _ in range(accum):
        ids, cursor = training_batch(mem, cursor, mb, seq, "cuda:0")
        lengths = torch.full((mb,), seq + 1, device="cuda:0", dtype=torch.long)
        # Build the virtual next update with the same forward precision as baseline training.
        use_bf16 = torch.cuda.is_bf16_supported()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, lengths)
        loss.backward()
        valid_total += count
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(valid_total)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["baseline"]["global_grad_clip_norm"])
    named = dict(model.named_parameters())
    target = named[target_name]
    old = target.detach().clone()
    grad = target.grad.detach().clone() if target.grad is not None else None
    if grad is None:
        raise RuntimeError("P0_FAIL: target gradient is None")
    bc = cfg["baseline"]
    a, _ = adamw_next(old, grad, opt.state[target], lr=eta, wd=bc["matrix_weight_decay"],
                      beta1=bc["betas"][0], beta2=bc["betas"][1], eps=bc["eps"])[1:]
    for group in opt.param_groups:
        group["lr"] = eta
    opt.step()
    torch.cuda.synchronize()
    expected = old * (1 - eta * bc["matrix_weight_decay"]) - eta * a
    error = float((target.detach() - expected).abs().max())
    if error > bc.get("fp32_atol", 1e-7) + cfg["numerics"]["fp32_rtol"] * float(expected.abs().max()):
        raise RuntimeError(f"P0_FAIL: actual target AdamW step mismatch {error}")
    base = old * (1 - eta * bc["matrix_weight_decay"])
    opt.zero_grad(set_to_none=True)
    return target, base, a.detach().cpu().double(), grad.detach().cpu().double(), cursor, valid_total, error


def _diagnostic_m(a_cpu, state, grad_cpu, cfg):
    bc = cfg["baseline"]
    beta = bc["betas"][0]
    m = state.get("exp_avg", torch.zeros_like(grad_cpu)).cpu().double()
    m_plus = beta * m + (1 - beta) * grad_cpu.double()
    u, s, vh = torch.linalg.svd(m_plus, full_matrices=False)
    paired = torch.einsum("ki,kl,il->i", u, a_cpu, vh)
    numerator = float(paired.square().sum())
    denominator = float(a_cpu.square().sum())
    return {"m_paired_diagonal_numerator": numerator,
            "a_frobenius_squared": denominator,
            "paired_diagonal_capture": numerator / denominator if denominator else None,
            "unexplained_fraction": 1 - numerator / denominator if denominator else None,
            "unexplained_interpretation": "includes in-basis off-diagonal and out-of-thin-basis components"}


def _direction_diagnostics(model, target, base, a_cpu, head, rest, rows,
                           *, seed, step, rep, name, eta):
    module_name = name.removesuffix(".weight")
    module = dict(model.named_modules())[module_name]
    with torch.no_grad():
        target.copy_(base)
    model.eval()
    h_norm = float(torch.linalg.vector_norm(head))
    r_norm = float(torch.linalg.vector_norm(rest))
    for row in rows:
        saved = []
        hook = module.register_forward_pre_hook(lambda _module, args: saved.append(args[0].detach().cpu().double()))
        try:
            ids = torch.tensor(row["ids"], dtype=torch.long, device=target.device)[None, :]
            lengths = torch.tensor([ids.shape[1]], device=target.device)
            t0 = time.perf_counter()
            logits = model(ids[:, :-1], use_cache=False).logits
            loss_sum, count = loss_sum_count(logits, ids, lengths)
            gradient = torch.autograd.grad(loss_sum / count, target, retain_graph=False)[0].detach().cpu().double()
            probe_seconds = time.perf_counter() - t0
        finally:
            hook.remove()
        x = saved[0].reshape(-1, saved[0].shape[-1])
        hx = x @ head.T
        rx = x @ rest.T
        head_energy = float(hx.square().sum())
        rest_energy = float(rx.square().sum())
        cross = float((hx * rx).sum())
        joint = float((hx + rx).square().sum())
        gh = float((gradient * head).sum())
        gr = float((gradient * rest).sum())
        common = {"seed": seed, "checkpoint_step": step, "replicate": rep,
                  "parameter_name": name, "split": "probe_select", "doc_id": row["doc_id"],
                  "eta": eta, "loss_sum": float(loss_sum.detach()), "token_count": count,
                  "probe_seconds": probe_seconds}
        append_csv(ROOT / "results" / "direction_signals.csv",
                   {**common, "gradient_dot_head": gh, "gradient_dot_rest": gr,
                    "head_frobenius": h_norm, "rest_frobenius": r_norm,
                    "gradient_frobenius": float(torch.linalg.vector_norm(gradient)),
                    "head_cosine": gh / (float(torch.linalg.vector_norm(gradient)) * h_norm)
                    if h_norm and torch.linalg.vector_norm(gradient) else None,
                    "rest_cosine": gr / (float(torch.linalg.vector_norm(gradient)) * r_norm)
                    if r_norm and torch.linalg.vector_norm(gradient) else None})
        append_csv(ROOT / "results" / "output_effects.csv",
                   {**common, "input_vectors": x.shape[0], "head_energy": head_energy,
                    "rest_energy": rest_energy, "cross_energy": cross,
                    "joint_energy": joint,
                    "orthogonality_identity_error": joint - head_energy - rest_energy - 2 * cross,
                    "eta_squared_joint_energy": eta * eta * joint})
    with torch.no_grad():
        target.copy_(base - eta * a_cpu.to(device=target.device, dtype=target.dtype))


def probe_cell(cfg, cfg_hash, binding, data, mem, eval_data, *, seed, peak, step, role,
               split, frozen=None):
    checkpoint = _checkpoint_dir(seed, peak) / f"step{step}.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"required checkpoint missing: {checkpoint}")
    model, opt, aliases, _ = model_and_optimizer(cfg, binding, peak, "cuda:0")
    require_resource(cfg, "after_probe_model_init", stage=f"probe_{role}_{split}")
    require_resource(cfg, "before_probe_checkpoint_restore", stage=f"probe_{role}_{split}")
    payload = load_checkpoint(checkpoint, model, opt, config_hash=cfg_hash,
                              data_hash=data["data_hash"], aliases=aliases)
    require_resource(cfg, "after_probe_checkpoint_restore", stage=f"probe_{role}_{split}")
    name = _target_name(role)
    if name not in dict(model.named_parameters()):
        raise RuntimeError(f"target parameter unresolved: {name}")
    eta = schedule_lr(peak, step + 1, cfg)
    wd = cfg["baseline"]["matrix_weight_decay"]
    rows = []
    null_errors = []
    for rep in range(cfg["probe"]["gradient_build_replicates"]):
        require_resource(cfg, "before_probe_replicate", stage=f"probe_{role}_{split}")
        cursor = payload["data_cursor"] + rep * cfg["runtime"]["effective_batch_size"] * cfg["runtime"]["sequence_length"]
        state = {"cursor": cursor, "step": step}
        with isolated_probe(model, opt, state):
            target = dict(model.named_parameters())[name]
            old_state = copy.deepcopy(opt.state[target])
            target, base, a_cpu, grad_cpu, _, build_tokens, parity_error = _next_build(
                model, opt, cfg, mem, cursor, eta, name)
            directions, head, rest, sigma, svd_seconds = _candidate_directions(a_cpu, cfg)
            diag = _diagnostic_m(a_cpu, old_state, grad_cpu, cfg)
            spectrum = {"seed": seed, "checkpoint_step": step, "replicate": rep,
                        "parameter_name": name, "shape": list(a_cpu.shape),
                        "a_frobenius": float(torch.linalg.vector_norm(a_cpu)),
                        "head_fraction": float(sigma[0].square() / sigma.square().sum()) if sigma.square().sum() else None,
                        "head_gap_relative": float((sigma[0] - sigma[1]) / sigma[0]) if len(sigma) > 1 and sigma[0] else None,
                        "near_degenerate": bool(sigma[0] == 0 or len(sigma) > 1 and (sigma[0] - sigma[1]) / sigma[0] <
                                                cfg["numerics"]["head_near_degeneracy_relative_gap_flag"]),
                        "svd_seconds": svd_seconds, "parity_max_abs": parity_error, **diag}
            append_csv(ROOT / "results" / "spectral_metrics.csv", spectrum)
            if split == "probe_select":
                _direction_diagnostics(model, target, base, a_cpu, head, rest,
                                       eval_data[split], seed=seed, step=step,
                                       rep=rep, name=name, eta=eta)
            if step == 256 and role == "ffn_down_projection" and split == "probe_select":
                repetitions = [_nll(eval_blocks(model, eval_data[split], "cuda:0"))
                               for _ in range(cfg["numerics"]["null_repeats"])]
                null_errors.append(max(repetitions) - min(repetitions))
            if frozen is not None:
                with_baseline = [*frozen, {"kind": "scalar", "rho": 1.0, "scale": 1.0, "q": None}]
                allowed = {(x["kind"], float(x.get("rho", 1)), float(x["scale"]), x.get("q"))
                           for x in with_baseline}
                directions = [d for d in directions if (d[0], d[1], d[2], d[3]) in allowed]
            # Ordinary AdamW is the s=rho=1 scalar candidate and shares every other parameter.
            for kind, rho, scale, q, direction in directions:
                require_resource(cfg, "before_probe_candidate", stage=f"probe_{role}_{split}")
                candidate = base - eta * direction.to(device="cuda:0", dtype=target.dtype)
                with torch.no_grad():
                    target.copy_(candidate)
                measures = eval_blocks(model, eval_data[split], "cuda:0")
                for measure in measures:
                    rows.append({"seed": seed, "checkpoint_step": step, "replicate": rep,
                                 "parameter_name": name, "shape": "x".join(map(str, a_cpu.shape)),
                                 "split": split, "kind": kind, "rho": rho, "scale": scale,
                                 "q": q, "eta": eta, "wd": wd, "eta_wd": eta * wd,
                                 "build_tokens": build_tokens, "parity_max_abs": parity_error,
                                 "direction_frobenius": float(torch.linalg.vector_norm(direction)),
                                 **measure, "nll": measure["loss_sum"] / measure["token_count"]})
                with torch.no_grad():
                    target.copy_(base - eta * a_cpu.to(device="cuda:0", dtype=target.dtype))
                require_resource(cfg, "after_probe_candidate", stage=f"probe_{role}_{split}")
        require_resource(cfg, "after_probe_replicate", stage=f"probe_{role}_{split}")
    dest = ROOT / "results" / ("probe_confirm.csv" if split == "probe_confirm" else "probe_grid.csv")
    for row in rows:
        append_csv(dest, row)
    if null_errors:
        atomic_json(ROOT / "artifacts" / "numerics" / "identity_probe.json",
                    {"status": "PASS", "seed": seed, "step": step, "role": role,
                     "repeat_spreads": null_errors, "epsilon_null": max(null_errors),
                     "meaning": "same model, same state, same data, three repeated FP32 evaluations"})
    event("probe_cell", seed=seed, step=step, role=role, split=split, rows=len(rows))
    return rows


def _key(row):
    return row["kind"], row["rho"], row["scale"], row["q"]


def _select(rows, kind, tolerance=0.0):
    keys = {_key(r) for r in rows if r["kind"] == kind}
    if not keys:
        raise RuntimeError(f"no {kind} candidates")
    values = {}
    for key in keys:
        subset = [r for r in rows if _key(r) == key]
        values[key] = _nll(subset)
    best_value = min(values.values())
    tied = [k for k, v in values.items() if v <= best_value + tolerance]
    best = sorted(tied, key=lambda k: (abs(math.log(k[1])) if k[1] else 0,
                                            abs(math.log(k[2])) if k[2] else float("inf"),
                                            str(k)))[0]
    return {"kind": best[0], "rho": best[1], "scale": best[2], "q": best[3],
            "select_nll": values[best]}


def _paired_gate(rows, scalar, spectral, cfg, epsilon_null):
    def matching(r, c):
        return _key(r) == (c["kind"], c["rho"], c["scale"], c["q"])
    scalar_rows = [r for r in rows if matching(r, scalar)]
    spectral_rows = [r for r in rows if matching(r, spectral)]
    def grouped(rs):
        groups = {}
        for r in rs:
            item = groups.setdefault(r["doc_id"], [0.0, 0])
            item[0] += r["loss_sum"]
            item[1] += r["token_count"]
        return groups
    sg, pg = grouped(scalar_rows), grouped(spectral_rows)
    ids = sorted(set(sg) & set(pg))
    if len(ids) < cfg["data"]["probe_confirm_blocks"]:
        return {"status": "DATA_INSUFFICIENT", "distinct_documents": len(ids)}
    delta = sum(sg[i][0] for i in ids) / sum(sg[i][1] for i in ids) - \
            sum(pg[i][0] for i in ids) / sum(pg[i][1] for i in ids)
    rng = np.random.default_rng(cfg["gate"]["bootstrap_seed"])
    samples = []
    for _ in range(cfg["gate"]["paired_bootstrap_repetitions"]):
        picked = rng.choice(ids, size=len(ids), replace=True)
        samples.append(sum(sg[i][0] for i in picked) / sum(sg[i][1] for i in picked) -
                       sum(pg[i][0] for i in picked) / sum(pg[i][1] for i in picked))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    threshold = max(cfg["gate"]["minimum_local_gain_nats_per_token"],
                    cfg["gate"]["null_floor_multiplier"] * epsilon_null)
    return {"status": "PASS" if delta > threshold and lower > 0 else "NO_LOCAL_SIGNAL",
            "delta_nats_per_token": delta, "delta_local": threshold,
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "distinct_documents": len(ids), "epsilon_null": epsilon_null}


def _gate_file(cfg, stage, verdict, next_action, fp, candidate_hash=None, reasons=None, **extra):
    gate = {"protocol_id": cfg["protocol_id"], "stage": stage, "verdict": verdict,
            "next_action": next_action, "candidate_hash": candidate_hash,
            "reasons": reasons or [], **fp, **extra,
            "method_novelty": "NOT_ESTABLISHED", "end_to_end_gain": "NOT_TESTED",
            "deployment_cost": "NOT_TESTED",
            "nearest_neighbor_comparison": "LOCAL_DIRECTION_CONTROL_ONLY"}
    atomic_json(ROOT / "results" / "local_gate.json", gate)
    return gate


def pilot(cfg, cfg_hash, stage):
    binding, data, mem = require_data(cfg, cfg_hash, formal=True)
    resources = resource_check(cfg)
    smoke = read_json(ROOT / "artifacts" / "smoke_run_manifest.json")
    fp = fingerprint(cfg_hash, data["data_hash"])
    if smoke.get("status") != "PASS" or any(smoke.get(k) != v for k, v in fp.items()):
        raise RuntimeError("P0_FAIL: matching successful smoke/replay artifact required")
    eval_data = read_json(ROOT / "artifacts" / "eval_blocks.json")
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with run_lock(f"pilot_{stage}"):
        if stage == "discover":
            seed = cfg["baseline"]["discovery_seed"]
            runs = [baseline_run(cfg, cfg_hash, binding, data, mem, eval_data, seed, peak,
                                 resources["process_vram_limit"])
                    for peak in cfg["baseline"]["peak_lr_grid"]]
            best = min(runs, key=lambda r: (r["lr_select_nll"],
                                           0 if r["peak_lr"] == cfg["baseline"]["peak_lr_grid"][1] else 1,
                                           r["peak_lr"]))
            peak = best["peak_lr"]
            lr_unresolved = peak in (cfg["baseline"]["peak_lr_grid"][0],
                                     cfg["baseline"]["peak_lr_grid"][-1])
            cells = []
            for checkpoint_step in cfg["probe"]["diagnostic_checkpoint_steps"]:
                for role in cfg["probe"]["diagnostic_module_roles"]:
                    cells.append((checkpoint_step, role))
            main_rows = None
            for step, role in cells:
                selected = probe_cell(cfg, cfg_hash, binding, data, mem, eval_data,
                                      seed=seed, peak=peak, step=step, role=role,
                                      split="probe_select")
                if step == cfg["probe"]["primary"]["checkpoint_step"] and \
                        role == cfg["probe"]["primary"]["module_role"]:
                    main_rows = selected
            if main_rows is None:
                raise RuntimeError("primary cell missing")
            null = read_json(ROOT / "artifacts" / "numerics" / "identity_probe.json")
            epsilon_null = null["epsilon_null"]
            spectral = _select(main_rows, "spectral", epsilon_null)
            scalar = _select(main_rows, "scalar", epsilon_null)
            spectra = _select(main_rows, "spectra_diagnostic", epsilon_null)
            frozen = {"protocol_id": cfg["protocol_id"], "seed": seed, "peak_lr": peak,
                      "primary_step": 256, "spectral": spectral, "scalar": scalar,
                      "spectra_diagnostic": spectra, "epsilon_null": epsilon_null, **fp}
            frozen["candidate_hash"] = sha_object(frozen)
            atomic_json(ROOT / "results" / "candidate_frozen.json", frozen)
            confirm_rows = probe_cell(cfg, cfg_hash, binding, data, mem, eval_data,
                                      seed=seed, peak=peak, step=256, role="ffn_down_projection",
                                      split="probe_confirm", frozen=[spectral, scalar, spectra])
            stat = _paired_gate(confirm_rows, scalar, spectral, cfg, epsilon_null)
            scale_grid = cfg["probe"]["scalar_control_scale_grid"]
            scale_unresolved = scalar["scale"] in (scale_grid[1], scale_grid[-1])
            verdict = ("BASELINE_TUNING_UNRESOLVED" if lr_unresolved else
                       "INCONCLUSIVE_SCALE_RANGE" if scale_unresolved else stat["status"])
            next_action = "RUN_CONFIRM" if verdict == "PASS" else "STOP"
            if verdict == "PASS":
                verdict = "DISCOVERY_PASS"
            return _gate_file(cfg, stage, verdict, next_action, fp,
                              frozen["candidate_hash"], [stat.get("status")],
                              chosen_peak_lr=peak, primary_statistics=stat)
        if stage == "confirm":
            gate = read_json(ROOT / "results" / "local_gate.json")
            frozen = read_json(ROOT / "results" / "candidate_frozen.json")
            if gate.get("stage") != "discover" or gate.get("verdict") != "DISCOVERY_PASS" or \
                    gate.get("next_action") != "RUN_CONFIRM" or gate.get("candidate_hash") != frozen.get("candidate_hash") or \
                    any(gate.get(k) != v for k, v in fp.items()):
                raise RuntimeError("confirm gate/frozen candidate mismatch; refusing training")
            seed = cfg["baseline"]["confirmation_seed"]
            peak = frozen["peak_lr"]
            baseline_run(cfg, cfg_hash, binding, data, mem, eval_data, seed, peak,
                         resources["process_vram_limit"])
            select_rows = probe_cell(cfg, cfg_hash, binding, data, mem, eval_data,
                                     seed=seed, peak=peak, step=256, role="ffn_down_projection",
                                     split="probe_select")
            scalar = _select(select_rows, "scalar", frozen["epsilon_null"])
            spectra = _select(select_rows, "spectra_diagnostic", frozen["epsilon_null"])
            selected = [frozen["spectral"], scalar, spectra, frozen["scalar"]]
            confirm_rows = probe_cell(cfg, cfg_hash, binding, data, mem, eval_data,
                                      seed=seed, peak=peak, step=256, role="ffn_down_projection",
                                      split="probe_confirm", frozen=selected)
            stat = _paired_gate(confirm_rows, scalar, frozen["spectral"], cfg,
                                frozen["epsilon_null"])
            grid = cfg["probe"]["scalar_control_scale_grid"]
            verdict = "INCONCLUSIVE_SCALE_RANGE" if scalar["scale"] in (grid[1], grid[-1]) else \
                      "GO_SHORT_BRANCH_ONLY" if stat["status"] == "PASS" else "NO_TRANSFER"
            return _gate_file(cfg, stage, verdict,
                              "STOP_FOR_REVIEW" if verdict == "GO_SHORT_BRANCH_ONLY" else "STOP",
                              fp, frozen["candidate_hash"], [stat.get("status")],
                              primary_statistics=stat, confirmation_scalar=scalar,
                              confirmation_spectra_diagnostic=spectra)
        raise ValueError(stage)
