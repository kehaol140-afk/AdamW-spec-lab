"""R2F main runner: L branches -> D_TRIGGER freeze -> 8 crossover branches ->
conditional D. Per-run persistent logs, COMPLETE certificates (load-verified +
hash), resume without overwriting other runs, base-rebuilt recipes each step."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition.model import build_model
from muon_transition.optimizers import build_muon_arm
from muon_transition.mixture import MixtureStream, build_step_manifest
from vendored_fr.rng import restore_rng_state, capture_rng_state
from src.state_restore import eval_V, build_q_sets
from src.r2f_recipe import RECIPES, effective

OUT = ROOT / "outputs" / "r2f_lrwd_retention_v1"
CORR = ROOT / "outputs" / "r2f_corrected_v1"
CORR.mkdir(parents=True, exist_ok=True)
R2DIR = ROOT / "outputs" / "r2_amplitude_continuation_v1"
PARENT = ROOT / "outputs" / "p0_p1_v1_1" / "parent"


def immutable_base_wds():
    """FIX (CONFIRMED_WRONG_DECAY): base WD binds to the OPTIMIZER IDENTITY from the
    immutable base_group_manifest (verified against the ORIGINAL parent checkpoints:
    muon wd = aux wd = 0.02). Source checkpoints restore STATE ONLY (model/buffers/
    moments/m/v/step/RNG/cursor) — their saved effective LR/WD are historical records
    of that branch's own recipe and must NEVER seed a new branch's base."""
    manifest = json.loads((OUT / "base_group_manifest.json").read_text())
    base = {g["group_id"]: g["base_wd"] for g in manifest["groups"]}
    for step in (512, 1536):
        b = torch.load(PARENT / f"parent_state_step{step}.pt", map_location="cpu", weights_only=False)
        for gi, key in ((0, "muon"), (1, "aux_adamw")):
            expected = base[key]
            for grp in b["optimizer_states"][gi]["param_groups"]:
                assert abs(grp["weight_decay"] - expected) < 1e-15, \
                    f"base manifest disagrees with parent artifact p{step}:{key}"
    return {"muon": base["muon"], "aux": base["aux_adamw"]}
CONFIG_SHA = hashlib.sha256((ROOT / "configs" / "r2f_executed.json").read_bytes()).hexdigest() \
    if (ROOT / "configs" / "r2f_executed.json").exists() else "PENDING"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model().to("cuda")
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    for opt in (muon_opt, aux_opt):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.clone()
    restore_rng_state(payload["rng"])
    # FIX: base WD NEVER comes from the source payload (AH sources save 0.04 as their
    # own historical effective value). Callers use immutable_base_wds().
    return model, [muon_opt, aux_opt], payload


def run_branch(run_id, src_path, start_step, recipe, n_steps, stream, q, logf):
    """One real continuation branch; per-run train.jsonl; COMPLETE certificate."""
    complete_marker = OUT / f"{run_id}.complete.json"
    if complete_marker.exists():
        return {"run_id": run_id, "status": "already_complete"}
    t0 = time.time()
    model, opts, payload = load_full(src_path)
    _bw = immutable_base_wds()
    base_wds = (_bw["muon"], _bw["aux"])
    src_sha = sha256_file(src_path)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    r_m = RECIPES[recipe]["r"]
    d_m = RECIPES[recipe]["d"]
    r_a, d_a = r_m, d_m  # both optimizers share the recipe's r/d
    logl = open(OUT / f"{run_id}_train.jsonl", "w")
    snapshots = {}
    for k in range(1, n_steps + 1):
        gstep = start_step + k
        base_lr = __import__("src.recipe_registry", fromlist=["lr_at"]).lr_at(gstep)
        # per-group effective values REBUILT from base each step
        eff = []
        for opt, r_, d_, bwd in ((opts[0], r_m, d_m, base_wds[0]), (opts[1], r_a, d_a, base_wds[1])):
            elr, ewd = effective(base_lr, bwd, recipe)
            for group in opt.param_groups:
                group["lr"] = elr
                group["weight_decay"] = ewd
            eff.append((elr, ewd))
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(k - 1):
            batch = mb.to("cuda")
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro.append(float(loss.detach()))
        preclip = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0)
        opts[0].step()
        opts[1].step()
        rec = {"run_id": run_id, "source_checkpoint_sha": src_sha, "global_step": gstep,
               "config_sha": CONFIG_SHA, "recipe": recipe, "train_loss": float(np.mean(micro)),
               "preclip_norm": float(preclip), "base_lr": base_lr,
               "eff_lr_muon": eff[0][0], "eff_wd_muon": eff[0][1],
               "decay_product_muon": eff[0][0] * eff[0][1],
               "elapsed_s": time.time() - t0}
        logl.write(json.dumps(rec) + "\n")
        if k in (1, 16, 64):
            snapshots[k] = {n_: p.detach().cpu().clone() for n_, p in model.named_parameters()}
        if k in (1, 16, 64, 128, 256):
            res = eval_V(model, q, "select")
            assert model.training
            logl.write(json.dumps({"run_id": run_id, "monitor_rel_step": k, **res}) + "\n")
    logl.close()
    # snapshots: 1/16/64 model state; 256 full state
    for k, snap in snapshots.items():
        torch.save({"model_state": snap, "meta": {"run_id": run_id, "rel_step": k}},
                   OUT / f"{run_id}_step{k}_model.pt")
    torch.save({"model_state": model.state_dict(),
                "optimizer_states": [o.state_dict() for o in opts],
                "rng": capture_rng_state(),
                "cursor": {"step": start_step + n_steps, "run_id": run_id},
                "meta": {"recipe": recipe, "start_step": start_step}},
               OUT / f"{run_id}_step256_full.pt.tmp")
    # COMPLETE certificate: verify loadable, then atomic replace
    blob = torch.load(OUT / f"{run_id}_step256_full.pt.tmp", map_location="cpu", weights_only=False)
    assert blob["cursor"]["step"] == start_step + n_steps
    (OUT / f"{run_id}_step256_full.pt.tmp").rename(OUT / f"{run_id}_step256_full.pt")
    json.dump({"run_id": run_id, "recipe": recipe, "start_step": start_step,
               "n_steps": n_steps, "source_sha": src_sha,
               "final_sha": sha256_file(OUT / f"{run_id}_step256_full.pt"),
               "final_loss": float(np.mean(micro)), "wall_s": time.time() - t0},
              open(complete_marker, "w"), indent=1)
    logf.write(f"[{time.time():.0f}] {run_id} COMPLETE loss={np.mean(micro):.4f}\n")
    logf.flush()
    return {"run_id": run_id, "status": "complete"}


def make_stream(start_step, n_steps):
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    m = build_step_manifest(a_train, b_train, 0.5, start_step + n_steps, seed=2030)
    return MixtureStream({"steps": m["steps"][start_step:start_step + n_steps]}, a_train, b_train)


def first_step_gates(q):
    """Protocol 3.1 real first-step checks on both parent origins (recipe N vs an
    independent native reference, bit-level; L/D data-delta & decay-product ratios)."""
    results = {}
    for step in (512, 1536):
        src = PARENT / f"parent_state_step{step}.pt"
        stream = make_stream(step, 1)
        # independent native reference (base recipe, single step)
        model, opts, _pl = load_full(src)
        _bw = immutable_base_wds()
        base_wds = (_bw["muon"], _bw["aux"])
        from src.recipe_registry import lr_at
        base_lr = lr_at(step + 1)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = base_lr
                group["weight_decay"] = base_wds[0] if opt is opts[0] else base_wds[1]
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        theta_pre = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = None
        for mb in stream.micro_batches(0):
            batch = mb.to("cuda")
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
        opts[0].step()
        opts[1].step()
        ref = {n: p.detach().clone() for n, p in model.named_parameters()}
        del model, opts
        torch.cuda.empty_cache()
        # recipe-N branch first step must match reference bit-level
        model, opts, _pl2 = load_full(src)
        _bw2 = immutable_base_wds()
        base_wds2 = (_bw2["muon"], _bw2["aux"])
        for k in range(1):
            gstep = step + 1
            blr = lr_at(gstep)
            for opt, bw in ((opts[0], base_wds2[0]), (opts[1], base_wds2[1])):
                elr, ewd = effective(blr, bw, "N")
                for group in opt.param_groups:
                    group["lr"] = elr
                    group["weight_decay"] = ewd
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            for mb in stream.micro_batches(0):
                batch = mb.to("cuda")
                with autocast:
                    logits = model(batch).logits[..., :-1, :].contiguous()
                    t = batch[..., 1:].contiguous()
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           t.reshape(-1), reduction="mean")
                (loss / 4).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
            opts[0].step()
            opts[1].step()
        maxdiff = max(float((p.detach() - ref[n]).abs().max())
                      for n, p in model.named_parameters())
        # L and D data-delta ratios vs N (Muon group, FP64)
        ratios = {}
        for rec in ("L", "D"):
            m2, o2, _pl3 = load_full(src)
            _bw3 = immutable_base_wds()
            bw2 = (_bw3["muon"], _bw3["aux"])
            for opt, bw in ((o2[0], bw2[0]), (o2[1], bw2[1])):
                elr, ewd = effective(blr, bw, rec)
                for group in opt.param_groups:
                    group["lr"] = elr
                    group["weight_decay"] = ewd
            for opt in o2:
                opt.zero_grad(set_to_none=True)
            for mb in stream.micro_batches(0):
                batch = mb.to("cuda")
                with autocast:
                    logits = m2(batch).logits[..., :-1, :].contiguous()
                    t = batch[..., 1:].contiguous()
                    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                           t.reshape(-1), reduction="mean")
                (loss / 4).backward()
            torch.nn.utils.clip_grad_norm_([p for p in m2.parameters() if p.grad is not None], 1.0)
            o2[0].step()
            o2[1].step()
            from src.recipe_registry import group_of_param
            num = den = 0.0
            for n, p in m2.named_parameters():
                if group_of_param(n) is None:
                    continue
                base = theta_pre[n] * (1 - blr * bw2[0])  # N decay base for muon coords
                d_new = (p.detach().double() - base.double()).pow(2).sum()
                d_ref = (ref[n].double() - base.double()).pow(2).sum()
                num += float(d_new)
                den += float(d_ref)
            ratios[rec] = (num ** 0.5) / (den ** 0.5) if den > 0 else None
            del m2, o2
            torch.cuda.empty_cache()
        results[str(step)] = {"N_vs_reference_maxdiff": maxdiff,
                              "L_data_delta_ratio": ratios["L"],
                              "D_data_delta_ratio": ratios["D"]}
        print(f"first-step gate s{step}: {results[str(step)]}", flush=True)
        del model, opts
        torch.cuda.empty_cache()
    return results


def main():
    t0 = time.time()
    logf = open(OUT / "runner_log.txt", "a")
    q = build_q_sets()
    gates = first_step_gates(q)
    json.dump(gates, open(OUT / "first_step_gates.json", "w"), indent=1)
    assert all(g["N_vs_reference_maxdiff"] == 0.0 for g in gates.values()), "N gate failed"

    # ---- Stage 1: L branches ----
    for step in (512, 1536):
        stream = make_stream(step, 256)
        r = run_branch(f"s{step}_L256", PARENT / f"parent_state_step{step}.pt",
                       step, "L", 256, stream, q, logf)
        print(f"[{time.time()-t0:.0f}s] {r}", flush=True)
    logf.write("STAGE_L_DONE\n"); logf.flush()

    # ---- D_TRIGGER freeze (needs L endpoint values -> analyze later; but the trigger
    # requires evaluating L endpoints NOW, before crossover) ----
    from muon_transition.model import build_model as bm
    model = bm().to("cuda")
    l_vals = {}
    for step in (512, 1536):
        blob = torch.load(OUT / f"s{step}_L256_step256_full.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        l_vals[step] = eval_V(model, q, "P2_endpoint")
    import csv as _csv
    old = {(int(r["restore_step"]), r["arm"]): float(r["V"])
           for r in _csv.DictReader(open(R2DIR / "endpoint_values.csv"))
           if r["t"] == "256" and r["arm"] in ("N", "AH")}
    b_decay = {s: old[(s, "N")] - old[(s, "AH")] - (old[(s, "N")] - l_vals[s]["V"])
               for s in (512, 1536)}  # B_decay = V_L - V_AH
    b_decay = {s: l_vals[s]["V"] - old[(s, "AH")] for s in (512, 1536)}
    d_required = all(v > 1e-3 for v in b_decay.values())
    trig = {"tau": 1e-3, "B_decay": {str(k): v for k, v in b_decay.items()},
            "V_L": {str(k): v["V"] for k, v in l_vals.items()},
            "V_AH_old": {str(k): old[(k, "AH")] for k in (512, 1536)},
            "D_required": d_required,
            "note": "frozen after L endpoints, before any crossover result"}
    json.dump(trig, open(OUT / "D_TRIGGER.json", "w"), indent=1)
    print(f"D_TRIGGER: B_decay={b_decay} required={d_required}", flush=True)

    # ---- Stage 2: eight crossover branches ----
    for step in (512, 1536):
        for src_arm in ("N", "AH"):
            src = R2DIR / f"s{step}_{src_arm}_step256_full.pt"
            for dst in ("N", "AH"):
                stream = make_stream(step + 256, 256)
                r = run_branch(f"s{step}_{src_arm}_to_{dst}", src, step + 256, dst,
                               256, stream, q, logf)
                print(f"[{time.time()-t0:.0f}s] {r}", flush=True)
    logf.write("STAGE_CROSS_DONE\n"); logf.flush()

    # ---- conditional D ----
    if d_required:
        for step in (512, 1536):
            stream = make_stream(step, 256)
            r = run_branch(f"s{step}_D256", PARENT / f"parent_state_step{step}.pt",
                           step, "D", 256, stream, q, logf)
            print(f"[{time.time()-t0:.0f}s] {r}", flush=True)
        logf.write("STAGE_D_DONE\n")
    else:
        logf.write("STAGE_D_SKIPPED (trigger false)\n")
    logf.close()
    print(f"R2F runner done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
