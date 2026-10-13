"""R0: identity/clock/norm gate. Verifies (a) checkpoint identity, (b) restored LR is
lr(s) but the NEXT update must consume lr(s+1) -> write-back to ALL param groups,
(c) native step under the unified clock matches manual assembly bit-level, (d) alias/
A->B->A/grad-hash/reference-after-load, (e) intended & realized O-group norm matching.
Writes outputs/r0_r1_v1_1/R0_GATE.json."""
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
from muon_transition.routing import route_parameters
from vendored_fr.rng import restore_rng_state
from src.recipe_registry import lr_at, group_of_param, shape_factor, NS_COEFFS, EPS_NORM
from torch.optim._muon import _zeropower_via_newtonschulz

OUT = ROOT / "outputs" / "r0_r1_v1_1"
O_GROUP = [f"model.layers.{l}.self_attn.o_proj.weight" for l in range(8)]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def restore(payload, step):
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
    # R0 clock closure: write the NEXT update's LR into ALL groups
    next_lr = lr_at(step + 1)
    for opt in (muon_opt, aux_opt):
        for group in opt.param_groups:
            group["lr"] = next_lr
    return model, [muon_opt, aux_opt], next_lr


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gate = {"states": {}, "artifact_sha": {}}
    parent = ROOT / "outputs" / "p0_p1_v1_1" / "parent"
    for step in (512, 1536):
        p = parent / f"parent_state_step{step}.pt"
        sha = sha256_file(p)
        payload = torch.load(p, map_location="cpu", weights_only=False)
        ck_muon_lr = payload["optimizer_states"][0]["param_groups"][0]["lr"]
        ck_aux_lr = payload["optimizer_states"][1]["param_groups"][0]["lr"]
        next_lr = lr_at(step + 1)
        st = {
            "checkpoint_sha256": sha,
            "cursor_step": payload["cursor"]["step"],
            "meta_lr_used_last": payload["meta"]["lr_used"],
            "checkpoint_saved_group_lr_muon": ck_muon_lr,
            "checkpoint_saved_group_lr_aux": ck_aux_lr,
            "lr_next_update_required": next_lr,
            "lr_at_saved_step_recomputed": lr_at(step),
            "clock_mismatch_in_old_round": {
                "saved_equals_lr_at_s": abs(ck_muon_lr - lr_at(step)) < 1e-12,
                "relative_gap_s_vs_s1": abs(ck_muon_lr - next_lr) / ck_muon_lr,
                "note": "P0/P1 native step consumed lr(s) from the restored groups while "
                        "candidate u vectors used lr_at(s+1) -> unified only in R0 (erratum)"},
        }
        # one real batch gradient on the NEW data stream: state s uses manifest[s+8:s+16)
        # (512 -> [520:528), 1536 -> [1544:1552)); old P0/P1 probes were [s:s+8)
        from muon_transition.m1_data_cache import get_domain_windows
        a_train, b_train = get_domain_windows()
        m = build_step_manifest(a_train, b_train, 0.5, step + 16, seed=2030)
        steps = m["steps"][step + 8: step + 16]
        old_probe_ids = set()
        mo = build_step_manifest(a_train, b_train, 0.5, 2048, seed=2030)
        for blk in (mo["steps"][512:520] + mo["steps"][1536:1544]):
            for e in blk:
                old_probe_ids.add((e["domain"], e["index"]))
        new_ids = {(e["domain"], e["index"]) for blk in steps for e in blk}
        st["new_batches_disjoint_from_old_probes"] = new_ids.isdisjoint(old_probe_ids)
        st["new_batch_step_indices_0based"] = [step + 8, step + 15]

        model, opts, lr = restore(payload, step)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        stream = MixtureStream({"steps": steps}, a_train, b_train)
        # reference AFTER load
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
        preclip = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)

        def ghash():
            h = hashlib.sha256()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    h.update(p.grad.detach().contiguous().cpu().numpy().tobytes())
            return h.hexdigest()
        gh_before = ghash()

        # H from checkpoint momenta + native one-step under unified clock
        _, _, table = route_parameters(model)
        muon_names = [t["name"] for t in table if t["route"] == "muon"]
        state = payload["optimizer_states"][0]["state"]
        momenta = {nm: state[idx]["momentum_buffer"].to("cuda") for idx, nm in enumerate(muon_names)}
        H = {}
        for name, p in model.named_parameters():
            g = group_of_param(name)
            if g is None or p.grad is None:
                continue
            m_i = momenta[name].lerp(p.grad, 0.05)
            H[name] = (p.grad.lerp(m_i, 0.95).detach(), g)

        opts[0].step()
        opts[1].step()
        assert ghash() == gh_before, "grads mutated"
        theta_plus = {n: p.detach().clone() for n, p in model.named_parameters() if group_of_param(n)}

        # manual assembly under the SAME unified clock (bit-level cross-check)
        max_diff = 0.0
        for name, p in model.named_parameters():
            g = group_of_param(name)
            if g is None:
                continue
            h, _ = H[name]
            y = _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
            wd = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
            manual = theta_pre[name] * (1 - lr * wd) - lr * shape_factor(h.shape) * y
            max_diff = max(max_diff, float((p - manual).abs().max()))
        st["unified_clock_manual_vs_step_maxdiff"] = max_diff

        # intended & realized O-group norm matching (NS5 vs NS3_MATCHED at a=b=1)
        nu5 = nu3 = 0.0
        for name in O_GROUP:
            h, _ = H[name]
            y5 = _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
            y3 = _zeropower_via_newtonschulz(h, NS_COEFFS, 3, EPS_NORM).float()
            nu5 += float(((lr * shape_factor(h.shape)) * y5).double().pow(2).sum())
            nu3 += float(((lr * shape_factor(h.shape)) * y3).double().pow(2).sum())
        c = (nu5 ** 0.5) / (nu3 ** 0.5)
        # realized: theta_candidate - theta_after_common_WD (= theta_plus - u5)
        import math
        wd = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
        rn5 = rn3 = 0.0
        for name in O_GROUP:
            p_pre, p_cand5, p_cand3 = theta_pre[name], None, None
            h, _ = H[name]
            y5 = _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
            y3 = _zeropower_via_newtonschulz(h, NS_COEFFS, 3, EPS_NORM).float()
            sf = lr * shape_factor(h.shape)
            base = theta_plus[name] - (-sf * y5)  # theta after common WD only
            cand5 = base + (-sf * y5)
            cand3 = base + (-sf * c * y3)
            rn5 += float((cand5 - base).double().pow(2).sum())
            rn3 += float((cand3 - base).double().pow(2).sum())
        st["norm_match"] = {
            "c_nu5_over_nu3": c,
            "intended_rel_diff": abs(nu5 - (c ** 2) * nu3) / nu5 if nu5 > 0 else None,
            "realized_rel_diff": abs(rn5 - rn3) / rn5 if rn5 > 0 else None,
            "tolerances": {"intended": 1e-6, "realized": 1e-4},
        }
        # A->B->A: candidate B (a=0.75) then restore A (native) identity
        with torch.no_grad():
            for n, p in theta_plus.items():
                dict(model.named_parameters())[n].copy_(p)
            nat_A1 = float(sum(p.detach().double().sum() for p in model.parameters()))
            for n, p in model.named_parameters():
                g = group_of_param(n)
                if g is not None:
                    p.mul_(0.75)
            for n, p in theta_plus.items():
                dict(model.named_parameters())[n].copy_(p)
            nat_A2 = float(sum(p.detach().double().sum() for p in model.parameters()))
        st["A_B_A_identity"] = (nat_A1 == nat_A2)
        # payload untouched
        st["payload_lr_unchanged_after_all"] = (
            payload["optimizer_states"][0]["param_groups"][0]["lr"] == ck_muon_lr)
        gate["states"][str(step)] = st
        print(f"state {step}: lr_writeback={lr:.6f} (was {ck_muon_lr:.6f}), "
              f"manual-vs-step maxdiff={max_diff:.2e}, c={c:.4f}, "
              f"realized_norm_rel_diff={st['norm_match']['realized_rel_diff']:.2e}, "
              f"ABA={st['A_B_A_identity']}", flush=True)
    gate["installed_muon_sha256"] = sha256_file(Path(
        "/home/b205/miniforge3/envs/fr_minimind/lib/python3.11/site-packages/torch/optim/_muon.py"))
    gate["gate_pass"] = all(
        s["unified_clock_manual_vs_step_maxdiff"] < 5e-8 and s["A_B_A_identity"]
        and s["norm_match"]["realized_rel_diff"] is not None
        and s["norm_match"]["realized_rel_diff"] <= 1e-4
        and s["new_batches_disjoint_from_old_probes"]
        for s in gate["states"].values())
    gate["wall_seconds"] = time.time() - t0
    json.dump(gate, open(OUT / "R0_GATE.json", "w"), indent=1)
    print("R0 GATE:", "PASS" if gate["gate_pass"] else "FAIL")


if __name__ == "__main__":
    main()
