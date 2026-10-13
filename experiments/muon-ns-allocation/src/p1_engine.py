"""P1 engine: one-step candidate evaluation for a state/batch set.

evaluate_batch(state_path, step_index_in_stream, q, side, schemes) ->
  per-scheme V plus geometry/nu/gamma. Schemes are dicts:
    {"kind": "native"}
    {"kind": "group", "group": g, "K": K, "probe": "raw"|"gain_only_live"|"direction_only"}
    {"kind": "global_amp", "scale": s}
    {"kind": "joint", "K_vector": {...}, "probe": "raw"|"gain_only_live"|"direction_only"|"gamma_const",
     "gamma_const": {g: c} (only for gamma_const)}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition.mixture import MixtureStream
from muon_transition.routing import route_parameters
from src.recipe_registry import GROUPS, lr_at, group_of_param
from src.state_restore import load_anchor, eval_V
from src.group_probe import (collect_G, compute_H, ns_cache, u_of, nu_g,
                             geometry_E, group_E2, apply_candidate, restore_theta)


def momenta_from_payload(payload, model):
    _, _, table = route_parameters(model)
    muon_names = [t["name"] for t in table if t["route"] == "muon"]
    state = payload["optimizer_states"][0]["state"]
    return {nm: state[idx]["momentum_buffer"].to("cuda") for idx, nm in enumerate(muon_names)}


def names_by_group(model):
    out = {g: [] for g in GROUPS}
    for name, p in model.named_parameters():
        g = group_of_param(name)
        if g is not None:
            out[g].append(name)
    return out


def evaluate_batch(state_path, global_step, batch_slice, q, side, schemes, repeat_native=False,
                   independent_check=False):
    """Runs on cuda. `global_step` = restore-point step; probe LR = lr_at(global_step+1).

    Audit-hardened (ROUND3 lessons): grad-hash before/after on the tensors actually
    consumed; optional independent second collection to verify the cached-G path; AdamW
    step aliasing assert after load.
    """
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    model, opts = load_anchor(payload)
    # AdamW 'step' aliasing check across loads (audit 3.5): compare with a second restore
    _, opts_b = load_anchor(payload)
    for oa, ob in zip(opts, opts_b):
        for ka, kb in zip(oa.state.keys(), ob.state.keys()):
            va, vb = oa.state[ka].get("step"), ob.state[kb].get("step")
            if torch.is_tensor(va) and torch.is_tensor(vb):
                assert va.data_ptr() != vb.data_ptr(), "optimizer step tensor aliased across restores"
    del opts_b
    momenta = momenta_from_payload(payload, model)
    nbg = names_by_group(model)
    name_to_param = dict(model.named_parameters())
    lr = lr_at(global_step + 1)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    from muon_transition.mixture import build_step_manifest
    steps = build_step_manifest(a_train, b_train, 0.5, batch_slice[1], seed=2030)["steps"][batch_slice[0]:batch_slice[1]]
    stream = MixtureStream({"steps": steps}, a_train, b_train)

    import hashlib
    def grad_hash():
        h = hashlib.sha256()
        for name, p in model.named_parameters():
            if p.grad is not None:
                h.update(p.grad.detach().contiguous().cpu().numpy().tobytes())
        return h.hexdigest()

    info = collect_G(model, stream, 0, autocast)
    g_hash_before = grad_hash()
    indep_ok = None
    if independent_check:
        info2 = collect_G(model, stream, 0, autocast)  # independent second fwd/bwd at theta*
        indep_ok = (grad_hash() == g_hash_before)  # determinism of the collection path
        info = info2
    H = compute_H(model, momenta)
    nsc = ns_cache(H)
    E_by_name = geometry_E(nsc)
    # native one-step (production optimizer path)
    opts[0].step()
    opts[1].step()
    assert grad_hash() == g_hash_before, "grads mutated by optimizer step (aliasing!)"
    theta_plus = {n: p.detach().clone() for n, p in model.named_parameters() if group_of_param(n)}
    u5 = u_of(H, nsc, {}, lr)
    gam = {(g, K): nu_g(u_of(H, nsc, {g: K}, lr), nbg[g]) / nu_g(u5, nbg[g])
           for g in GROUPS for K in (3, 7)}
    nu5 = {g: nu_g(u5, nbg[g]) for g in GROUPS}

    rows = []
    for sch in schemes:
        kind = sch["kind"]
        if kind == "native":
            restore_theta(name_to_param, theta_plus)
            res = eval_V(model, q, side)
            rows.append({"scheme": sch.get("label", "native"), **res})
            if sch.get("label", "native") == "native" and repeat_native:
                restore_theta(name_to_param, theta_plus)
                rows.append({"scheme": "native_repeat", **eval_V(model, q, side)})
            continue
        if kind == "group":
            g, K, probe = sch["group"], sch["K"], sch["probe"]
            gamma = gam[(g, K)]
            uK = u_of(H, nsc, {g: K}, lr)
            if probe == "raw":
                u_use = uK
            elif probe == "gain_only_live":
                u_use = {n: u5[n] * gamma for n in nbg[g]}
            elif probe == "direction_only":
                u_use = {n: (uK[n] / gamma if gamma > 0 else uK[n]) for n in nbg[g]}
            apply_candidate(name_to_param, theta_plus, u5, {g: (u_use, nbg[g])})
            res = eval_V(model, q, side)
            restore_theta(name_to_param, theta_plus)
            rows.append({"scheme": f"{g}|K{K}|{probe}", "group": g, "K": K, "probe": probe,
                         "gamma": gamma, "nu_g5": nu5[g],
                         "E_g2_5": group_E2(E_by_name, H, g, 5),
                         "E_g2_K": group_E2(E_by_name, H, g, K), **res})
            continue
        if kind == "global_amp":
            s = sch["scale"]
            u_use = {n: u5[n] * s for n in name_to_param if group_of_param(n)}
            apply_candidate(name_to_param, theta_plus, u5,
                            {"__all__": (u_use, [n for n in name_to_param if group_of_param(n)])})
            res = eval_V(model, q, side)
            restore_theta(name_to_param, theta_plus)
            rows.append({"scheme": f"global_amp|x{s}", "scale": s, **res})
            continue
        if kind == "joint":
            kv = dict(sch["K_vector"])
            probe = sch.get("probe", "raw")
            uK = u_of(H, nsc, kv, lr)
            groups_changed = [g for g in GROUPS if kv.get(g, 5) != 5]
            u_groups = {}
            for g in groups_changed:
                if probe == "raw":
                    u_groups[g] = {n: uK[n] for n in nbg[g]}
                elif probe == "gain_only_live":
                    u_groups[g] = {n: u5[n] * gam[(g, kv[g])] for n in nbg[g]}
                elif probe == "direction_only":
                    gg = gam[(g, kv[g])]
                    u_groups[g] = {n: (uK[n] / gg if gg > 0 else uK[n]) for n in nbg[g]}
                elif probe == "gamma_const":
                    u_groups[g] = {n: u5[n] * sch["gamma_const"][g] for n in nbg[g]}
            apply_candidate(name_to_param, theta_plus, u5,
                            {g: (u_groups[g], nbg[g]) for g in groups_changed})
            res = eval_V(model, q, side)
            restore_theta(name_to_param, theta_plus)
            rows.append({"scheme": sch.get("label", "joint|" + probe),
                         "K_vector": kv, "probe": probe, **res})
            continue
    geom_rows = {g: {K: group_E2(E_by_name, H, g, K) for K in (3, 5, 7)} for g in GROUPS}
    return rows, {"grad_info": info, "lr_probe": lr, "E_g2": geom_rows,
                  "independent_recollection_ok": indep_ok,
                  "grad_hash_unchanged_through_step": True,
                  "gamma": {f"{g}|{K}": gam[(g, K)] for g in GROUPS for K in (3, 7)},
                  "nu5": nu5}


def kj_schemes():
    sch = [{"kind": "native"}, {"kind": "global_amp", "scale": 0.5}, {"kind": "global_amp", "scale": 2.0}]
    for g in GROUPS:
        for K in (3, 7):
            for probe in ("raw", "gain_only_live", "direction_only"):
                sch.append({"kind": "group", "group": g, "K": K, "probe": probe})
    return sch
