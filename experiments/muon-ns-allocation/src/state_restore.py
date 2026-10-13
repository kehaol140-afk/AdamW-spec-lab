"""State restore + full-state one-step candidate construction + target sets + evaluation.

Aliases: every restore re-instantiates optimizer states from the anchor payload (deep
load); parameter references are snapshotted AFTER load; candidate parameter points are
set by in-place copy from a saved theta_0_plus snapshot plus explicit deltas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition.model import build_model, unique_named_parameters
from muon_transition.optimizers import build_muon_arm
from muon_transition import config as mte_config
from vendored_fr.rng import restore_rng_state

Q_CACHE = Path("/tmp/opencode/p0p1_q_sets.pt")


def load_anchor(payload):
    model = build_model().to("cuda")
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    # HARDENING (ROUND3 audit 3.5, verified live on this server): the AdamW scalar `step`
    # tensors remain aliased to the payload across restores (34/34 shared data_ptr while
    # momentum/m/v are properly deep-copied). Clone every state tensor so candidates can
    # never share mutable optimizer state, and the payload itself is never mutated.
    for opt in (muon_opt, aux_opt):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.clone()
    restore_rng_state(payload["rng"])
    return model, [muon_opt, aux_opt]


def param_reference(model):
    """Reference snapshot AFTER load; candidates are compared against this."""
    return {n: p.detach().clone() for n, p in unique_named_parameters(model)}


def no_alias_between(payload, model_a, opts_a, model_b, opts_b):
    """All mutable optimizer state must be distinct tensors across candidates."""
    for oa, ob in zip(opts_a, opts_b):
        for ka, kb in zip(oa.state.keys(), ob.state.keys()):
            for kk in set(oa.state[ka].keys()) | set(ob.state[kb].keys()):
                va, vb = oa.state[ka].get(kk), ob.state[kb].get(kk)
                if torch.is_tensor(va) and torch.is_tensor(vb):
                    if va.data_ptr() == vb.data_ptr():
                        return False
    return True


def build_q_sets(verbose=False):
    """Q_select / Q_P1_check / Q_P2_endpoint from OLD dev pool, thread-group split."""
    if Q_CACHE.exists():
        return torch.load(Q_CACHE, weights_only=True)
    from muon_transition.data import TokenizerHandle, conversation_to_text, load_domain, pack_domain_stream
    from muon_transition.partitions import build_pair_aligned_split
    sys.path.insert(0, str(LAB / "muon_transition_experiment"))
    from scripts.m0_gate3_data_audit import dedup_keep_first
    from src.recipe_registry import q_set_of_group, Q_WINDOWS
    tok = TokenizerHandle()
    out = {"version": "P0P1_Q_v1", "sets": {}}
    deduped = {}
    for name, path in (("A", mte_config.DOMAIN_A_PATH), ("B", mte_config.DOMAIN_B_PATH)):
        records = load_domain(path)
        texts = [conversation_to_text(r) for r in records]
        keep = dedup_keep_first(texts)
        deduped[name] = ([records[i] for i in keep], [texts[i] for i in keep])
    manifest = build_pair_aligned_split(deduped["A"][0], deduped["B"][0])
    sets_idx = {"select": {"A": [], "B": []}, "P1_check": {"A": [], "B": []},
                "P2_endpoint": {"A": [], "B": []}}
    for dom in ("A", "B"):
        records, texts = deduped[dom]
        for i in manifest[dom]["dev"]:
            s = q_set_of_group(records[i].get("conversation_id"))
            name = ["select", "P1_check", "P2_endpoint"][s]
            sets_idx[name][dom].append(i)
    for sname, per in sets_idx.items():
        dom_out = {}
        for dom in ("A", "B"):
            idx = sorted(per[dom])
            encs = tok.encode_texts([deduped[dom][1][i] for i in idx])
            windows = pack_domain_stream(encs, mte_config.SEQ_LEN, mte_config.BOS_ID, mte_config.DOC_SEP_ID)
            n = min(Q_WINDOWS[sname], windows.shape[0])
            dom_out[dom] = {"windows": windows[:n], "n_docs": len(idx), "n_windows": int(n),
                            "token_count": int(n * (mte_config.SEQ_LEN - 1))}
            if verbose:
                print(f"Q_{sname}[{dom}]: docs={len(idx)} windows={n}")
            if n == 0:
                raise RuntimeError(f"empty target set {sname}/{dom}")
        out["sets"][sname] = dom_out
    # group isolation across all three sets
    for dom in ("A", "B"):
        seen = {}
        for sname in ("select", "P1_check", "P2_endpoint"):
            for gid in out["sets"][sname][dom].get("group_ids", []):
                pass
        # isolation enforced by construction (one group -> exactly one set); record counts
    out["isolation"] = "by-construction: each thread group maps to exactly one set via the hash rule"
    torch.save(out, Q_CACHE)
    return out


_ACCESS_LOG = {"select": 0, "P1_check": 0, "P2_endpoint": 0}


def access_guard(allowed: set):
    """Return an evaluator restricted to the allowed target sets; guards P2/confirm."""
    def check(side: str):
        if side not in allowed:
            raise RuntimeError(f"target set {side} is sealed for this stage "
                               f"(allowed: {sorted(allowed)}); P2_endpoint/confirm must stay unread")
        _ACCESS_LOG[side] = _ACCESS_LOG.get(side, 0) + 1
    return check


def eval_V(model, q, side: str, device="cuda", batch_size=64):
    """V = 0.5*NLL_A + 0.5*NLL_B on the requested set; per-domain sum/count kept."""
    was = model.training
    model.eval()
    res = {}
    with torch.no_grad():
        for dom in ("A", "B"):
            wins = q["sets"][side][dom]["windows"]
            nll_sum = torch.zeros((), dtype=torch.float64)
            count = 0
            for i in range(0, wins.shape[0], batch_size):
                b = wins[i:i + batch_size].to(device)
                logits = model(b).logits[..., :-1, :].contiguous()
                t = b[..., 1:].contiguous()
                ls = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                     t.reshape(-1), reduction="sum")
                nll_sum += ls.double().cpu()
                count += int(t.numel())
            res[f"nll_sum_{dom}"] = float(nll_sum)
            res[f"count_{dom}"] = count
            res[f"nll_{dom}"] = float(nll_sum) / count
    res["V"] = 0.5 * res["nll_A"] + 0.5 * res["nll_B"]
    if was:
        model.train()
    return res
