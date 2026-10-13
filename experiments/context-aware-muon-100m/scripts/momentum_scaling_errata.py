"""Momentum-scaling equivalence errata check (ZERO GPU, CPU-only).

Question (task-mandated): are
  Keller/torch:  B_t = mu*B_{t-1} + (1-mu)g_t ;  X_K = (1-mu)g_t + mu*B_t
  Moonlight:     C_t = mu*C_{t-1} + g_t      ;  X_M = g_t + mu*C_t
exactly related by B_t = (1-mu)C_t and X_K = (1-mu)X_M (same grads, same mu,
zero init)? And does NS5 normalization remove the (1-mu) factor such that
per-step updates and training trajectories coincide up to bf16/eps?

Method:
  E1 analytic identity (fp64 buffers, T=64 random grad stream)
  E2 vendored ns5(X_K) vs mirror zeropower(X_M): fp64 cos / norm ratio
     + control ns5(X_K) vs ns5(X_M) (isolates mirror-vs-torch NS5 plumbing
       from the momentum scaling itself)
  E3 per-step parameter trajectories: torch.optim.Muon(adjust_lr_fn=
     match_rms_adamw) vs mirror Muon (same 0.2*sqrt(max) scaling), same
     grads/lr/wd — relative param diff over 64 steps (expect bf16-level,
     non-systematic)
  E4 short REAL training trajectory: MiniMind tiny config (2-layer), T=60
     steps, CPU fp32 autocast-free, torch-Muon(match_rms)+auxAdamW vs
     mirrorMuon(+its internal AdamW): loss curves + final param rel-diff.

Read-only imports: mirror optimizer from audits staging (no modification).
Writes results/momentum_errata.json. Interpretation gate: differences beyond
bf16/eps => must be located in implementation ops, NOT attributed to
'momentum mechanism'.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_aware_muon.ns5 import ns5 as ns5_vendored  # torch semantics

MIRROR = Path(
    "/home/b205/projects/training_dynamics_lab/audits/context_muon_baselines_20261012/staging/vendor/moonlight_muon_llamafactory.py"
)


def load_mirror():
    spec = importlib.util.spec_from_file_location("moonlight_mirror", str(MIRROR))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    out = {}
    mu = 0.95
    T = 64
    torch.manual_seed(11)
    shapes = [(2432, 768), (768, 2432), (512, 512)]

    # ---------------- E1 analytic identities (fp64) ----------------
    e1 = {}
    for shape in shapes:
        B = torch.zeros(*shape, dtype=torch.float64)
        C = torch.zeros(*shape, dtype=torch.float64)
        max_buf = 0.0
        max_x = 0.0
        for t in range(T):
            g = torch.randn(*shape, dtype=torch.float64)
            B = mu * B + (1 - mu) * g
            C = mu * C + g
            X_K = (1 - mu) * g + mu * B
            X_M = g + mu * C
            max_buf = max(max_buf, float((B - (1 - mu) * C).abs().max() / C.abs().max()))
            max_x = max(max_x, float((X_K - (1 - mu) * X_M).abs().max() / X_M.abs().max()))
        e1[str(shape)] = {"rel_err_B_vs_1m1C": max_buf, "rel_err_XK_vs_1m1XM": max_x}
    out["E1_analytic_fp64"] = e1
    print("E1", json.dumps(e1))

    # ---------------- E2 NS5 outputs: vendored vs mirror ----------------
    mirror = load_mirror()
    e2 = {}
    for shape in shapes:
        torch.manual_seed(12)
        B = torch.zeros(*shape)
        C = torch.zeros(*shape)
        cos_vm, ratio_vm = [], []
        cos_ctrl, ratio_ctrl = [], []
        for t in range(16):
            g = torch.randn(*shape)
            B = mu * B + (1 - mu) * g          # fp32 EMA (Keller semantics)
            C = mu * C + g                     # fp32 classic (Moonlight)
            X_K = (1 - mu) * g + mu * B
            X_M = g + mu * C
            o_K = ns5_vendored(X_K).double()
            o_M_mirror = mirror.zeropower_via_newtonschulz5(X_M, steps=5).double()
            o_M_ven = ns5_vendored(X_M).double()

            def cos(a, b):
                return float((a * b).sum() / (a.norm() * b.norm()))

            cos_vm.append(cos(o_K, o_M_mirror))
            ratio_vm.append(float(o_K.norm() / o_M_mirror.norm()))
            cos_ctrl.append(cos(o_K, o_M_ven))
            ratio_ctrl.append(float(o_K.norm() / o_M_ven.norm()))
        e2[str(shape)] = {
            "cos_torchK_vs_mirrorM_min": min(cos_vm),
            "norm_ratio_torchK_vs_mirrorM_mean": sum(ratio_vm) / len(ratio_vm),
            "cos_torchK_vs_ns5M_min": min(cos_ctrl),
            "norm_ratio_torchK_vs_ns5M_mean": sum(ratio_ctrl) / len(ratio_ctrl),
        }
    out["E2_ns5_outputs"] = e2
    print("E2", json.dumps(e2))

    # ---------------- E3 optimizer trajectories (matrix-level) ----------------
    from torch.optim import Muon as TorchMuon

    e3 = {}
    for shape in [(2432, 768), (768, 2432)]:
        lr, wd = 3e-3, 0.02
        torch.manual_seed(13)
        p1 = torch.randn(*shape)
        p2 = p1.clone()
        o1 = TorchMuon([p1], lr=lr, weight_decay=wd, momentum=mu, nesterov=True,
                       adjust_lr_fn="match_rms_adamw")
        o2 = mirror.Muon(lr=lr, wd=wd, muon_params=[p2], momentum=mu, nesterov=True,
                         ns_steps=5)
        rel = []
        for t in range(T):
            torch.manual_seed(1000 + t)
            g = torch.randn(*shape)
            p1.grad = g.clone()
            p2.grad = g.clone()
            o1.step()
            o2.step()
            rel.append(float((p1 - p2).norm() / p1.norm()))
        e3[str(shape)] = {"final_rel_param_diff": rel[-1], "max_rel_param_diff": max(rel)}
    out["E3_optimizer_matrix_trajectory"] = e3
    print("E3", json.dumps(e3))

    # ---------------- E4 short real-model training trajectory ----------------
    sys.path.insert(0, str(ROOT.parent / "minimind_base"))
    import importlib.util as ilu

    spec = ilu.spec_from_file_location(
        "mm_model", str(ROOT.parent / "minimind_base" / "model" / "model_minimind.py")
    )
    mm = ilu.module_from_spec(spec)
    spec.loader.exec_module(mm)
    cfg = mm.MiniMindConfig(
        hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=384, vocab_size=6400,
        dropout=0.0, flash_attn=False,
    )
    dev = "cpu"
    T4 = 60
    loss_curves = {}

    def run(which):
        torch.manual_seed(3471)
        model = mm.MiniMindForCausalLM(cfg).to(dev)
        model.train()
        muon_params, aux_params = [], []
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                aux_params.append(p)
        lr, wd = 2e-3, 0.02
        if which == "torch":
            opt1 = TorchMuon(muon_params, lr=lr, weight_decay=wd, momentum=mu,
                             nesterov=True, adjust_lr_fn="match_rms_adamw")
            opt2 = torch.optim.AdamW(aux_params, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                                     weight_decay=wd)
        else:
            opt1 = mirror.Muon(lr=lr, wd=wd, muon_params=muon_params,
                               adamw_params=aux_params, momentum=mu, nesterov=True,
                               ns_steps=5)
            opt2 = None
        torch.manual_seed(777)
        data = torch.randint(0, 6400, (T4, 4, 64))
        losses = []
        for t in range(T4):
            x, y = data[t], data[t]
            if which == "torch":
                for o in (opt1, opt2):
                    for g_ in o.param_groups:
                        g_["lr"] = lr
                out_ = model(input_ids=x, labels=y).loss
                out_.backward()
                opt1.step()
                opt2.step()
                opt1.zero_grad(set_to_none=True)
                opt2.zero_grad(set_to_none=True)
            else:
                for g_ in opt1.param_groups:
                    g_["lr"] = lr
                out_ = model(input_ids=x, labels=y).loss
                out_.backward()
                opt1.step()
                opt1.zero_grad(set_to_none=True)
            losses.append(float(out_))
        fp = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return losses, fp

    lA, fpA = run("torch")
    lB, fpB = run("mirror")
    per_param = []
    for k in fpA:
        if fpA[k].ndim == 2 and "embed" not in k:
            per_param.append(float((fpA[k] - fpB[k]).norm() / fpA[k].norm()))
    e4 = {
        "loss_torch_first_last": [lA[0], lA[-1]],
        "loss_mirror_first_last": [lB[0], lB[-1]],
        "max_abs_loss_diff_after_step1": max(abs(a - b) for a, b in zip(lA[1:], lB[1:])),
        "final_rel_param_diff_muon_matrices_max": max(per_param),
    }
    out["E4_training_trajectory_tiny_cpu"] = e4
    print("E4", json.dumps(e4))

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "momentum_errata.json").write_text(json.dumps(out, indent=1))

    # verdict gates
    ok_e1 = all(v["rel_err_B_vs_1m1C"] < 1e-12 and v["rel_err_XK_vs_1m1XM"] < 1e-12 for v in e1.values())
    ok_e2 = all(v["cos_torchK_vs_mirrorM_min"] > 0.999 and abs(v["norm_ratio_torchK_vs_mirrorM_mean"] - 1) < 0.05 for v in e2.values())
    print("VERDICT_E1_exact_scalar_identity:", ok_e1)
    print("VERDICT_E2_ns5_removes_scalar:", ok_e2)


if __name__ == "__main__":
    main()
