"""P0-A: native Muon NS5 spectral baseline from frozen m1 checkpoints (offline CPU).

Usage:
  python -m specmuon.run_p0a --config configs/p0a_audit_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .io_utils import M1_DIR, file_sha256, load_m1_muon_buffers
from .metrics import (
    directional_energy_shares,
    energy_shares,
    normalized_gap,
    orthogonality_defect,
    participation_ratio,
    top1_cosine,
    top_subspace_alignment,
)
from .native_ns import (
    MUON_WHEEL_SHA256,
    converged_polar_reference,
    native_ns5_installed,
    polar_express_single_pass,
    vendored_zeropower_via_newtonschulz,
)
from .operators import bulk_spec_muon, exact_polar, head_spec_muon, muon_power, svd_basis


def module_class(name: str) -> str:
    for tag in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if tag in name:
            return tag.replace("_proj", "")
    return "other"


def layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def analyze_matrix(M: torch.Tensor, full: bool) -> dict:
    """full=True: canonical metrics incl. polar/PE references + operator preview.
    full=False: condensed."""
    t0 = time.perf_counter()
    row = {}
    U, S, Vh = svd_basis(M)
    P = U @ Vh
    V = Vh.T
    r = min(M.shape)

    Y5 = vendored_zeropower_via_newtonschulz(M.float()).float()
    Y5n = Y5.double() / Y5.double().norm()
    s5 = torch.linalg.svdvals(Y5.double())
    x = S / S.double().norm()  # sigma_i / ||M||_F

    e_in = energy_shares(S, ks=(1, 8))
    e_out = energy_shares(s5, ks=(1, 8))
    row.update(
        fro_M=float(S.double().norm()),
        sigma_max_over_fro=float(S[0] / S.double().norm()),
        in_E1=e_in[1], in_E8=e_in[8], in_PR=participation_ratio(S), r=r,
        out_E1=e_out[1], out_E8=e_out[8], out_PR=participation_ratio(s5),
        out_smax=float(s5[0]), out_smin=float(s5[-1]),
        defect_Y5=orthogonality_defect(Y5),
        norm_Y5=float(Y5.double().norm()),
        gap_Y5_polar=normalized_gap(Y5, P),
        polar_top1_cos=top1_cosine(Y5, P),
        polar_top1_cos_right=top1_cosine(Y5, P, side="right"),
        polar_top8_align=top_subspace_alignment(Y5, P, k=min(8, r)),
        x_top1=float(x[0]), x_top8=float(x[min(8, r) - 1]), x_last=float(x[-1]),
    )
    # operator preview: what each frozen variant delivers on this real matrix
    k10 = max(1, r // 10)
    T = row["norm_Y5"]
    arms = {
        "ep": exact_polar(M, target_norm=T)["Y"],
        "head": head_spec_muon(M, lam=0.25, target_norm=T)["Y"],
        "bulk": bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"],
        "mp": muon_power(M, p=0.5, target_norm=T)["Y"],
    }
    d2 = {n: torch.diagonal(U.T @ Y.double() @ V) ** 2 for n, Y in arms.items()}
    ratio = {n: float(d2[n][:k10].mean() / d2[n][k10:].mean()) for n in arms}
    share10 = {n: float(d2[n][:k10].sum() / d2[n].sum()) for n in arms}
    row.update(
        k10=k10,
        hr_ratio_polar=ratio["ep"], hr_ratio_head=ratio["head"],
        hr_ratio_bulk=ratio["bulk"], hr_ratio_muonp=ratio["mp"],
        share10_polar=share10["ep"], share10_head=share10["head"],
        share10_bulk=share10["bulk"], share10_muonp=share10["mp"],
    )
    if full:
        Ycv, info = converged_polar_reference(M)
        Ype = polar_express_single_pass(M)
        row.update(
            conv_iters=info["iterations"], conv_defect=info["defect"],
            conv_converged=int(info["converged"]),
            gap_Y5_conv=normalized_gap(Y5, Ycv),
            gap_conv_polar=normalized_gap(Ycv, P),
            conv_oracle_valid=int(normalized_gap(Ycv, P) <= 1e-3),
            gap_Y5_pe5=normalized_gap(Y5, Ype),
            gap_pe5_polar=normalized_gap(Ype, P),
            pe5_defect=orthogonality_defect(Ype),
        )
    row["seconds"] = time.perf_counter() - t0
    return row, x.tolist(), s5.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    torch.set_num_threads(cfg["threads"])

    t_start = time.perf_counter()
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # provenance: installed wheel must match the frozen sha; vendored must be bit-exact
    import torch.optim._muon as _tm

    wheel_sha = file_sha256(Path(_tm.__file__))
    probe = torch.randn(64, 48, generator=torch.Generator().manual_seed(0))
    vendored_eq = bool(
        torch.equal(
            vendored_zeropower_via_newtonschulz(probe),
            native_ns5_installed(probe),
        )
    )
    provenance = {
        "torch_version": torch.__version__,
        "muon_wheel_sha256": wheel_sha,
        "muon_wheel_sha_matches_frozen": wheel_sha == MUON_WHEEL_SHA256,
        "vendored_bitexact_with_installed": vendored_eq,
    }

    runs = cfg["runs"]
    per_matrix_rows = []
    gprofile_points = []
    for run in runs:
        full = run == cfg["canonical_run"]
        loaded = load_m1_muon_buffers(M1_DIR / run)
        for name in loaded["names"]:
            M = loaded["buffers"][name]
            row, x_list, s5_list = analyze_matrix(M, full=full)
            row = {"run": run, "name": name, "module": module_class(name),
                   "layer": layer_idx(name), **row}
            per_matrix_rows.append(row)
            if full:
                for xi, si in zip(x_list, s5_list):
                    gprofile_points.append(
                        {"module": row["module"], "layer": row["layer"], "x": xi, "g": si}
                    )
        print(f"[p0a] {run}: analyzed {len(loaded['names'])} matrices "
              f"(sha_ok={loaded['payload_sha256_ok']})", flush=True)

    fields = sorted({k for r in per_matrix_rows for k in r})
    with open(out_dir / "p0a_per_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_matrix_rows)

    with open(out_dir / "p0a_gprofile_points.json", "w") as f:
        json.dump(gprofile_points, f)

    # aggregate by run x module
    agg = {}
    for r in per_matrix_rows:
        key = (r["run"], r["module"])
        agg.setdefault(key, []).append(r)
    agg_rows = []
    for (run, mod), rows in sorted(agg.items()):
        def m(k):
            return sum(rr[k] for rr in rows) / len(rows)
        agg_rows.append({
            "run": run, "module": mod, "n": len(rows),
            "in_E1_mean": m("in_E1"), "in_E8_mean": m("in_E8"),
            "in_PR_mean": m("in_PR"),
            "out_E1_mean": m("out_E1"), "out_E8_mean": m("out_E8"),
            "out_PR_mean": m("out_PR"),
            "defect_Y5_mean": m("defect_Y5"),
            "gap_Y5_polar_mean": m("gap_Y5_polar"),
            "gap_Y5_conv_mean": m("gap_Y5_conv") if "gap_Y5_conv" in rows[0] else "",
            "gap_Y5_pe5_mean": m("gap_Y5_pe5") if "gap_Y5_pe5" in rows[0] else "",
            "polar_top8_align_mean": m("polar_top8_align"),
        })
    with open(out_dir / "p0a_by_module.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)

    report = {
        "config": cfg,
        "provenance": provenance,
        "n_matrices": len(per_matrix_rows),
        "wall_seconds": time.perf_counter() - t_start,
    }
    with open(out_dir / "p0a_run_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(provenance, indent=2))
    print(f"[p0a] done in {report['wall_seconds']:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
