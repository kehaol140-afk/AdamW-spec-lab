#!/usr/bin/env python3
"""AdamW-spec-lab independent recompute suite (CPU-only).

Recomputes published headline numbers directly from raw per-doc eval JSONL / CSV
in this staging tree. Audit reference checkpoints are comparison targets only;
mismatches are reported, never forced.

Usage:  python3 recompute/run_recompute.py          (from repo root of staging)
Output: recompute/outputs/<ck>_recompute.json + recompute/outputs/SUMMARY.md
"""
import json, os, sys, glob, csv, math, random, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "outputs"); os.makedirs(OUT, exist_ok=True)
E = os.path.join(ROOT, "experiments")

def agg_jsonl(path):
    """token-weighted NLL over per-doc rows"""
    ls = ts = 0.0; n = 0
    for line in open(path):
        r = json.loads(line); ls += r["loss_sum"]; ts += r["token_count"]; n += 1
    return ls / ts, n

def per_doc(path):
    """doc_id -> token-weighted NLL (paired bootstrap material)"""
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["doc_id"]] = (r["loss_sum"], r["token_count"])
    return d

def paired_bootstrap(docsA, docsB, reps=1000, seed=9071):
    """bootstrap CI for pooled token-weighted NLL(B) - NLL(A), doc-level pairing.
    Point estimate = difference of the published token-weighted NLL aggregates."""
    keys = sorted(set(docsA) & set(docsB))
    a = [(docsA[k][0], docsA[k][1]) for k in keys]
    b = [(docsB[k][0], docsB[k][1]) for k in keys]
    def pooled(xs):
        return sum(x[0] for x in xs) / sum(x[1] for x in xs)
    mean = pooled(b) - pooled(a)
    rng = random.Random(seed); n = len(keys); boots = []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(pooled([b[i] for i in idx]) - pooled([a[i] for i in idx]))
    boots.sort()
    lo, hi = boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]
    return mean, lo, hi, n

CHECKS = {}

def ck(name):
    def deco(fn): CHECKS[name] = fn; return fn
    return deco

# ---------------- CK1: V16-C 3-seed final_holdout ----------------
@ck("V16C_3seed")
def ck1():
    base = os.path.join(E, "adamw-spectral-v13-v16/campaign_v16/raw/eval")
    out = {}
    for s in (3408, 3409, 3410):
        S, _ = agg_jsonl(f"{base}/c_b_S1_plain_lr0.001_s{s}__H4096__final_holdout.jsonl")
        F, _ = agg_jsonl(f"{base}/c_b_F1_a8_lr0.001_s{s}__H4096__final_holdout.jsonl")
        out[f"s{s}"] = {"S1": S, "F1": F, "S_minus_F": S - F}
    out["mean_S_minus_F"] = sum(out[f"s{s}"]["S_minus_F"] for s in (3408, 3409, 3410)) / 3
    out["reference"] = {"s3408": 0.036276, "s3409": 0.094018, "s3410": 0.151494, "mean": 0.093929}
    return out

# ---------------- CK2: DV Layer-1 ----------------
@ck("DV_LAYER1")
def ck2():
    base = os.path.join(E, "adamw-direction-validation/raw/eval")
    arms = ["dv_S1", "dv_S4", "dv_S8", "dv_F8", "dv_P8"]
    vals = {a: agg_jsonl(f"{base}/{a}__H2048__development.jsonl")[0] for a in arms}
    docs = {a: per_doc(f"{base}/{a}__H2048__development.jsonl") for a in arms}
    out = {"dev_H2048": vals}
    for label, a, b in [("F8-S8", "dv_S8", "dv_F8"), ("P8-S8", "dv_S8", "dv_P8"), ("S8-S1", "dv_S1", "dv_S8"), ("F8-P8", "dv_P8", "dv_F8")]:
        mean, lo, hi, n = paired_bootstrap(docs[a], docs[b])
        out[label] = {"mean(b-a)": mean, "ci95": [lo, hi], "n_docs": n}
    out["reference"] = {"F8-S8": -0.204090, "P8-S8": -0.235050, "S8-S1": 0.082396,
                        "ci": {"F8-S8": [-0.214854, -0.192917], "P8-S8": [-0.246108, -0.223252]}}
    return out

# ---------------- CK3: 30M T1 ----------------
@ck("S30M_T1")
def ck3():
    base = os.path.join(E, "adamw-direction-30m/raw/eval")
    arms = ["s30_S1_lr0.001_L1", "s30_S2_lr0.001_L1", "s30_F2_lr0.001_L1",
            "s30_P2_lr0.001_L1", "s30_P2_lr0.001_L2", "s30_F2_P2"]
    vals = {}
    for a in set(glob.glob(f"{base}/*__H2048__development.jsonl")):
        vals[os.path.basename(a).split("__")[0]] = agg_jsonl(a)[0]
    docs = {a: per_doc(f"{base}/{a}__H2048__development.jsonl") for a in vals}
    muon_best = min(v for k, v in vals.items() if k.startswith("s30_M_"))
    out = {"dev_H2048": vals}
    for label, a, b in [("F2-S2@L1", "s30_S2_lr0.001_L1", "s30_F2_lr0.001_L1"),
                        ("P2-S2@L1", "s30_S2_lr0.001_L1", "s30_P2_lr0.001_L1"),
                        ("P2-S1", "s30_S1_lr0.001_L1", "s30_P2_lr0.001_L1"),
                        ("P2@L2-S2@L1", "s30_S2_lr0.001_L1", "s30_P2_lr0.001_L2")]:
        if a in docs and b in docs:
            mean, lo, hi, n = paired_bootstrap(docs[a], docs[b])
            out[label] = {"mean(b-a)": mean, "ci95": [lo, hi], "n_docs": n}
        else:
            out[label] = {"diff_of_means": vals.get(b) - vals.get(a)}
    out["Muon_best_minus_P2"] = muon_best - vals["s30_P2_lr0.001_L1"]
    out["reference"] = {"F2-S2@L1": -0.037930, "P2-S2@L1": -0.045737, "P2-S1": -0.013957, "Muon_best_minus_P2": -0.272500}
    return out

# ---------------- CK4: MSA P0/P1 screen + R2F attribution ----------------
@ck("MSA_attribution")
def ck4():
    out = {}
    rows = list(csv.DictReader(open(os.path.join(E, "muon-ns-allocation/outputs/p0_p1_v1_1/check_rows.csv"))))
    reallocation = [r for r in rows if r["scheme"] not in ("native",) and r.get("group")]
    # 28 single-group check rows: group-level candidates at states 512/1536
    checks = [r for r in rows if r.get("group") and r["state"] in ("512", "1536")]
    benefit = []
    for r in checks:
        try: benefit.append(float(r["V"]) - float(rows[0]["V"]) if False else float(r["nll_B"]) - float(r["nll_A"]))
        except Exception: pass
    out["p0p1_check_rows"] = len(checks)
    at = list(csv.DictReader(open(os.path.join(E, "muon-ns-allocation/outputs/r2f_lrwd_retention_v1/lrwd_attribution.csv"))))
    row512 = at[0]
    V_N, V_AH, V_L = float(row512["V_N"]), float(row512["V_AH"]), float(row512["V_L"])
    B_L = V_N - V_L
    out["R2F_512"] = {"V_N": V_N, "V_AH": V_AH, "V_L": V_L, "B_L": B_L,
                      "B_L_over_AH_gain_pct": 100.0 * B_L / (V_N - V_AH)}
    pa = list(csv.DictReader(open(os.path.join(E, "muon-ns-allocation/outputs/p0_p1_v1_1/amplitude_controls.csv"))))
    out["amplitude_controls"] = [{k: r[k] for k in list(r)[:5]} for r in pa[:4]]
    out["reference"] = {"B_L_over_AH_gain_pct": 101.7, "global_x05_512": 3.4235e-4}
    return out

# ---------------- CK5: SpecMuon P1 14 branches ----------------
@ck("SPECMUON_P1")
def ck5():
    rows = list(csv.DictReader(open(os.path.join(E, "specmuon-pilot/results/p1/P1_RESULTS.csv"))))
    out = {"n_rows": len(rows), "rows": rows}
    return out

# ---------------- CK6: 100M 15 arms x 3 horizons ----------------
@ck("CAM_100M")
def ck6():
    pil = os.path.join(E, "context-aware-muon-100m")
    table = {}
    csvrows = list(csv.DictReader(open(os.path.join(pil, "PILOT_100M_RESULTS.csv"))))
    for r in csvrows:
        table[r["arm"]] = {"H1024": float(r["dev_nll_H1024"]), "H2048": float(r["dev_nll_H2048"]),
                           "H4096": float(r["dev_nll_H4096"])}
    # recompute H4096 from raw per-doc files
    recomputed = {}
    for arm in table:
        p = os.path.join(pil, f"results/pilot/{arm}/eval_docs_H4096.jsonl")
        if os.path.exists(p):
            v, n = agg_jsonl(p); recomputed[arm] = {"H4096_jsonl": v, "n_docs": n,
                                                    "delta_vs_csv": v - table[arm]["H4096"]}
    best_anchor = min(table[a]["H4096"] for a in table if a.startswith("B_muon"))  # verdict anchor = best Muon-family baseline
    out = {"table": table, "H4096_recomputed": recomputed,
           "best_B_anchor_H4096": best_anchor,
           "C_A_lam0p85_minus_best_anchor": table["C_A_lam0p85"]["H4096"] - best_anchor,
           "C_B_minus_C_A": table["C_B"]["H4096"] - table["C_A_lam0p85"]["H4096"],
           "strict_monotone_rise": {a: table[a]["H4096"] > table[a]["H2048"] > table[a]["H1024"] for a in table},
           "rise_from_H1024": {a: table[a]["H4096"] > table[a]["H1024"] for a in table}}
    out["reference"] = {"C_A_minus_best_anchor": 0.1367, "C_B_minus_C_A": 0.0997}
    return out

# ---------------- CK7: Hyperball A1 ----------------
@ck("HYPERBALL_A1")
def ck7():
    hb = os.path.join(ROOT, "baselines/hyperball-a1-evidence")
    def last_val(p):
        for line in open(p):
            if "val_loss" in line: last = line.strip()
        return float(last.split("val_loss:")[1].split()[0])
    M = last_val(os.path.join(hb, "run_M_stdout.log")); H = last_val(os.path.join(hb, "run_H_stdout.log"))
    def norms(p):
        first = last = None
        for line in open(p):
            r = json.loads(line)
            if "total_muon_norm" in r:
                if first is None: first = r["total_muon_norm"]
                last = r["total_muon_norm"]
        return first, last
    hN = norms(os.path.join(hb, "a1_H_diagnostics.jsonl")); mN = norms(os.path.join(hb, "a1_M_diagnostics.jsonl"))
    return {"Muon_M_val_loss": M, "MuonH_H_val_loss": H, "H_minus_M": H - M,
            "H_norm_first_last": hN, "M_norm_first_last": mN,
            "reference": {"M": 3.27906, "H": 3.27921, "H_norm_const": 261.928}}

# ---------------- CK8: V16 A2 four branches ----------------
@ck("V16_A2")
def ck8():
    base = os.path.join(E, "adamw-spectral-v13-v16/campaign_v16/raw/eval")
    out = {}
    native = {"S": "a1_S_a8_b2", "F": "a1_F_a8_b2", "P": "a1_P_a8_b2"}
    for k, cid in native.items():
        out[f"native_{k}_H256"] = agg_jsonl(f"{base}/{cid}__H256__development.jsonl")[0]
    for cid in ["a2_Fdir_qS", "a2_Pdir_qS", "a2_Sdir_qF", "a2_Sdir_qP"]:
        out[f"{cid}_H256"] = agg_jsonl(f"{base}/{cid}__H256__development.jsonl")[0]
    out["reference"] = {"Fdir_qS_minus_nativeS": -0.00266, "Pdir_qS_minus_nativeS": -0.00257,
                        "Sdir_qF_minus_nativeF": 0.00294, "Sdir_qP_minus_nativeP": 0.00298}
    return out

def main():
    summary = {}
    for name, fn in CHECKS.items():
        try:
            res = fn()
            with open(os.path.join(OUT, f"{name}_recompute.json"), "w") as f:
                json.dump(res, f, indent=1, default=str)
            summary[name] = "OK"
        except Exception as e:
            summary[name] = f"ERROR: {type(e).__name__}: {e}"
    with open(os.path.join(OUT, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    main()
