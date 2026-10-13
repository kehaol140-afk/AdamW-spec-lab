"""Independent analysis for R0/R1: recompute V from sums, paired effects for the six
predefined comparisons, per-domain results, boundary status, paired bootstrap."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r0_r1_v1_1"
BOOT_N, BOOT_SEED = 10000, 31010408
DELTA = 1e-5


def main():
    chk = list(csv.DictReader(open(OUT / "check_values.csv")))
    cal = list(csv.DictReader(open(OUT / "calibration.csv")))
    lock = json.loads((OUT / "selection_lock.json").read_text())
    ext = json.loads((OUT / "extension_lock.json").read_text())

    def vmap(state, pid_str, batches=(1, 2, 3, 4)):
        rows = [r for r in chk if str(r["state"]) == str(state)
                and r["point_id"] == pid_str and int(r["batch"]) in batches]
        assert len(rows) == 4, (state, pid_str, len(rows))
        return (np.array([float(r["V"]) for r in rows]),
                np.array([float(r["nll_A"]) for r in rows]),
                np.array([float(r["nll_B"]) for r in rows]))

    def paired(state, ref_id, cand_id, label):
        vr, ar, br = vmap(state, ref_id)
        vc, ac, bc = vmap(state, cand_id)
        d = vr - vc
        rng = np.random.default_rng(BOOT_SEED + int(state))
        boots = [float(np.mean(d[rng.integers(0, 4, 4)])) for _ in range(BOOT_N)]
        return {"state": state, "comparison": label, "ref": ref_id, "cand": cand_id,
                "diffs": d.tolist(), "mean": float(d.mean()), "median": float(np.median(d)),
                "sd": float(d.std(ddof=1)), "n_pos": int((d > 0).sum()),
                "min": float(d.min()), "max": float(d.max()),
                "domain_A_mean": float((ar - ac).mean()), "domain_B_mean": float((br - bc).mean()),
                "boot_lo": float(np.percentile(boots, 2.5)),
                "boot_hi": float(np.percentile(boots, 97.5)),
                "local_signal_rule": bool(d.mean() >= DELTA and (d > 0).sum() >= 3)}

    rows = []
    for state in ("512", "1536"):
        R = lock["roles"][state]
        pairs = [
            ("N", "G", "1_global_gain"),
            ("G", "A", "2_group_vs_global_fair"),
            ("G_at_A", "A", "3_group_pure_paired"),
            ("A", "S_at_A", "4_matched_structure"),
            ("A", "S", "5_equal_search_structure"),
            ("A_at_S", "S", "6_reverse_matched_structure"),
        ]
        for ref_role, cand_role, label in pairs:
            r, c = R.get(ref_role), R.get(cand_role)
            if r is None or c is None or r["id"] == c["id"]:
                rows.append({"state": state, "comparison": label, "skipped": "role missing or identical"})
                continue
            rows.append(paired(state, r["id"], c["id"], label))
        # references vs native (diagnostic)
        rows.append(paired(state, R["N"]["id"], R["Z"]["id"], "Z_no_data_diag"))
        rows.append(paired(state, R["N"]["id"], R["P"]["id"], "P_pre_state_diag"))

    # boundary status per family
    A_fin, B_fin = ext["A_final"], ext["B_final"]
    boundary = {}
    for state in ("512", "1536"):
        R = lock["roles"][state]
        boundary[state] = {
            "G_touches_a": R["G"]["a"] == min(A_fin),
            "A_touches_a": R["A"]["a"] == min(A_fin),
            "A_touches_b": R["A"]["b"] in (min(B_fin), max(B_fin)),
            "S_touches_a": R["S"]["a"] == min(A_fin),
            "S_touches_b": R["S"]["b"] in (min(B_fin), max(B_fin)),
        }
    any_touch = any(any(d.values()) for d in boundary.values())
    boundary["AMPLITUDE_RANGE_UNRESOLVED"] = any_touch
    json.dump(boundary, open(OUT / "boundary_status.json", "w"), indent=1)

    with open(OUT / "paired_effects.csv", "w", newline="") as f:
        keys = ["state", "comparison", "ref", "cand", "mean", "median", "sd", "n_pos",
                "min", "max", "domain_A_mean", "domain_B_mean", "boot_lo", "boot_hi",
                "local_signal_rule", "diffs", "skipped"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # verdict assembly
    def sig(state, label):
        r = [x for x in rows if x.get("state") == state and x.get("comparison") == label]
        return r[0] if r else None
    global_ok = all(sig(s, "1_global_gain") and sig(s, "1_global_gain")["local_signal_rule"]
                    for s in ("512", "1536"))
    group_ok = all(sig(s, "2_group_vs_global_fair") and sig(s, "2_group_vs_global_fair")["local_signal_rule"]
                   for s in ("512", "1536"))
    struct_ok = all(sig(s, "4_matched_structure") and sig(s, "4_matched_structure")["local_signal_rule"]
                    for s in ("512", "1536"))
    struct_search_ok = all(sig(s, "5_equal_search_structure") and sig(s, "5_equal_search_structure")["local_signal_rule"]
                           for s in ("512", "1536"))
    verdict = {
        "AMPLITUDE_RANGE_UNRESOLVED": any_touch,
        "GLOBAL_ONLY_LOCAL_SIGNAL": global_ok and not group_ok and not struct_ok,
        "GROUP_INCREMENT_LOCAL_SIGNAL": group_ok and not struct_ok,
        "MATCHED_STRUCTURE_LOCAL_SIGNAL": struct_ok,
        "structure_equal_search_signal": struct_search_ok,
    }
    json.dump(verdict, open(OUT / "verdict.json", "w"), indent=1)
    print(json.dumps(boundary, indent=1))
    for r in rows:
        if "skipped" in r:
            print(f"[{r['state']} {r['comparison']}] SKIPPED")
        else:
            print(f"[{r['state']} {r['comparison']}] mean={r['mean']:+.4e} "
                  f"pos={r['n_pos']}/4 A={r['domain_A_mean']:+.3e} B={r['domain_B_mean']:+.3e} "
                  f"CI=[{r['boot_lo']:+.2e},{r['boot_hi']:+.2e}] signal={r['local_signal_rule']}")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
