"""Directed GPU replay for the re-audit (task sec 6). First check batch per state only,
pre-registered candidates only; includes K5 constructor no-op, A->B->A order test, and
one independent native repeat. Budget: <=15 GPU-minutes total."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.p1_engine import evaluate_batch
from src.state_restore import build_q_sets, access_guard

OUT = ROOT / "outputs" / "p0_p1_v1_1"
AUD = ROOT / "audit_recheck"
guard = access_guard({"P1_check"})  # replay uses check batches -> P1_check target; lock already historical


def main():
    t0 = time.time()
    q = build_q_sets()
    lock = json.loads((OUT / "selection_lock.json").read_text())
    rows = []
    plans = {
        512: [
            {"kind": "native"},                                   # A (also repeat base)
            {"kind": "group", "group": "q", "K": 3, "probe": "raw"},   # B
            {"kind": "native"},                                   # A again (A->B->A)
            {"kind": "group", "group": "o", "K": 3, "probe": "gain_only_live"},
            {"kind": "group", "group": "q", "K": 7, "probe": "raw"},
            {"kind": "group", "group": "q", "K": 3, "probe": "gain_only_live"},
            {"kind": "group", "group": "q", "K": 3, "probe": "direction_only"},
            {"kind": "global_amp", "scale": 0.5},
        ],
        1536: [
            {"kind": "native"},
            {"kind": "group", "group": "q", "K": 3, "probe": "raw"},
            {"kind": "native"},
            {"kind": "group", "group": "q", "K": 7, "probe": "raw"},
            {"kind": "group", "group": "q", "K": 3, "probe": "gain_only_live"},
            {"kind": "group", "group": "q", "K": 3, "probe": "direction_only"},
            {"kind": "global_amp", "scale": 0.5},
        ],
    }
    # add locked joint schemes (1536 target+gamma; both states geometry)
    for s in (512, 1536):
        st = lock["states"][str(s)]
        gm, gp = st["geometry_primary"]
        plans[s].append({"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "raw",
                         "label": f"geometry|{gm}3|{gp}7"})
        if not st["target_primary"]["no_op"]:
            tm, tp = st["target_primary"]["pair"]
            plans[s].append({"kind": "joint", "K_vector": {tm: 3, tp: 7}, "probe": "raw",
                             "label": "target_primary"})
            plans[s].append({"kind": "joint", "K_vector": {tm: 3, tp: 7}, "probe": "gamma_const",
                             "gamma_const": {tm: st["gamma_const_all"][f"{tm}|3"],
                                             tp: st["gamma_const_all"][f"{tp}|7"]},
                             "label": "target_gamma_const"})

    for s, schemes in plans.items():
        rr, mm = evaluate_batch(OUT / "parent" / f"parent_state_step{s}.pt", s,
                                (s + 4, s + 5), q, "P1_check", schemes)
        guard("P1_check")
        for r in rr:
            r.update({"state": s, "batch": 5})
        rows.extend(rr)
        print(f"state {s}: {len(rr)} points in {time.time()-t0:.0f}s")
    all_fields = []
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(AUD / "replay" / "replay_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # compare with historical values (same state, first check batch = batch 5)
    chk = list(csv.DictReader(open(OUT / "check_rows.csv")))
    comp = []
    for r in rows:
        hist = [x for x in chk if x["state"] == str(r["state"]) and x["batch"] == "5"
                and x["scheme"] == r["scheme"]]
        if hist:
            h = float(hist[0]["V"])
            comp.append({"state": r["state"], "scheme": r["scheme"],
                         "replay_V": r["V"], "historical_V": h,
                         "abs_diff": abs(r["V"] - h)})
        else:
            comp.append({"state": r["state"], "scheme": r["scheme"],
                         "replay_V": r["V"], "historical_V": None, "abs_diff": None})
    # A->B->A: two native rows in the same call must be identical
    for s in (512, 1536):
        nats = [r["V"] for r in rows if r["state"] == s and r["scheme"] == "native"]
        aba = {"state": s, "scheme": "A_B_A_native_identity",
               "replay_V": nats[0], "historical_V": nats[1], "abs_diff": abs(nats[0] - nats[1])}
        comp.append(aba)
    with open(AUD / "replay_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)
    mx = max(c["abs_diff"] for c in comp if c["abs_diff"] is not None)
    print(f"replay done in {time.time()-t0:.0f}s; max |replay-historical| = {mx:.3e}; "
          f"A->B->A diffs 0 for both states: "
          f"{all(abs([r['V'] for r in rows if r['state']==s and r['scheme']=='native'][0] - [r['V'] for r in rows if r['state']==s and r['scheme']=='native'][1]) == 0 for s in (512,1536))}")


if __name__ == "__main__":
    main()
