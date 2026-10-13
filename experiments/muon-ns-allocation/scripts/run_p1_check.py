"""P1 check: locked schemes + full KJ probe display on the 4 check batches per state,
evaluated on Q_P1_check ONLY (after selection_lock.json exists). No new candidates."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.p1_engine import evaluate_batch, kj_schemes
from src.state_restore import build_q_sets, access_guard
from src.recipe_registry import P1_STATES

OUT = ROOT / "outputs" / "p0_p1_v1_1"
guard = access_guard({"P1_check"})


def main():
    t0 = time.time()
    lock = json.loads((OUT / "selection_lock.json").read_text())
    final = json.loads((OUT / "final_schemes.json").read_text())
    q = build_q_sets()
    rows = []
    for s in P1_STATES:
        sp = OUT / "parent" / f"parent_state_step{s}.pt"
        locked = json.loads(json.dumps(final[str(s)]))  # deep copy of scheme dicts
        schemes = kj_schemes() + [x for x in locked if x.get("kind") != "native" or x.get("label")]
        # ensure a native row always present exactly once
        schemes = [{"kind": "native"}] + [x for x in schemes if not (x.get("kind") == "native" and not x.get("label"))]
        for bi in range(4, 8):
            rr, mm = evaluate_batch(sp, s, (s + bi, s + bi + 1), q, "P1_check", schemes)
            guard("P1_check")
            for r in rr:
                r.update({"state": s, "batch": bi + 1})
            rows.extend(rr)
            nat = [r["V"] for r in rr if r["scheme"] == "native"][0]
            print(f"[{time.time()-t0:.0f}s] state {s} check batch {bi+1}: native V={nat:.6f} "
                  f"({len(rr)} pts) indep_ok={mm['independent_recollection_ok']}", flush=True)
    all_fields = []
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(OUT / "check_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"check stage done in {time.time()-t0:.0f}s; {len(rows)} rows")


if __name__ == "__main__":
    main()
