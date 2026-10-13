# P1 DECISION (deliverable 5)

Frozen screening (H5 / SPECMUON_FROZEN_PILOT.md, thresholds fixed before any
development result): a candidate must beat native NS5 by >2e-3 H256 nats at BOTH
restore points, beat the whole equal-budget LR grid at both points, and exceed the
ExactPolar control by >1e-3 at both points.

## Verdict (computed by `src/specmuon/analyze_p1.py` from the frozen tree)

| candidate | s512 d(native) | s1536 d(native) | beats LR grid | beyond polar | verdict |
|---|---|---|---|---|---|
| Head-SpecMuon λ=0.25 | −0.005096 | −0.000999 | no (−0.0791 / −0.0089) | no | **NO_EFFECT** |
| Bulk-SpecMuon λ=0.25 | +0.002796 | +0.000708 | no (−0.0712 / −0.0072) | +0.0036 / +0.0008 | **SINGLE_POINT → no confirmed effect** |

- Bulk's single-point +0.0028 exceeds the 2e-3 screen only at s512 and decays to
  +0.0007 (< screen) at s1536, and never approaches the LR-grid control: per the
  frozen tree this is SINGLE_POINT = insufficient, and additionally LR_DOMINATED.
- ExactPolar ≈ native (≤1e-3, native better): no POLAR_ONLY signal either.
- p-power 0.5 (non-authoritative variant): clearly harmful. No claims are made
  against published Muon^p/MuPS, SAMuon, or AMO (SOURCE_BLOCKED,
  `BASELINE_SOURCE_AUDIT.md`).

## Overall answer: is the spectral prior worth continuing?

**No.** At this scale and setting (30M MiniMind, 256-step continuations from a
verified parent, norm-matched updates, oracle fp64 SVD):

1. The fixed spectral prior (either direction) does not beat tuned native Muon;
   the gap to the LR-grid control is 1-2 orders of magnitude larger than any
   spectral effect.
2. The operator-level NS5-vs-polar difference (P0-A: 13.3%) is training-irrelevant
   (≤1e-3 nats, wrong sign for a "closer-to-polar-is-better" story).
3. The cost of the exact-SVD oracle (5-6x optimizer wall) is not defensible for
   effects of this size.

## Consequences (frozen non-goals enforced)

- **No P2**: no dynamic time-spectral controller, no alpha schedule, no
  time-PSD selector training (P0-C data gap also stands).
- **No AdamSpec**: the Muon-side prerequisite (fixed spectral prior beating strong
  controls) failed; designing an AdamW transfer is not justified by this evidence.
- **No large-scale rerun, no new spectral parameter search** (single frozen λ/p
  values were the protocol; widening the grid now would be post-hoc).
- The spectral_prior_muon_pilot question is **closed negative** at pilot scale.
  If the operator nonetheless wants one follow-up (not recommended, not
  pre-registered): the only hint with stable sign is the mild bulk-tilt; any
  future study must first show it beating the LR grid at both points.

## Status of deliverables

- `results/p1/P1_RESULTS.csv`, `results/p1/P1_RUNTIME_COST.csv` (generated),
  `results/p1/decision.json` (frozen decision computation + checks),
  `results/p1/{replay_gate.json, bench_gpu.json, init_eval_s*.json,
  branches/*/train_metrics.jsonl, branches/*/result.json, branches/*/branch_meta.json}`.
- Evidence bundle: `P1_EVIDENCE_BUNDLE.zip` (tables, configs, code + hashes,
  recomputation script; no checkpoints).
