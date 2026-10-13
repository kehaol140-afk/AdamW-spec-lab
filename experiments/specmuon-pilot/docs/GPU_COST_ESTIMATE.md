# GPU cost estimate (deliverable 5)

Date: 2026-10-10. Purpose: pre-compute the P1 pilot cost so the <= 1 GPU-hour gate can
be evaluated before launch. All inputs are measured facts, not guesses, except where
marked [est].

## Measured anchors

- m1 Muon run (identical model/data/optimizer path): **2048 steps in 778 s wall** on
  one RTX 5090 (`m1_muon_lr1.0_wd0.01_p0.5_s2030/result.json: wall_seconds`) =>
  **0.38 s/step** including periodic hooks; final H256 dev eval included.
- Pilot branch length: 256 steps => **~98 s compute** per branch + final eval ~20-30 s
  [est] => **~2.1 min per native-family branch**.
- SVD-oracle arms add, per step, 56 SVDs (largest 1664x512 fp32) + 56 NS5 calls
  (already counted in native path). RTX 5090 fp32 SVD of 1664x512: **3-8 ms [est]**;
  total +0.17-0.45 s/step => **+0.7-1.9 min per branch**.

## Budget table (14 branches + smoke + replay)

| item | count | unit cost | total |
|---|---|---|---|
| runner smoke (2 steps + eval) | 1 | ~1 min [est] | 1 min |
| native replay gate (1 step + compare) | 1 | ~1 min | 1 min |
| native-family branches (native x3 LR) | 6 | ~2.1 min | ~13 min |
| oracle arms (exact_polar, head, bulk, muon_p) | 8 | ~2.8-4.0 min | ~22-32 min |
| **total** | | | **~37-47 min** |

Headroom vs the 1 h gate: ~13-23 min (23-38%). The dominant uncertainty is GPU SVD
throughput; it is bounded by the smoke test before any branch starts.

## Overhead accounting to be reported (per branch, frozen)

- `optimizer_extra_seconds`: wall time inside SpecMuon beyond the native NS5 path
  (SVD + reweight + norm matching), logged every step.
- `wall_seconds` per branch; quality always reported BOTH same-token-budget and
  same-wall-clock (per-step cost ratio = branch wall ratio at equal 256 steps).

## Shrink rule (frozen)

If the smoke test or the first oracle branch measures > 20% above this estimate:
run only the 512-restore-point batch (7 branches, ~20-24 min), report, and re-estimate
before requesting gate re-approval for the 1536-point batch. Never shorten branches,
drop arms, or subsample evals to fit budget.

## GPU gate checklist (all required before ANY GPU run)

1. V16 GPU tasks terminated: verify no `adamw_spectral_pilot/campaign_v16/run.lock`
   holder is alive (`nvidia-smi`, process list, V16 STATUS.md) — currently **FAILED:
   V16 stage A0 alive on GPU0 (verified 2026-10-10)**.
2. Shared GPU0 explicitly approved by the operator coordinating V16.
3. Native replay gate passed (SPECMUON_FROZEN_PILOT.md gate 2).
4. This estimate stands (re-check anchors if > 7 days old).
5. GPU1 never used (another-user vLLM, 27.8 GB resident).
6. CPU side stays capped (8 threads, `nice`) even during GPU runs, to protect V16's
   CPU stages and the vLLM service.

**Current status: STOPPED AT GPU GATE, condition 1 not met. No GPU work started.**
Prepared launch command sequence (for when the gate opens), dry-run only:

```
cd /home/b205/projects/training_dynamics_lab/spectral_prior_muon_pilot
PYTHONPATH=src python -m specmuon.run_pilot --config configs/pilot_frozen_v1.json \
    --stage smoke        # gate 1: runner smoke (may run CPU-only)
PYTHONPATH=src python -m specmuon.run_pilot --config configs/pilot_frozen_v1.json \
    --stage replay       # gate 2: native step-513 replay (GPU, minutes)
PYTHONPATH=src python -m specmuon.run_pilot --config configs/pilot_frozen_v1.json \
    --stage pilot --restore-point 512   # batch 1 (~22 min)
PYTHONPATH=src python -m specmuon.run_pilot --config configs/pilot_frozen_v1.json \
    --stage pilot --restore-point 1536  # batch 2 (~22 min)
```

(`src/specmuon/run_pilot.py` is intentionally NOT written yet: the runner must be
implemented and reviewed at pilot time against this frozen doc; writing it earlier
would invite drift while the gate is closed.)
