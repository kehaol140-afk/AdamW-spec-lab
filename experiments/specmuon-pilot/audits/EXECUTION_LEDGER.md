# Execution ledger — spectral_prior_muon_pilot

All runs CPU-only, `nice -n 10`, `OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`.
Environment: `/home/b205/miniforge3/envs/fr_minimind/bin/python` (python 3.11.16,
torch 2.11.0+cu130, wheel `_muon.py` sha256 `38e95918...` = frozen pin, verified at
runtime by `run_p0a` provenance block).

| # | date (UTC) | action | inputs (read-only) | outputs | wall |
|---|---|---|---|---|---|
| 1 | 2026-10-10 | Project skeleton + frozen protocol/hypotheses docs | — | docs/, src/, tests/, configs/ | — |
| 2 | 2026-10-10 | Operator + semantics tests (3 iterations until green) | — | 12/12 + 7/7 PASS | ~6 min |
| 3 | 2026-10-10 | P0-A audit, 6 m1 Muon runs x 56 matrices, canonical run full metrics incl. converged polar reference + PE5 | `muon_transition_experiment/outputs/m1/*` (payload sha verified per run) | `results/p0a/{p0a_per_matrix.csv, p0a_by_module.csv, p0a_gprofile_points.json, p0a_run_report.json}` | 262 s |
| 4 | 2026-10-10 | P0-B phase 1 (4 states resource check) | v07 bank seed {2030,2033} step {46,64} | `results/p0b/p0b_phase1_*` | 1.9 s |
| 5 | 2026-10-10 | P0-B phase 2 (all 56 states) | v07 bank 8 seeds x 7 steps | `results/p0b/p0b_phase2_*` | 52 s |
| 6 | 2026-10-10 | P0-C file-level audit (no compute) | T2a outputs, m1 obs, V16 logs | section in `docs/P0_SPECTRAL_AUDIT.md` | — |

## External-asset integrity

- m1 checkpoints: `payload_sha256` recomputed and matched on all 6 loaded runs
  (`payload_sha256_ok=True` in run logs).
- v07 bank: loaded read-only via `torch.load(map_location="cpu", weights_only=False)`;
  original files never opened for write; `PREREG_LOCK.json` untouched.
- No V16 asset touched. V16 campaign observed alive on GPU0 throughout
  (`adamw_spectral_pilot/campaign_v16/run.lock`, stage A0) -> GPU gate remains CLOSED.

## Gate status at end of P0

GPU pilot NOT started (V16 GPU tasks alive). Waiting conditions and full run
instructions: `docs/SPECMUON_FROZEN_PILOT.md`, `docs/GPU_COST_ESTIMATE.md`.

---

# P1 ledger (2026-10-11)

| # | stage | action | outcome | wall |
|---|---|---|---|---|
| 7 | review | re-ran 19/19 tests; re-verified parent/R2 checkpoint shas; confirmed frozen 14-branch task set; confirmed the flagged fp64-SVD-in-production vs fp32-in-estimate issue | go, with cost re-measurement required | ~10 min |
| 8 | data | rebuilt `/tmp/opencode/mte_domain_train_windows.pt` deterministically (cleared by tmp cleanup); M0 counts asserted (385071/295825) | cache restored | 109 s |
| 9 | runner | wrote `src/specmuon/run_pilot.py` (+ timing/norm instrumentation in spec_muon; 19/19 tests re-green; step-vs-frozen-operator equality 2.4e-7) | smoke gate PASS | — |
| 10 | smoke | 2-step exact_polar branch on GPU0 | PASS (norm err 2e-16, 56 matrices) | ~2 min |
| 11 | replay | native step 513/1537 vs R2 step1 snapshots | **PASS bit-exact (maxdiff 0.0 both points; s1537 loss = R2's recorded 2.515048086643219)** | ~2 min |
| 12 | bench | GPU cost on real 56-matrix batches | GPU fp64 SVD 10.5 s/step (>20% over estimate) -> implemented same-numerics CPU-pool fp64 SVD (0.61 s); variant opt step 0.79-0.96 s; peak 704 MiB | ~6 min |
| 13 | pilot s512 | shrink rule exercised: 7 branches @ restore 512 | complete; native endpoint loss = R2 record 2.6309; double-run native branches bit-identical | 20.1 min |
| 14 | pilot s1536 | re-estimate (20 min wall, cumulative GPU-active ~11 min << 1 h gate, definition unchanged) then executed | complete | 20.0 min |
| 15 | analysis | `analyze_p1.py` frozen decision | Head NO_EFFECT, Bulk SINGLE_POINT; norm max 3.7e-16; no nonfinite | <1 min |

Totals: pilot wall 40.2 min; CPU-SVD bundle 28.5 min; estimated GPU-active 10.7 min;
GPU0 only; GPU1 untouched; V16 assets untouched (V16 campaign finished, run.lock absent).
