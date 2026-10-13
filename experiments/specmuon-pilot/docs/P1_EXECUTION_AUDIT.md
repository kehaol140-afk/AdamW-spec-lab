# P1 EXECUTION AUDIT (deliverable 1)

Date: 2026-10-11. Working dir `spectral_prior_muon_pilot`. All runs on GPU0
(`CUDA_VISIBLE_DEVICES=0`, required by the parent checkpoint's saved CUDA-RNG
device count), process `nice -n 10`, `OMP_NUM_THREADS=1` (SVD parallelism from a
deterministic 16-worker CPU pool). GPU1 (another-user vLLM) never touched. No V16 /
historical Muon project asset modified (verified: all writes under
`spectral_prior_muon_pilot/results/` and `/tmp/opencode/` only).

## Phase 1 — integrity & numerical review (completed before any GPU work)

- Re-ran the full test suite: **19/19 PASS** (12 operator + 7 semantics).
- Parent checkpoints sha256 re-verified:
  `parent_state_step512.pt = 7563daec…`, `parent_state_step1536.pt = 014480c7…`
  (both match the R0_GATE registry); R2 replay references
  `s512_N_step1_model.pt = 35f6f45c…`, `s1536_N_step1_model.pt = 1985dace…`.
- Frozen task set confirmed: 14 branches = 7 arms x {512, 1536} x 256 steps;
  primary endpoint H256 dev NLL (m0 A_dev/B_dev pools, v_q = 0.5(NLL_A+NLL_B));
  per-matrix norm-matching to ||NS5(u)||_F; stop gates as in
  `docs/SPECMUON_FROZEN_PILOT.md` / `configs/pilot_frozen_v1.json`
  (one JSON typo fixed pre-parse: stray quote in `grad_clip` line; no frozen value
  changed).
- **FP64-vs-FP32 SVD issue (pre-flagged by the task) confirmed present**: production
  operators compute the SVD basis in fp64 (`operators.SVD_DTYPE`), while
  `GPU_COST_ESTIMATE.md` had assumed fp32 GPU SVD (3-8 ms/matrix). Measured reality
  below.
- Data-cache gap found and repaired: `/tmp/opencode/mte_domain_train_windows.pt` had
  been cleared; rebuilt deterministically in 109 s via the read-only
  `get_domain_windows()` (window counts re-asserted against the M0 audit:
  A 385,071 / B 295,825).

## Phase 2 — runner (`src/specmuon/run_pilot.py`, newly written)

- Restores full `mte_state_v1` payloads (model + Muon momentum + aux AdamW + RNG),
  R2-hardened (state tensors cloned; RNG restored last).
- Native arms carried by `torch.optim.Muon` itself (SpecMuon(native_ns5) is
  bit-equivalent by test t01; the upstream optimizer is the exact producer of the
  parent/R2 runs). Variant arms use SpecMuon with the frozen spectral operator;
  aux AdamW identical everywhere; wd pinned to the immutable 0.02; LR clock
  `lr_at(gstep)` asserted against R0_GATE anchors (513/1537 to 1e-15).
- All arms share the manifest continuation stream (seed 2030, slice [s, s+256)).
- Safe checkpoint/resume: per-branch `ckpt_last.pt` every 64 steps (atomic rename),
  pruned on success (0 `.pt` files remain in results/).
- Identity logging: per-branch `branch_meta.json` with parent-ckpt sha, code shas
  (5 source files), config sha, torch version, fingerprints; per-step
  `train_metrics.jsonl` (loss, preclip norm, lr, optimizer timings, norm-match err).

## Phase 3 — native replay gate: **PASS (bit-exact)**

`s512`: param maxdiff vs R2 `s512_N_step1_model.pt` = **0.0**; step-513 loss
2.97857004404068. `s1536`: maxdiff = **0.0**; step-1537 loss
**2.515048086643219** — digit-for-digit equal to the only surviving R2 per-step
record (s1536_MH row, whose loss is optimizer-independent). No tolerance was
adjusted; the gate compared parameters, which is stricter than the originally
planned loss check (the parent never persisted step-513/1537 losses — documented
in `results/p1/replay_gate.json`).

## Phase 4 — GPU cost measurement and adjudication

Measured on GPU0, real 56-matrix batches from the parent step-512 momentum buffers
(`results/p1/bench_gpu.json`):

| item | measured |
|---|---|
| native NS5, 56 matrices | 0.022 s |
| exact_polar / head / bulk / muon_p (fp64 SVD **on GPU**) | **10.5 s / step** |
| fp32-SVD polar (informational, not the frozen operator) | 1.08 s / step |
| batched fp64 SVD (informational) | 10.4 s / step |
| optimizer-only step: native / variants | 0.025 s / 0.79-0.96 s (after fix below) |
| peak memory (variant step) | 704 MiB |

Adjudication per the frozen rules: the original estimate (37-47 min) was exceeded
by far (>20%) **if** the fp64 SVD ran on GPU. Before invoking the shrink rule, the
implementation was moved to the *identical frozen numerics* on a different device:
**CPU fp64 LAPACK SVD** (the exact numerics path of the P0 audit), 56 matrices in a
fixed 16-worker pool = 0.61 s; weights + reconstruction on GPU in fp64. Verified:
smoke losses bit-stable across the change; SpecMuon.step vs frozen
`apply_operator` max param diff 2.4e-7 (FMA rounding only); 19/19 tests green.
Re-measured variant optimizer step: **0.79-0.96 s**.

**Shrink rule then exercised as written**: the 512-restore batch was executed
first (20.1 min wall), then re-estimated (1536 batch ≈ 20 min wall; cumulative
estimated GPU-active ≈ 11 min << 1 h GPU gate; no arm/step/eval definition
changed) before executing the 1536 batch, as pre-authorized by the task phase-5
instruction to run both restore points. No budget expansion occurred: total pilot
wall **40.2 min**, of which the CPU-SVD bundle is 28.5 min; estimated
**GPU-active 10.7 min**; peak GPU memory 704 MiB per variant step.

## Phase 5 — frozen P1 execution (complete: 14/14 branches)

- INIT H256 v_q: s512 = 3.000238, s1536 = 2.454251.
- All branches: 256/256 steps, no non-finite loss/grad, no resume failures.
- Per-matrix norm matching held at **max 3.7e-16** relative error across all steps
  and branches (`decision.json:norm_checks`).
- Cross-validations: (i) native step-513/1537 bit-exact vs R2 (Phase 3);
  (ii) s512_native_lr1.0 endpoint train loss **2.6309** = R2's s512_N final loss
  record; (iii) three native branches executed twice (relaunch after a tool-side
  timeout killed the first detached process) — loss/preclip/lr bit-identical
  between executions (free determinism check, `determinism_relaunch_check` all
  true); (iv) data-cache rebuild reproduced M0-audit counts.

## Anomalies & dispositions

| anomaly | disposition |
|---|---|
| GPU fp64 SVD 10.5 s/step (50x over estimate) | root-caused (consumer fp64 + cusolver); replaced by same-numerics CPU-pool SVD; documented; GPU fp32 variant rejected (changes frozen operator) |
| first detached pilot process killed by tool timeout | re-launched with setsid; duplicated native metrics halves verified bit-identical, then deduplicated in analysis |
| parent RNG requires 1 visible CUDA device | runner pins CUDA_VISIBLE_DEVICES=0 |
| /tmp data cache cleared | rebuilt deterministically; M0 counts asserted |

## Resource conduct

V16 assets untouched (V16 GPU campaign completed before launch; `run.lock` absent,
queue 71/71). GPU0 shared — only this pilot. CPU capped at 17 threads total
(1 main + 16 SVD workers), niced. GPU1 untouched throughout.
