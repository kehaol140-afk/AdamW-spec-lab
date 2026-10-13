# SPECMUON frozen pilot (deliverable 4)

Frozen 2026-10-10, before any P1 execution and before observing any development
result. Machine-readable twin: `configs/pilot_frozen_v1.json`.

## Objective

Test whether a fixed spectral prior on the Muon orthogonalization step improves
short-horizon continued training, against strong equal-budget controls, with
per-matrix update-norm equality across all arms. This is the only P1 question;
P2 (dynamic time-spectral control) is out of scope regardless of outcome (P0-C gap).

## Design

- **Parent**: the verifiable seed-3101 parent trajectory of
  `muon_spectral_allocation/outputs/p0_p1_v1_1/parent/` (m1-recipe Muon 30M MiniMind,
  full `mte_state_v1` states incl. Muon momentum + aux AdamW + 4-way RNG + cursor;
  integrity sha verified on load). Two established restore points: **step 512** and
  **step 1536** -> branches of **256 consecutive steps** (513-768, 1537-1792).
- **Not two seeds**: the two restore points are two phases of ONE trajectory with ONE
  data stream; they are paired evidence (a candidate must win at BOTH), never pooled
  as independent samples.
- **Identical conditions across arms**: same batch stream (parent manifest
  continuation), same RNG restoration and advancement, same grad clip (1.0), same
  schedule clock (parent cosine evaluated at absolute step; LR multipliers apply to
  the scheduled value), same eval pools (m0 A_dev/B_dev, 256-token windows), same
  256-step budget, same per-matrix update Frobenius norm (||NS5(u)||_F, enforced by
  the operator contract and asserted in tests).
- **Primary judgment criterion**: final **H256 dev NLL** `v_q = 0.5*(NLL_A + NLL_B)`
  at branch end (256-token dev pools; the m1/V-series H256 convention). Secondary:
  endpoint train NLL, per-domain NLL, wall-clock, optimizer overhead.

## Arms (7 x 2 restore points = 14 branches)

| arm | optimizer | LR | role |
|---|---|---|---|
| native_lr1.0 | SpecMuon(native_ns5) | 1.0x | native baseline (= replay carrier) |
| native_lr0.5 | SpecMuon(native_ns5) | 0.5x | same-budget LR-grid control |
| native_lr2.0 | SpecMuon(native_ns5) | 2.0x | same-budget LR-grid control |
| exact_polar | SpecMuon(exact_polar) | 1.0x | equal-norm exact-polar control |
| head_lam0.25 | SpecMuon(head), lambda=0.25 | 1.0x | candidate C |
| bulk_lam0.25 | SpecMuon(bulk), lambda=0.25 | 1.0x | candidate D |
| muon_p0.5 | SpecMuon(muon_p), p=0.5 | 1.0x | p-power variant, non-authoritative (SOURCE_BLOCKED label) |

WD, momentum, Nesterov, NS5 coefficients, eps, LR shape adjustment, parameter
routing (56 Muon / 34 aux) and the aux AdamW arm are identical everywhere
(bit-exact native equivalence proven by tests). No bias-strength grid: single frozen
conservative values (lambda=0.25, p=0.5).

## Execution order and gates (all must pass, in order)

1. **Runner smoke test** (CPU or idle-GPU, 2 steps): shapes, norm logging, eval path,
   checkpoint save/load. No branch data consumed.
2. **Native replay gate**: restore parent step 512, execute step 513 with the native
   arm under the parent's batch manifest; the train loss must match the parent's
   recorded step-513 loss to < 1e-12. **If it fails: STOP the pilot and diagnose the
   implementation.** (The parent states were previously proven bit-replayable by the
   R2 round of `muon_spectral_allocation`; failure would indicate OUR harness bug.)
3. **GPU gate** (see `GPU_COST_ESTIMATE.md`): all of — V16 GPU tasks terminated;
   shared GPU0 use approved; replay passed; this config frozen; estimate <= 1 GPU-h.
   While V16 lives: **wait, do not run**. Never touch GPU1 (another-user vLLM).
4. **Pilot**: 14 branches in frozen order (512-point batch first, then 1536-point),
   logging per-step loss/grad-norm/native-norm/optimizer-overhead and branch-end
   H256 eval + param fingerprint.

## Fairness rules (frozen)

- Per-step exact SVD is an *oracle for validating the spectral prior*, not a
  production proposal: overhead seconds are logged per branch and reported next to
  every quality number (same-token vs same-wall-clock comparisons both reported).
- If measured GPU cost exceeds the pre-estimate by > 20%, **shrink or stop** (drop the
  1536-point batch after reporting 512-point results), never silently degrade
  comparability (no shorter branches, no skipped arms, no eval subsampling).
- If any arm produces non-finite loss or diverges (loss > start + 1.0), record and
  exclude only that arm with a DIVERGED flag; baselines unaffected.

## Decision tree (H5, frozen thresholds)

Let `d(arm, point) = v_q(native_lr1.0) - v_q(arm)` at each restore point (positive =
arm better), `bestLR(point) = min over LR-grid arms`.

- **PROMISING** (Head or Bulk): `d > 2e-3` vs native at BOTH points AND
  `v_q(arm) < bestLR - 0` (strictly beats the whole LR grid) at BOTH points AND
  `d(arm) - d(exact_polar) > 1e-3` at BOTH points.
- **POLAR_ONLY**: exact_polar meets 1+2 but no candidate meets 3 -> the gain is the
  NS5-vs-polar approximation gap, not the spectral prior; verdict NO new spectral
  method.
- **LR_DOMINATED**: some candidate beats native but not the LR grid -> NO_EFFECT
  (amplitude, not spectrum — consistent with the prior project's R2F finding).
- **SINGLE_POINT**: improvement at only one restore point -> NO_EFFECT (insufficient).
- Anything else -> NO_EFFECT. All numbers reported with both points separately;
  no cherry-picking, no averaging across points.
- SAMuon/AMO/MuPS: no claims in any direction (SOURCE_BLOCKED,
  `BASELINE_SOURCE_AUDIT.md`).

## Relation to SpecLoRA (scope statement)

Research idea borrowed: *moderate structural prior + preserved adaptability*. Three
distinct objects, not to be conflated: SpecLoRA's Fourier PSD over LoRA updates; the
Muon matrix singular-value spectrum (this pilot's ONLY object); the gradient-history
time PSD (P0-C: DATA_GAP, not measurable with existing assets). Until a fixed
spectral prior beats the strong baselines above: **no time-spectral selector
training, no adaptive alpha, no composite optimizer, no AdamSpec design**. If a
candidate is PROMISING: plan P2 = dynamic spectral control (requires first
instrumenting a dedicated run with fixed-direction gradient projections), and only
then design AdamSpec against AdamW / Muon^p / PolarAdamW head-to-head.
