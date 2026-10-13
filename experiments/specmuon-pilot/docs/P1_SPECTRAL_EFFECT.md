# P1 SPECTRAL EFFECT (deliverable 3)

All numbers from `results/p1/P1_RESULTS.csv` (primary endpoint: H256 dev NLL
`v_q = 0.5(NLL_A+NLL_B)` on the frozen m0 dev pools; lower is better). Same parent
trajectory (seed 3101), two restore points — paired phases, NOT independent seeds.
Every arm applied per-matrix updates of identical Frobenius norm
(max relative mismatch 3.7e-16), identical data stream, RNG, schedule clock, wd,
clip, and aux AdamW.

## Endpoint table

| arm | s512 v_q | d vs native | s1536 v_q | d vs native |
|---|---|---|---|---|
| INIT (restore) | 3.000238 | — | 2.454251 | — |
| native_lr1.0 (NS5) | 2.798868 | 0 | 2.402358 | 0 |
| native_lr0.5 | **2.724887** | **+0.073981** | **2.394462** | **+0.007896** |
| native_lr2.0 | 2.916667 | −0.117799 | 2.441317 | −0.038959 |
| exact_polar | 2.799642 | −0.000774 | 2.402456 | −0.000098 |
| head λ=0.25 | 2.803964 | −0.005096 | 2.403357 | −0.000999 |
| bulk λ=0.25 | 2.796071 | +0.002796 | 2.401650 | +0.000708 |
| p-power 0.5 (non-authoritative) | 2.850063 | −0.051195 | 2.415177 | −0.012819 |

## The seven questions

**1. Real training gap between Native NS5 and ExactPolar?**
At most ~1e-3 nats, and in favor of NS5: exact_polar is worse by 0.00077 (s512)
and 0.00010 (s1536). The P0-A operator-level gap (13.3% normalized Frobenius
difference, 0.34 orthogonality defect) collapses to a ≤1e-3 endpoint difference
under real training — the NS5-vs-polar approximation is **functionally
inconsequential** at this scale. (Same sign at both points.)

**2. Do Head/Bulk beat tuned Native Muon?**
No, overwhelmingly. The equal-budget LR grid's best (lr0.5) beats head by
0.079/0.009 nats and bulk by 0.071/0.007 nats (s512/s1536). The dominant lever is
again update amplitude, exactly as the prior muon_spectral_allocation R2F round
concluded for its own interventions.

**3. Gain direction: head or spectral body?**
Consistently the body, but marginally: bulk > native > polar > head at BOTH points
(bulk +0.0028/+0.0007; head −0.0051/−0.0010 vs native). The head-tilt is mildly
harmful, the bulk-tilt is mildly helpful, and the strong power reshaping
(p-power 0.5) is clearly harmful (−0.051/−0.013). Sign ordering is stable across
restore points; magnitudes are at or below the ~1-3e-3 resolution band and shrink
toward zero at the later restore point.

**4. Does the effect survive exact update-norm equalization?**
The norm confound was removed to 3.7e-16 and nothing survives it at threshold:
bulk's +0.0028 (s512) sits just above the 2e-3 screen at one point but decays to
+0.0007 (< threshold) at the other. Whatever small ordering exists is genuinely
directional, not amplitude — but it is not a confirmed effect.

**5. Does wall-clock erase the quality benefit?**
There is no quality benefit to protect. For completeness: oracle arms cost
5.2-6.3x native branch wall (mean step 1.02-1.10 s vs 0.18 s; the frozen fp64-SVD
oracle bundle is 28.5 of the 40.2 total pilot minutes; estimated GPU-active only
10.7 min). A ≤3e-3-nat directional hint cannot justify a 5-6x optimizer overhead.

**6. Are the two restore points consistent?**
Yes in ordering (lr0.5 < bulk < native < polar < head < p-power < lr2.0 at s512;
lr0.5 < bulk < native < polar < head < p-power < lr2.0 at s1536), no sign flips
for any arm; effect magnitudes shrink at the later point where the LR lever itself
also shrinks (0.074 -> 0.008). This is the signature of regime-dependent amplitude
sensitivity, not of a spectral mechanism.

**7. Could an SVD-approximation artifact explain the (null) results?**
No candidate passed, so strictly there is nothing to explain; for completeness:
exact_polar differs from native by −0.0008/−0.0001, while candidates deviate from
exact_polar by −0.0043..+0.0036 — the candidate deltas are their own spectral-tilt
effects measured against the polar control, not NS5-truncation artifacts.

## Interpretation (mechanism, tied to P0)

P0-A showed NS5 already flattens the momentum spectrum into a ~1.16 plateau
(top-decile share 0.099 after vs 0.32 before), and P0-B showed the surviving
head/bulk tilts at λ=0.25 shift head:rest per-direction energy by only
1.11x/0.91x. The P1 endpoints now demonstrate those tilts translate into ≤3e-3
nat directional hints that decay with training phase, while a plain 0.5x LR moves
the same endpoint by 0.074/0.008 nats. In this setting the orthogonalized update's
spectral allocation is a second-order knob; the update's overall scale is
first-order.

The p-power arm (our first-principles variant; SOURCE_BLOCKED — NOT a reproduction
of the published Muon^p/MuPS) shows that making the tilt strong enough to matter
(4.5x head:rest ratio) is net harmful, bounding the useful direction to the mild
regime where it is indistinguishable from noise.
