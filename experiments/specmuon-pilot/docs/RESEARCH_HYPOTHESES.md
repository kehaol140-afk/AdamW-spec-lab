# Research hypotheses (frozen before P0 execution)

Date frozen: 2026-10-10. These hypotheses and all thresholds are declared before any
P0 measurement is looked at. Nothing below may be re-tuned after observing results.

## H1 (operative non-uniform spectrum)

The Nesterov effective momentum matrices `M` of a real Muon run (m1, step 2048) have
materially non-uniform singular spectra: participation ratio `PR < 0.9 * r` and
top-8 energy share `E8 > 1.3 * (8/r)` for most of the 56 Muon-routed matrices
(`r = min(m, n)`, `E_k` defined in `P0_PROTOCOL.md`).

## H2 (NS5 != exact polar, in a structured way)

Native NS5 (torch 2.11, coeffs `(3.4445, -4.7750, 2.0315)`, K=5, bf16 internal) is a
fixed scalar map `g` acting on singular values `sigma_i / ||M||_F` and preserves the
singular basis of `M`. Therefore:

- `Y5 = NS5(M)` and `P = U V^T` share singular directions (top-1 subspace cosine > 0.99);
- the entire NS5-vs-polar gap is (up to bf16 noise) a **singular-value profile gap**:
  `s_i(Y5) = g(x_i)`, `x_i = sigma_i / ||M||_F`, vs the all-ones profile of polar;
- increasing K with the *same* coefficients does **not** converge to polar
  (prior analytic finding `f(1) = 0.701`; fixed points of the quintic differ from 1),
  so the high-step reference must be a validated convergent iteration
  (Polar Express cascade) cross-checked against the SVD oracle.

## H3 (spectral shaping moves energy measurably)

**Amendment 2026-10-10 (pre-data, during unit-test calibration):** the original
formulation ("top-8 energy share moves by >= 0.02 absolute") was mis-specified:
output energy shares depend only on the weight profile `w(t_i)`, so for `r` directions
a bounded linear tilt (lambda = 0.25, max/min weight ratio 1.25) caps the possible
top-8-of-r share shift well below 0.02 for large r. This was discovered from synthetic
unit tests before any real-data analysis. Calibrated formulation:

At the frozen conservative bias `lambda = 0.25` (Head/Bulk linear tilts) and `p = 0.5`
(Muon^p), the operators must, on non-uniform spectra:

- change the per-direction head:rest energy ratio (mean squared output weight of the
  input-top-8 directions vs the rest) by at least 15% in the intended direction
  (Head: >= 1.15x; Bulk: <= 1/1.15x), with strict monotonicity in the bias strength;
- move the input-basis top-8 directional energy share strictly in the intended
  direction; magnitudes on REAL momentum matrices are reported in P0-A (no threshold);
- preserve the per-matrix Frobenius norm to `<= 1e-6` relative error and never produce
  non-finite values (including rank-deficient, repeated-singular-value, transposed,
  and bf16-rounded inputs).

## H4 (AdamW effective update is also non-uniform)

For the v07 bank, the effective AdamW update `A = m_hat / (sqrt(v_hat) + eps)`
(bias-corrected, eps outside the sqrt, exactly as native torch AdamW computes it) has
non-uniform spectra on matrix modules: `PR < 0.9 * r` for most modules, and the
per-module spectral statistics are stable across seeds/steps (CV of top-1 energy share
< 0.2 across the 56 states).

**Descriptive-only caveat (pre-registered):** observing spectral non-uniformity in H1/H4
does NOT by itself demonstrate that spectral reshaping improves training. Only the P1
pilot against frozen strong baselines can test that.

## H5 (P1 decision hypotheses — frozen screening thresholds)

Candidate screening gate (declared now, before any development result is observed;
historical eval resolution reference: m1 LR-grid final dev NLL differences across
3x LR are ~0.05-0.3 nats; muon_spectral_allocation's R2 256-step continuation effects
~3e-3..7e-2 nats; noise/resolution floor ~1e-3 nats):

A spectral candidate (Head-SpecMuon or Bulk-SpecMuon) is *promising* only if ALL hold:

1. Beats native Muon NS5 (same LR) on final H256 dev NLL at BOTH restore points
   (512 and 1536), each by more than `2e-3` nats (2x the ~1e-3 resolution floor);
2. Beats the best arm of the equal-budget native Muon LR grid {0.5x, 1x, 2x} at both
   restore points;
3. The ExactPolar control does NOT explain the entire gain: candidate advantage over
   ExactPolar > `1e-3` nats at both restore points;
4. Gain is not carried by a single module group alone (no per-group ablation in P1;
   this item is checked in P1 by the two restore points acting as different regimes
   plus the LR-grid control).

If only one restore point improves, or the gain is <= the LR-grid best, or ExactPolar
captures the gain, the verdict is NO_EFFECT (see `SPECMUON_FROZEN_PILOT.md` for the
full decision tree).

## Non-goals (frozen)

- No time-frequency PSD claims: P0-C only audits data availability; no dynamic
  spectral controller is built in P0/P1 regardless of P0-A/B outcomes.
- No adaptive alpha, no learned spectral selector, no composite optimizers before a
  fixed spectral prior beats the strong baselines above.
- No modification of weight decay, momentum, parameter routing, LR semantics, or the
  aux-AdamW behavior of non-Muon parameters.
- No large bias-strength grid search: a single conservative value per operator.
