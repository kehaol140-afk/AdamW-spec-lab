# P0 spectral audit (deliverable 1)

Date: 2026-10-10. All computations offline CPU, read-only on external assets.
Config: `configs/p0a_audit_v1.json`, `configs/p0b_audit_v1.json` (frozen before runs).
Raw tables: `results/p0a/`, `results/p0b/`. Hypotheses: `docs/RESEARCH_HYPOTHESES.md`.

---

## P0-A — native Muon spectral baseline (m1 frozen checkpoints)

**Object.** Nesterov momentum buffers `M` (fp32) of the 56 Muon-routed matrices at
step 2048, from all 6 m1 Muon runs (`m1_muon_lr{0.5,1,2}_wd{0.01,0.02}_p0.5_s2030`;
payload-sha verified). Registered caveat: the true next-step NS input is
`u = 0.05*g_next + 0.95*M`; `g_next` is not stored, so `M` (weight 0.95) is the
recoverable object. All 6 runs share data seed 2030 — cross-run stability below is
stability across LR/WD, not across data seeds.

**Provenance.** Vendored NS5 is bit-exact with the installed torch 2.11
`_zeropower_via_newtonschulz` (probe matrices, `vendored_bitexact_with_installed=true`);
installed wheel sha256 equals the frozen pin.

### A1. The momentum spectrum is strongly non-uniform (H1: CONFIRMED)

Canonical run `m1_muon_lr1.0_wd0.01` (56 matrices):

| metric (input `M`) | mean | min | max |
|---|---|---|---|
| top-1 energy share `E1` | 0.184 | 0.023 | 0.729 |
| top-8 energy share `E8` | 0.389 | 0.104 | 0.855 |
| participation ratio `PR` (of `r`, mean r=439) | 47.5 | 1.9 | 229.2 |

Uniform reference: `E8_uniform = 8/439 = 0.018`. The real momentum spectra carry
**~21x more energy in their top-8 directions than a flat spectrum** (H1 predicted
>1.3x). Module ranking (E8): v 0.690 > k 0.472 > q 0.419 > o 0.384 > gate/up/down
0.25. So yes — **an operable non-uniform spectrum exists** (report Q1).

### A2. NS5 acts as a scalar map g on the normalized spectrum; the profile collapses

For `x_i = sigma_i/||M||_F` and `s_i = sigma_i(NS5(M))`, all 24,484 (x, s) points from
the canonical run collapse onto one curve (module-independent):

| x-bin | n | g mean | g std |
|---|---|---|---|
| [0, 0.05) | 20,071 | 0.839 | 0.152 (steep transition + decay tail) |
| [0.05, 0.5) | 4,500 | ~1.16 | ~0.03 |
| >= 0.5 | 9 | ~1.16 | ~0.03 |

Interpretation: **NS5 = soft rank-truncation + near-polar plateau**. Every direction
with `sigma >= 5% ||M||_F` is mapped to a plateau amplitude ~1.16 (bimodal 1.137/1.203
at bf16 resolution); directions below are progressively attenuated toward 0. The
plateau occupies ~84-91% of directions (per module: q 0.840 ... down 0.907).
Consequence: the NS5 *output* spectrum is nearly flat (out_E8 = 0.033 vs uniform
0.018) — i.e. **native Muon already destroys most of the input spectral shape**, and
whatever spectral prior one wants to inject must fight this flattening.

### A3. NS5 vs exact polar: sizeable, structured, iteration-count-irrelevant gap (H2: mechanism CONFIRMED with one amendment)

| quantity (canonical run) | mean | min | max |
|---|---|---|---|
| normalized gap `||Y5/||Y5|| − P/||P|||| / sqrt(2)` | 0.133 | 0.112 | 0.211 |
| orthogonality defect of NS5 output | 0.342 | 0.303 | 0.439 |
| gap(NS5, converged reference) | 0.133 | 0.112 | 0.211 |
| gap(converged reference, SVD polar) — validation | **0.000** | 0.000 | 0.000 |

- The **converged cubic-Newton reference** (spectral-normalized, provable monotone
  convergence; `converged_polar_reference`) reached defect <= 1e-6 in 10-35 iterations
  and matched the SVD oracle `U V^T` with gap 0.0 (< 1e-12) on **56/56** matrices —
  the high-step reference is validated, not assumed.
- `gap(NS5, converged) == gap(NS5, polar)` to 5 decimals: the NS5-vs-polar gap is
  **entirely the quintic family's fixed-point error, not insufficient iterations**.
  Re-running the same quintic with K=20 stays away from polar (unit test t11), as the
  prior analytic finding `f(1)=0.701` predicted.
- A single pass of the Polar Express 5-step cascade lands closer to polar than NS5
  (gap 0.081 vs 0.133, defect 0.209 vs 0.342) — a coefficient-engineered same-budget
  family exists that halves the truncation error. Cycling PE diverges (NaN, observed),
  which is why it is used only as a diagnostic.
- **H2 amendment (pre-registered prediction partially wrong):** the predicted
  "top-1 singular-vector cosine > 0.99 between Y5 and P" is empirically MEANINGLESS:
  both spectra are nearly flat, so "the top direction" is numerically ill-posed
  (observed cosines ~0.04). The correct verification of basis-preservation is the
  A2 profile collapse itself (std ~0.03 within the plateau across all matrices),
  which confirms the scalar-map mechanism.

### A4. Update energy shares and module differences

Output (post-NS5) head energy: `out_E1 = 0.004`, `out_E8 = 0.033` (uniform 0.002/0.018)
— a residual ~1.8x head concentration survives the flattening (the plateau's soft
edge). Defect and gap are module-dependent: q/o (square 512x512) have the largest
gaps (0.17-0.21, layers 1/3/4/5 q_proj worst 0.211); mlp/k/v ~0.11-0.12. So
**ExactPolar vs NS5 is a genuinely different update especially for attention
square-projections**.

### A5. Operator preview on real matrices (report Q3)

Frozen strengths lambda=0.25, p=0.5 (canonical run, 56 matrices; `k10 = r//10`):

| arm | head:rest per-direction energy ratio | top-decile share |
|---|---|---|
| ExactPolar | 1.000 (by construction) | 0.099 |
| Head-SpecMuon (0.25) | 1.114 [1.026, 1.207] | 0.109 |
| Bulk-SpecMuon (0.25) | 0.912 [0.846, 0.981] | 0.091 |
| Muon^p (0.5) | 4.48 [2.28, 9.28] | 0.322 |

Head/Bulk deliver a moderate, sign-correct, strictly monotone tilt (unit tests t04/t05)
— consistent with the frozen "conservative" design. Muon^p p=0.5 is a much stronger
reshaping (4.5x). **Caveat for P1 interpretation: because NS5 already flattens the
spectrum (A2), the Head/Bulk arms differ from ExactPolar mainly by re-introducing a
mild version of the momentum spectrum's own tilt.**

### A6. Cross-run stability

Across the 6 runs (LR 0.5-2x, WD 0.01/0.02), every aggregate moves in the 3rd-4th
decimal (out_E8 0.0327-0.0328, defect 0.342-0.344, gap 0.1326-0.1333, head ratios
1.108-1.114). The spectral statistics of the trained momentum state are effectively
invariant to LR/WD at fixed data seed.

---

## P0-B — AdamW effective-update spectra (sealed v07 bank, read-only)

**Reconstruction.** `A = m_hat/(sqrt(v_hat)+eps)` with native semantics:
`m_hat = m/(1-0.9^t)`, `v_hat = v/(1-0.999^t)`, `eps=1e-8` outside the sqrt, per-param
true step `t` from the stored counter. Phase 1 (4 states) confirmed trivial cost;
phase 2 covered **all 56 states** (8 seeds x 7 steps, steps 46-82 of 128; MiniMind
H256, 4 layers, 29 2D modules per state = 1,624 matrix rows).

### B1. AdamW effective updates are extremely head-concentrated (H4 shape part: CONFIRMED)

| module | E1 | E10% | PR/r | sigma_1/||A||_F |
|---|---|---|---|---|
| embed | 0.808 | 0.965 | 0.006 | 0.898 |
| v | 0.520 | 0.942 | 0.026 | 0.716 |
| o | 0.510 | 0.963 | 0.015 | 0.704 |
| gate/up | 0.450-0.454 | 0.925-0.927 | 0.019 | 0.659-0.661 |
| k/q | 0.409-0.411 | 0.860-0.897 | 0.023-0.040 | 0.630-0.635 |
| down | 0.392 | 0.924 | 0.021 | 0.617 |

Top decile of directions carries **86-96% of update energy** — an order of magnitude
more concentrated than the Muon momentum matrices of P0-A (top 1.8% of directions
carrying 39%). The spectral non-uniformity available to a hypothetical AdamSpec is
large (descriptive only — see the pre-registered caveat; this does not show reshaping
would help).

### B2. Cross-state stability and matrix-level differences

- Spectral-shape statistics are moderately stable: E1 CV across all 56 states
  0.23 (mean; max 0.32) — decomposed as CV 0.175 within a fixed step across seeds,
  CV 0.180 within a fixed seed across steps. There is a systematic flattening trend:
  mean E1 declines 0.547 (step 46) -> 0.397 (step 82). H4's "CV < 0.2 across states"
  holds only after separating the step trend; the naive pooled claim is REFUTED
  (0.227 > 0.2).
- The fine-grained direction of `A` is essentially seed-orthogonal: cross-seed cosine
  of vectorized same-name `A` is 0.000 +/- 0.06 for all 5,684 same-step pairs and all
  cross-step pairs. Within a seed it decays fast with step distance (cos ~0.45 at
  dstep 6, ~0.05 at dstep 36). **Only the spectral shape, not the directions, is a
  stable object** — any spectral prior for AdamW must be shape-level, not
  direction-level, at this stage of training.

---

## P0-C — time-frequency data availability (gradient time-PSD)

File-level audit of every candidate source (paths, cadence, length, continuity,
identity). Verdict per source:

| source | quantity | cadence/length | qualifies for gradient time-PSD? |
|---|---|---|---|
| T2a `raw_natural.pt` `G_post_clip` | full post-clip gradient matrices, 6 fixed matrices | 1/batch, **10 samples**, frozen params (zero-update replay) | NO — 10 samples at one frozen state measures batch noise, not training dynamics |
| T2a `H_i` | one-shot blends with common M* | 10 | NO — non-recursive by design; report itself disclaims time-covariance use |
| T2b C-tail | 128 consecutive frozen-theta gradients | streamed then **deleted in code** (`del grads_c`) | NO — never persisted |
| T2b real branches | 16 branches x 138 real steps | endpoints only | NO |
| m1 `obs.jsonl` | grad_global_norm (scalar) | every 100 steps, 21 points | NO — scalar, sparse; not a gradient |
| m1 `metrics.jsonl` | loss/grad_norm | 26 sparse rows / 2048 steps | NO |
| V16 `campaign_v16/raw/steps/*.jsonl` | per-step q_t, A_norm, D_norm (update/target scalars) | every step, 256/branch | NO — update-direction and displacement norms, not gradients; a q_t PSD would not be a gradient PSD |
| V16 `direction_signals.csv` | per-document gradient projections onto SVD head/rest | doc axis, not time | NO — cross-sectional |
| V16 `anchors.jsonl` (g/A/m norms, cos, sigma1) | anchor-step matrix summaries | **file absent** (B-stage pending) | NO — not yet produced; would still be scalars |

**Conclusion: DATA_GAP.** No existing source provides a real gradient (or fixed
singular-direction projection) time series that is long (>=64), continuous, and
identity-consistent. Per the task rules, **no dynamic spectral controller is built**;
P2 stays blocked until instrumentation exists (e.g. persisting per-step projections
onto fixed singular directions during a future dedicated run — not V16's).

---

## Hypothesis verdict summary (P0)

| hyp | verdict |
|---|---|
| H1 non-uniform momentum spectrum | CONFIRMED (21x top-8 concentration; module-dependent) |
| H2 NS5 != polar, structured scalar-map gap | CONFIRMED with amendment: gap = 0.133 (quintic fixed-point error, not iteration count); direction-cosine formulation ill-posed, replaced by profile collapse |
| H3 operators deliver sign-correct measurable shaping + norm/edge contracts | CONFIRMED (all 19 unit tests; real-matrix magnitudes in A5) |
| H4 AdamW A non-uniform + stability | shape CONFIRMED (E10% 0.86-0.96); pooled CV claim REFUTED (0.227 >= 0.2; step-trend + seed noise), directions seed-orthogonal |
| H5 | untestable in P0 (P1 screening gate, frozen) |

## Answers to the six report questions (P0 portion)

1. **Operable non-uniform spectrum?** Yes for Muon momentum (21x) and even more so
   for AdamW effective updates (top decile ~92%). But NS5 already flattens most of it.
2. **NS5 vs exact polar?** Normalized gap 0.133 (13% of maximum), defect 0.34,
   entirely a truncation/fixed-point error (validated: gap to converged iteration ==
   gap to SVD oracle; converged == oracle exactly). PE-style coefficients halve the
   gap at the same budget. Worst on q/o square projections (0.17-0.21).
3. **Do Head/Bulk change the spectral shape as intended?** Yes, moderately
   (head:rest ratio 1.11x / 0.91x at lambda=0.25, top-decile share 0.109/0.091 vs
   0.099), Muon^p strongly (4.5x / 0.322). All contracts (zero-bias, norm-match,
   edge cases) hold by test.
4-6. Require the P1 pilot (frozen design + gate in the companion docs).
