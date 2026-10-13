# P0 protocol (frozen v1.0, 2026-10-10)

All analysis is offline, CPU-only, read-only on external assets. Threads capped at 8
(`OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`), process `nice`d, to avoid slowing V16.

## P0-A: native Muon spectral baseline

**Input.** Frozen m1 Muon checkpoints
`muon_transition_experiment/outputs/m1/m1_muon_*/final_state.pt` (format `mte_state_v1`,
integrity-verified by recomputing `payload_sha256`). The analyzed object is the
**Nesterov momentum buffer** `M` (fp32) of each of the 56 Muon-routed matrices at
step 2048. Caveat (registered): the true next-step NS input is
`u = (1-mu) g_next + mu * M` with `mu = 0.95`; since `g_next` is not stored, `M` is the
dominant (weight 0.95) and only real recoverable component. All conclusions are stated
for `M`.

**Provenance pinning.** Native NS5 = `torch.optim._muon._zeropower_via_newtonschulz`
of torch 2.11.0, wheel file sha256 `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52`.
A vendored byte-equivalent copy lives in `src/specmuon/native_ns.py`; the runner asserts
the installed-file sha and cross-checks vendored vs installed outputs bit-exactly on a
probe matrix.

**Quantities per matrix** (SVD in fp64 unless noted):

1. Input spectrum: `sigma(M)`; `x_i = sigma_i / ||M||_F`; top-1/top-8 energy share;
   participation ratio `PR = (sum s^2)^2 / sum s^4`; `sigma_max / ||M||_F`.
2. Native product: `Y5 = NS5(M)` (bf16 internal, as in training; cast to fp32).
   - singular values `s(Y5)`; head energy `E_1, E_8` (of `s^2`);
   - orthogonality defect `D(Y) = ||G - I||_F / sqrt(r)`, `r = min(m,n)`,
     `G = Y^T Y` (tall) / `Y Y^T` (wide);
   - `||Y5||_F` (this is the per-matrix native update norm every variant must match);
   - empirical map points `(x_i, s_i(Y5))` = the NS5 spectral allocation profile `g`.
3. Exact polar (oracle): `P = U V^T` from `svd(M)` (fp64).
   - normalized gap `|| Y5 / ||Y5||_F - P / sqrt(r) ||_F / sqrt(2)` (0 = identical);
   - top-1 left/right singular-vector cosines between `Y5` and `P`;
   - top-8 subspace alignment `|| U8(Y5)^T U8(P) ||_F / sqrt(8)` (1 = identical).
4. High-step convergent reference (validated, not K=20/30 of the same quintic):
   Polar Express coefficient cascade (5-coefficient sequence from modded-nanogpt git
   history, itself citing the Polar Express paper), cycled in fp64 with Frobenius-margin
   normalization until `||G_k - I||_F / sqrt(r) <= 1e-4` or 40 cycles.
   - validation vs SVD oracle: normalized gap (must be `<= 1e-3` to count as converged);
   - gap `NS5 -> PE-limit`: `||Y5/||Y5||_F - Ype/||Ype||_F||_F / sqrt(2)`.
   - This separates "quintic truncation at K=5" from "iteration-family limit" and tests
     whether the NS5-vs-polar gap is an artifact of insufficient iterations.

**Aggregation.** Canonical run `m1_muon_lr1.0_wd0.01_p0.5_s2030` gets the full table;
the other 5 Muon runs get the condensed metric set (E1/E8 of Y5, defect, polar gap) to
give cross-run error bars. Grouped by module type (q/k/v/o/gate/up/down) and layer.

## P0-B: AdamW update spectral diagnostics

**Input (read-only).** Sealed v07 bank
`legacy_fluctuation_response/outputs/e3_confirmatory/v07/seed_{2030..2037}/states/state_step_{046,...,082}.pt`
(8 seeds x 7 steps = 56 states; native `torch.optim.AdamW`, betas (0.9, 0.999),
eps 1e-8 outside the sqrt, wd 0.01 decoupled, fp32 states, MiniMind H256 4-layer).

**Effective update reconstruction (exact native semantics):**

```
t = state['step']                      # per-param true step count
m_hat = exp_avg / (1 - beta1 ** t)
v_hat = exp_avg_sq / (1 - beta2 ** t)
A = m_hat / (sqrt(v_hat) + eps)        # == - (param update) / lr  (up to wd term)
```

Param names: `model` state_dict key order minus tied `lm_head.weight`, aligned to
optimizer state indices 0..45. Only 2D params (embed 6400x256; per layer q 256x256,
k/v 128x256, o 256x256, gate/up 832x256, down 256x832) enter the spectral tables.

**Phasing.** Phase 1 (resource check): seeds {2030, 2033} x steps {46, 64} = 4 states.
Phase 2 (full): all 56 states, only if phase-1 CPU cost confirms < 30 min wall.

**Metrics per matrix:** `PR`, `E_1`, `E_10%` (top ceil(0.1 r) share), `sigma_1/||A||_F`,
`||A||_F`. Cross-state stability: per-name mean/CV of each metric across states;
matrix-to-matrix difference: cosine between vectorized `A` of the same name across all
state pairs (reported as mean/min per name class). All descriptive only (see H4 caveat).

## P0-C: time-frequency data availability

File-level audit (no new computation) of T2a, Muon training records, and V16 logs.
A source qualifies for a true gradient time-PSD only if it contains an actual gradient
(or fixed singular-direction projection) time series that is (i) long enough (>= 64
samples), (ii) sampled continuously (no gaps), (iii) identity-consistent (same
parameter/matrix tracked). Scalar `q_t` / `A_norm` series do not qualify. If no source
qualifies, the gap is reported and no dynamic spectral controller is built.

## Operator contract (frozen; tests in tests/)

For a momentum matrix `M` with SVD `M = U diag(sigma) V^T` (fp64 internals),
`t_i = sigma_i / sigma_1`, target norm `T = ||NS5(M)||_F` (native bf16 output norm):

| op | output | zero bias |
|---|---|---|
| native NS5 | `NS5(M)` (bf16 internal) | n/a (baseline A) |
| ExactPolar | `(U V^T) * (T / ||U V^T||_F)` | is the zero-bias point |
| Head-SpecMuon | `U diag(1 + lambda t_i) V^T` scaled to `T` | `lambda=0` -> ExactPolar |
| Bulk-SpecMuon | `U diag(1 + lambda (1 - t_i)) V^T` scaled to `T` | `lambda=0` -> ExactPolar |
| Muon^p | `U diag(t_i^p) V^T` scaled to `T` | `p=0` -> ExactPolar |

Frozen bias strengths: `lambda = 0.25` (both C and D), `p = 0.5` (E). No grid search.

Numerical tolerances (declared): norm-match relative error `<= 1e-6` (fp64/fp32 paths);
zero-bias equality `<= 1e-12` (fp64); native-vs-vendored NS5 bit-exact; SpecMuon
wrapper with `spectral_op=native_ns5` bit-exact vs `torch.optim.Muon` over 3 steps on
CPU fp32.

Edge cases that must not fail: zero singular values, repeated singular values,
low-rank inputs, transposed shapes (wide/tall consistency `op(M^T) = op(M)^T`),
bf16-rounded inputs, 1x1 matrices.

Semantics that must not change: decoupled weight decay at unadjusted lr, momentum
buffer update, Nesterov blend, LR shape adjustment `sqrt(max(1, A/B))`, parameter
routing (56 Muon / 34 aux), aux AdamW untouched.
