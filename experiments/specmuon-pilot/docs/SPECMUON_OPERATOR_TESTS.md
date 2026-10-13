# SPECMUON operator tests (deliverable 2)

Date: 2026-10-10. Runner (no pytest dependency, exit code 1 on failure):

```
cd spectral_prior_muon_pilot
OMP_NUM_THREADS=8 PYTHONPATH=src python tests/test_operators.py            # 12/12 PASS
OMP_NUM_THREADS=8 PYTHONPATH=src python tests/test_spec_muon_semantics.py  # 7/7 PASS
```

Environment: fr_minimind (python 3.11.16, torch 2.11.0+cu130, CPU), `nice -n 10`,
8 threads. Full source of truth: `tests/test_operators.py`,
`tests/test_spec_muon_semantics.py`.

## Operator tests (`tests/test_operators.py`)

| id | claim verified | result |
|---|---|---|
| t01 | vendored NS5 is **bit-exact** with installed `torch.optim._muon._zeropower_via_newtonschulz` on 5 shapes incl. 1664x512 / 512x1664 | PASS |
| t02 | zero bias recovers ExactPolar: Head(lam=0) = Bulk(lam=0) = Muon^0 = ExactPolar, tol 1e-12 (fp64) | PASS |
| t03 | Native NS5 and ExactPolar are **independent baselines**: normalized gap > 1e-3; unscaled polar orthogonal to 1e-10; NS5 defect > 1e-3; (documented: the norm-matched polar arm carries uniform sv T/sqrt(r) — its "defect" is a pure scale signature) | PASS |
| t04 | Head raises head weight: two-block spectrum head:rest per-direction energy ratio >= 1.4x at lambda=0.25 (closed form 1.46); top-8 directional share +0.02; strictly monotone in lambda on random spectra | PASS |
| t05 | Bulk raises bulk weight: ratio <= 1/1.4; top-8 directional share -0.02; non-head share +0.02; strictly anti-monotone in lambda | PASS |
| t06 | norm matching: all 4 operators match the native per-matrix update Frobenius norm to <= 1e-6 relative, on tall/wide/square shapes | PASS |
| t07 | zero singular values (exact rank-2 input): finite outputs, norm matched; repeated singular values: invariance under basis rotation within exactly-degenerate blocks, tol 1e-9 | PASS |
| t08 | transpose consistency `op(M^T) = op(M)^T` to 1e-9 for all SVD-based ops; native NS5 tall/wide handling produces correct shapes, finite | PASS |
| t09 | bf16-rounded inputs: all operators finite + norm-matched (simulates bf16 optimizer states) | PASS |
| t10 | 1x1 matrices: finite, norm-matched | PASS |
| t11 | high-step reference: spectral-normalized cubic-Newton converges to the SVD oracle (gap <= 1e-3, defect <= 1e-6, 3 shapes); PE single pass finite and near-polar; **same-quintic K=20 does not reach polar** (why a validated convergent iteration is required) | PASS |
| t12 | frozen weight functions positive and bounded: w in [1, 1+lambda] for Head/Bulk; w >= 0 for Muon^p | PASS |

## Optimizer semantics tests (`tests/test_spec_muon_semantics.py`)

| id | claim verified | result |
|---|---|---|
| t01 | `SpecMuon(spectral_op="native_ns5")` is **bit-exact** vs `torch.optim.Muon` over 3 steps, mixed shapes, identical grads (torch.equal) | PASS |
| t02 | `_adjust_lr` reproduces torch's for all shapes and all modes (None/original/match_rms_adamw) | PASS |
| t03 | variant arms apply per-matrix updates of the **same Frobenius norm** as the native arm through the full optimizer (wd=0 isolation; rel <= 1e-5 end-to-end fp32; 1e-6 enforced at operator level in t06) | PASS |
| t04 | unchanged Muon semantics: decoupled wd at unadjusted lr, `buf.lerp_(g,1-mu)`, Nesterov blend, adjusted-lr update — hand-computed replay matches (atol 1e-7 for FMA-level rounding); momentum buffer state matches torch convention | PASS |
| t05 | parameter routing untouched: 56 Muon-route matrices / 34 aux tensors == frozen M0 `routing_report.json`; embedding not in Muon route | PASS |
| t06 | non-2D params rejected (same as torch Muon) | PASS |
| t07 | all variant arms (exact_polar/head/bulk/muon_p) produce finite parameters and correct matrix accounting through the optimizer | PASS |

## Calibration notes (documented deviations)

1. **H3 amendment (pre-data):** the original "top-8 share moves >= 0.02" formulation
   was mis-specified (output shares depend only on the weight profile; bounded linear
   tilts cap the achievable shift for r >> 8). Replaced by the head:rest per-direction
   energy-ratio criterion + strict monotonicity before any real-data analysis
   (see `docs/RESEARCH_HYPOTHESES.md` amendment).
2. **PE cascade fix (pre-data):** cycling the Polar Express 5-step sequence diverges
   (NaN within 2 cycles, observed in unit-test development). The high-step reference
   therefore uses the provable spectral-normalized cubic-Newton iteration
   (`converged_polar_reference`), which converged on 56/56 real matrices and matched
   the SVD oracle exactly (`gap_conv_polar = 0.0`).
3. **End-to-end norm tolerance:** through the fp32 optimizer path, torch's blocked
   norm summation adds ~1e-6..1e-5 relative noise; the 1e-6 contract is enforced at
   the operator (fp64) level, the 1e-5 end-to-end check at the optimizer level.

## Coverage map to the task's required test list

- zero bias -> ExactPolar: t02
- Native != ExactPolar as independent baselines: t03 (+P0-A A3 measurements)
- Head truly raises head relative weight: t04 (+P0-A A5 on real matrices)
- Bulk truly raises bulk relative weight: t05 (+A5)
- norm-match within declared tolerance: t06 (+optimizer t03)
- zero sv / repeated sv / low-rank / transposed shapes / bf16+fp32: t07, t08, t09, t10
- weight decay, momentum, routing, lr semantics unchanged: t01, t02, t04, t05 (semantics file)
- aux AdamW untouched: by construction (SpecMuon only accepts the Muon param group;
  routing check t05 pins the 56/34 split; the aux optimizer is never wrapped)
