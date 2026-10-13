# BASELINE source audit (deliverable 3)

Date: 2026-10-10. Purpose: establish, per baseline, whether an authoritative public
source (paper + code + config) is available for a reproducible head-to-head, or the
baseline must be marked SOURCE_BLOCKED and excluded from superiority claims.

## Network egress status (re-verified today)

- `webfetch` to arxiv.org (2505.16932, 2608.25990, 2508.17535): **transport error** on
  all three.
- `curl` to arxiv.org / github.com: HTTP code `000` (connection fails).
- Consistent with `muon_spectral_allocation/docs/NEXT_RESEARCH_DECISION_CN.md`
  (2026-09-29): gateway blocks HTTPS to arxiv/github.

## Baseline-by-baseline

| baseline | authoritative source available? | status | evidence / local fallback |
|---|---|---|---|
| Native Muon NS5 | YES (local, frozen) | VERIFIED_LOCAL | `torch.optim._muon` torch 2.11.0, wheel sha256 `38e95918...` matches the pin in `muon_transition_experiment/outputs/m0/muon_freeze.json`; vendored copy bit-exact (test t01); coefficient provenance comment in torch source cites Keller Jordan's post/permlink |
| ExactPolar (equal-norm) | YES (mathematical) | VERIFIED_LOCAL | `P = U V^T` from SVD; plus a provable convergent iteration cross-validated against the SVD oracle on 56/56 real matrices (P0-A A3) |
| Muon LR-grid control | YES (local) | VERIFIED_LOCAL | same optimizer, LR x {0.5, 1, 2}; recipe template = `m1` frozen configs |
| Muon^p (spectral power) | NO (paper+code unfetchable) | **SOURCE_BLOCKED** (principled reimplementation only) | Candidate primary source: MuPS "Muon with Power Scheduling" (arXiv 2508.17535), unfetchable today; the name/definition could not be independently verified against the authors' text. Implemented here from first principles as `U diag((sigma/sigma_1)^p) V^T`, p=0.5, norm-matched (documented definition, not claimed to replicate any published algorithm). Any P1 result is reported as "our p-power variant", NOT as beating the published Muon^p/MuPS |
| SAMuon / SAMuon-lite | NO | **SOURCE_BLOCKED** | arXiv 2608.25990 unfetchable; no local cache (searched). Prior project's 9 replication-screening questions remain UNKNOWN. Cannot be included as a comparator and no superiority over it may be claimed |
| AMO | NO | **SOURCE_BLOCKED** | arXiv 2605.17806 unfetchable; same status as in `muon_spectral_allocation/docs/RECIPE_AND_AMO_COMPARISON.md` |
| Polar Express (coefficients) | PARTIAL (secondary provenance) | VERIFIED_NUMERIC | Coefficient sequence recovered from the local `modded-nanogpt` git history (commit 9350f4d) with in-repo citation of arXiv 2505.16932; original paper unfetchable. Used only as a *diagnostic* single-pass reference (PE5); its numerical behavior is validated against our SVD oracle rather than trusted |
| SpecLoRA | concept reference only | N/A | Fourier-PSD LoRA adaptation idea; distinct object class from Muon singular-value shaping (see SPECMUON_FROZEN_PILOT.md §scope). Not a comparator |

## Consequences for P1 claims (frozen)

1. The strong-baseline set is exactly: native NS5, equal-norm ExactPolar, same-budget
   Muon LR grid, Head-SpecMuon, Bulk-SpecMuon (+ our p-power variant, labeled as
   non-authoritative reimplementation).
2. Any sentence of the form "beats SAMuon/AMO/MuPS" is prohibited in all outputs
   unless their sources become fetchable and a reproducible config is obtained;
   in that case rerun the comparison before claiming.
3. The `Muon^p` arm's role is exploratory (does a strong power reshaping help?), not
   a claim against the published method.
