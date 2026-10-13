# Release integrity and preservation notes

This release is derived from `ADAMW_SPEC_LAB_REPRO_READY.tar.gz` (SHA256 `0404be70fe6ae60179827cd4a2cec3dc60e89f6266ffb63a3909e6b6c7111dfe`). The frozen research source files, evaluation data and original results remain **byte-for-byte identical**. Changes are limited to packaging / verification files: `.gitignore`, `.gitattributes`, `RELEASE_INTEGRITY.md`, `tools/verify_release_integrity.py`, `.github/workflows/cpu_ci.yml`, root `manifest.json`, root `README.md` (corrected the V16-C 3-seed versus subsequent single-seed evidence wording), `docs/03_EXPERIMENT_DESIGN_INDEX.md` and `docs/ERRATA_AND_RECONCILIATION.md` (corrected 283/287 run-index arithmetic), and directory `MANIFEST.sha256` checksum lists.

## Why checksum lists were repaired
The original release listed `manifest.json` in itself and each per-directory `MANIFEST.sha256` in itself. Hashing a file containing its own SHA256 gives an impossible self-reference; all other original archive members matched. The corrected manifests **exclude their own checksum file**, with unchanged commitments for frozen experiment data. The root `manifest.json` excludes only itself; its `n_files` and `total_bytes` describe all *other* tracked files. The external `ADAMW_SPEC_LAB_GITHUB_VERIFIED.tar.gz.sha256` authenticates the final compressed release.

Use `python3 tools/verify_release_integrity.py` after unpacking (no third-party libraries). CPU result recomputation is a **separate** operation: `python3 tools/repro.py recompute --all`; it may rewrite output JSON byte formatting/key order without changing numeric content. The release integrity check in CI runs *before* recomputation; use `git diff` before committing any subsequent recompute output changes.

R0 numerical checks pass. R1/R2 training cannot be run without separately stored data, caches, parent optimizer checkpoints and GPU resources. See `docs/REPRODUCIBILITY_MATRIX.csv` and `docs/ASSET_RESTORATION_GUIDE.md`. Uploaded materials do **not** back up these large required assets.

The previously used V16 `campaign_v16/eval_sets.json` and six `final_holdout` *result rows* are preserved for historical traceability; this is not the unopened `holdout_v2` dataset, which is absent. If publishing to an unrestricted public repository, confirm you wish to disclose the historical V16 tokenized evaluation examples.
