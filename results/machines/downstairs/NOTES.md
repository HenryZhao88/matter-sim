# Downstairs PC export (Ryzen 9 5900X, 32 GB, RTX 3080 Ti; Windows 11, Python 3.14.7)

Exported 2026-10-04 from `main` at `4d0c532`. Nothing was running; nothing is unfinished.

## The export (`share_cache.py export downstairs`)

- `materials__dft.json`: **175 DFT labels**, all converged (refused labels are never cached).
  - **Copper, 89** (71 crystal + 18 MD at 870/1450/2040 K): LDA, h 0.19, xc_grid 2, fp32 GPU path.
    They have **no `pseudo_version`**: they predate that field. Copper's pseudopotential is bit for bit the
    same from v4 to v6 (D1, D2), so they are current. 66 were made before the fp32 eigensolver fix
    `1cdae28`, and the 5 cells it had refused were relabelled after it. A relabelled cell agrees with the
    stored label to 0.0002 meV/atom. Some were made before the complex64-gram change (`precision` says which).
  - **Iron, 86** (68 crystal + 18 MD at 1160/1940/2720 K): PBE, spin-polarised from 3 μB, h 0.20,
    xc_grid 2, hybrid magnetisation mixing, pseudo v6, fp32 GPU path. 3 crystal cells were refused at
    200 iterations and are absent.
- `fe_magnetism.json`: **34 points**. 21 are LDA fcc phase-scan points (NM/FM/AFM, V 60–84; plain Pulay
  mixing; 2026-09-26, so pseudopotential v5, which for Fe equals v6 on this machine (D2); the records carry
  no version). 13 are the 2026-10-04 hybrid-mixing rechecks (`*_hybrid.json`, LDA and PBE, v6).
  The LDA bcc points are Linux's and the PBE scan is the Mac's: not here.

## `extra/` (local-only caches as JSON, same format; `share_cache.py import` does not read `extra/`)

- `materials__dft_check.json`: 11 float64 NumPy recomputations of labels (cross-checks), keyed like the
  labels. 9 are copper's, one per tag (summarised in `results/cu_crosscheck.json`). **2 are iron's**
  (`bcc-strain`, `bcc-volume`: −0.013 and −0.33 meV/atom against the labels). These have no summary file anywhere.
- `materials__dft_superseded.json`: 1 iron `bcc-thermal` 4-atom label from 2026-09-29, converged by an
  early damped-magnetisation fallback. **Superseded**: it was moved aside when hybrid mixing became the
  procedure (2026-09-30), and its configuration was relabelled. Don't mix it into a training set.
- `materials__cu_experiments_c6d57b3637a6.json`: stage cache of copper's experiments (potential
  `cu_eam_final.npz`, sha256 c6d57b3637a6); the result is `results/cu_results.json`.
- `materials__fe_experiments_12f961f14481.json`: iron's experiments on the **current** potential
  (tag-stratified fit); the result is `results/fe_results.json`.
- `materials__fe_experiments_75f1fbd4b9e9.json`: iron's experiments on the **superseded** random-split
  potential (in git at `11d7a24`). It includes `latent_*.json` (liquid minus a melted "solid", wrong) and
  `latent_v2_*.json` (extrapolated 1050 K, wrong), both kept on purpose as a record.

Not exported: pseudopotential caches (every machine regenerates them), run logs, and a local copy of
the old iron potential (it is in git history).
