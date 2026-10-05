# Windows laptop export (HP Victus 15, Ryzen 5 8645HS, RTX 4050 6 GB, 7.8 GB RAM)

Exported 2026-10-05 from main at `99210be`.

## What's here

- **`materials__dft.json`: 29 DFT labels.**
  - **28 aluminium labels were not computed here.** They were copied over from the Mac with
    `.cache/` on 2026-09-22, so they duplicate labels in the Mac's own export. All 28 are NumPy
    float64, k-spacing 45 and T_e 0.01 (the converged set). 24 predate provenance and have no
    `kspacing`, `T_e` or `solver` fields; 4 carry `kspacing`/`T_e`. Tags: 9 fcc-volume, 19
    fcc-thermal. Superseded coarse-k labels (`dft_coarse_k/`) are not exported.
  - **1 copper label was computed here**, on 2026-09-25: the compressed fcc-volume cell (a₀ × 0.90 at
    a₀ = 6.704 bohr, 8³ k), key `0f5146c079ed2647`.
    - Path and grid: fp32 GPU path (`solver: torch`), h 0.19, xc_grid 2.
    - Functional: no `functional` field, so LDA.
    - Older code: it predates two later fp32 fixes, the eigensolver fault fixed in `1cdae28` and the
      LOBPCG speed-ups (a)/(b).
    - Older pseudopotentials: made on v4. Copper is bit-for-bit identical across v4, v5 and v6 (D1, D2).
    - It converged in 15 SCF iterations (732 s). The downstairs PC has the full copper set, made
      with the fixed code, so prefer theirs where the two overlap.
- **`materials__melt_5x5x12.json`, `materials__melt_20x20x60.json`: 6 + 6 coexistence points.** These
  are aluminium's melting bisection at 1 200 and 96 000 atoms (GPU MD, float64), summarised in
  `results/al_melting_1200.json` (867.2 K) and `results/al_melting_96000.json` (898.4 K). They
  used the aluminium potential `results/al_eam_aluminium.npz` and the a(T) from
  `results/al_results.json`.

## extra/

- **`cu_experiments_legacy/`: copper's materials experiments, unfinished.**
  - What it covers: the stages done before Claude Code's low-memory guard stopped the run on
    2026-09-29. Temperature scale T_order = 1900 K; thermal curve at 133–950 K (a = 3.5546 →
    3.5898 Å); one coexistence point (1425 K: melts) on `results/cu_eam_final.npz` as it was then.
  - Old directory name: `scripts/element_experiments.py` now keys its cache by the potential's hash
    (`cu_experiments_<hash>/`), so it would not resume from these. Check the hash before reusing
    them: if `cu_eam_final.npz` has been refitted since, they are stale.
  - Still owed: 5 coexistence points and the latent heat.
- **`diagnostics/`: raw logs behind numbers quoted in commit messages.**
  - fp32 versus float64 on full-mesh labels (`dft_full.log`, `dft_numpy_full.log`).
  - The eigensolver-floor scan (`floor_scan.log`) that set `RESIDUAL_FLOOR`.
  - The first 8-atom fp32 check (`fcc8_fp32.log`).
  - The copper label timing (`cu_label_time.log`).

## Not here

- No E(V), iron grid or iron magnetism points: this machine computed none.
- Nothing is running on this machine.
- 28 macOS AppleDouble files (`._*.pkl`, 4 096 bytes each, from the Mac copy) sat in
  `.cache/materials/dft/` and broke the export with `UnpicklingError`. They are metadata, not labels.
  I moved them to `.cache/appledouble_quarantine/`, and more of the same sit at the top of `.cache/`
  (`._collider`, `._materials`, ...). Any machine whose `.cache/` came from the Mac by copy may
  have them.
