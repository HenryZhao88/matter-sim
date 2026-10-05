# Mac (M4, 16 GB): export of 2026-10-04

All 170 items are converged. Nothing was running on this machine when it was exported.

| File | Items | What |
|---|---|---|
| `materials__dft.json` | 89 | **Aluminium** DFT labels: the training set behind `results/al_eam_aluminium.npz` (bcc 6, fcc-8 16, fcc-strain 16, fcc-thermal 24, fcc-volume 9, md-600/1000/1400 6 each). From 2026-09-22/23, before labels recorded provenance, so they carry no `solver`, `h` or `functional` field. They were all NumPy float64, LDA, h = 0.3, xc_grid 1 (the aluminium defaults then and now), on an older pseudopotential cache version. Aluminium's pseudopotential is unchanged since (identical v4→v5, and D2 left every element outside Ti–Ni unchanged), so these labels are still current. |
| `materials__eos.json` | 20 | **Copper** E(V) points from `scripts/eos.py` (h 0.19, xc_grid 2): LDA 9 points at k 10 plus 2 at k 12 (a k-convergence check), PBE 9 at k 10. These give `results/cu_eos.json` and `cu_eos_pbe.json`. The LDA points predate v6 and the PBE points use v6; copper's LDA pseudopotential is identical across v4–v6. |
| `fe_grid.json` | 16 | **Iron** grid convergence (`scripts/fe_grid.py`, bcc, xc_grid 2), h = 0.16/0.20/0.24 at V = 68/76/84 bohr³, spin on/off, egg-box shift 0/0.5. 15 points are PBE (v6). The one without `_pbe`, `h0.24_x2_v76.0_s1_d0.0.json`, is LDA from the stray launch I killed; it is complete and converged, but it is a single point. |
| `fe_magnetism.json` | 45 | **Iron, PBE**, all five phases × 9 volumes (V = 60–92 bohr³, h 0.20, xc_grid 2, bcc k 8, fcc k 6, v6 pseudopotentials). Gives `results/fe_magnetism_pbe.json`. The LDA scan was run on Linux and downstairs, not here. |

Not exported, on purpose:
- `.cache/materials/dft_coarse_k/`: 98 superseded aluminium labels from the k-mesh that varied
  with cell size (the "labels must be mutually consistent" lesson). No code reads them.
- `.cache/materials/al_round*.pkl` and `al_eam_round*/try*.npz`: intermediate aluminium fitting
  rounds, superseded by `results/al_eam_aluminium.npz`. `al_crystal.json`, `al_results.json` and
  `al_eam_errors.json` are byte-identical to the copies already in `results/`.
- Pseudopotentials (`.cache/pseudo`, v6) and `pp_check` candidates: each machine regenerates them.

There is no `extra/` folder: every finished Mac result is already in `results/`. The harder iron
pseudopotential test (r_s 2.31, a₀ 2.934 Å) and the PBE water relaxation were one-off scratch runs.
Their numbers are only in AGENTS.md and the commit messages; the raw points were not kept.
