# Linux cloud container — export notes (2026-10-05, main at 4d0c532)

4 cores, 15 GB, no GPU, ephemeral (anything not in git is lost when it is reclaimed). Python 3.12.3,
NumPy 2.5.3, SciPy 1.18.1, Numba 0.67.0. Nothing is running here; nothing is half-finished.

## Exported by `share_cache.py`

- **fe_grid (30 points):** `scripts/fe_grid.py`, bcc iron, spin-polarised (start 3 μB) plus
  non-magnetic, LDA, 6³ k, h ∈ {0.30, 0.24, 0.20, 0.16} × xc_grid ∈ {1, 2} (0.20 and 0.16 with
  xc_grid 2 only). These chose iron's grid h = 0.20, xc_grid 2. All converged. Iron's pseudopotential
  is the D1 1.25× potential (cache v5; v6 left Fe unchanged).
- **fe_magnetism (14 points):** the bcc-nonmagnetic and bcc-ferromagnetic phases of
  `scripts/fe_magnetism.py`, 7 volumes each, h 0.20, xc_grid 2, 8³ k, MP 0.01 Ha, LDA, v5. All
  converged. These are the bcc points in `results/fe_magnetism.json`.
- **No DFT labels, no `eos.py` points, no melting runs:** none were made on this machine.

## `extra/` (scratch output from diagnostics, not in any cache)

- `cu_eggbox.jsonl`: copper half-step egg-box, h 0.30–0.19, plain xc grid, v4 copper (the same as
  v5/v6). **Superseded** by the Mac's `xc_grid = 2` fix. Kept as the independent measurement.
- `cu_softcore_eggbox.jsonl`: the same with prototype softer partial cores. A **withdrawn** option
  (DECISIONS.md: copper's grid decision).
- `fe_eggbox.jsonl`, `fe_conv.jsonl`: iron egg-box and convergence on the **old v4 iron potential**
  (s-local 1.5×, the one that collapses in the solid). **Superseded; don't use the energies.**
- `eos_checks.log`: non-magnetic E(V) for Ni, Co (fcc), Cr, V, Mn (bcc), h 0.24, xc_grid 2, 6³ k,
  **v5 potentials**. Ni changed in v6 (D2), so **Ni's curve is for the old potential and needs a
  rerun**. V/Cr/Mn/Co are unchanged in v6.
- `rc_scales.jsonl`: which radius stretch and local channel the generator chose for K–Kr, **v4
  (before the D1 cap)**. A historical record only.

Not exported: candidate pseudopotential pickles (`.cache/pp_check`, `pp_check.py candidates` remakes
them) and the pseudopotential cache itself (regenerated per machine).
