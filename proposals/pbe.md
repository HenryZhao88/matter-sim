# Proposal: a gradient-corrected functional (PBE), alongside LDA

**Status:** open. See `DECISIONS.md` (D3) to comment or decide.
**Author:** Claude, Mac, 2026-09-26.
**Touches:** `engine/electrons/xc.py` (new functional), and every place the functional is called:
`atoms/radial.py` (the all-electron and pseudo-atom), `atoms/pseudo.py` (generation, ghost check),
`crystal/periodic.py` and `periodic_torch.py` (`_xc`, `_xc_spin`), `electrons/scf.py` (molecules).
Pseudopotential cache names gain the functional.

## Why

Plain LDA gets iron's structure wrong. Our own scan (`results/fe_magnetism.json`, h 0.20,
xc_grid 2) puts non-magnetic fcc about 40–45 meV/atom below ferromagnetic bcc. It is LDA's known
failure for iron, and the scan is reported disagreeing, as it should be. Everything downstream of
iron (its crystal, elasticity and magnetism, then steel) inherits that. The next rung of the
functional ladder is the generalised-gradient approximation. Perdew–Burke–Ernzerhof (PBE, PRL 77,
3865, 1996) is the one that fits this project's rule. Its constants (κ = 0.804, μ = 0.21951,
β = 0.066725) come from exact constraints on the exchange–correlation hole, the uniform gas and its
gradient expansion; none is fitted to molecules or solids. That is the same standing as the
Perdew–Wang correlation LDA already uses.

**What this proposal does not claim:** that PBE will make bcc iron come out lowest. That has to be
measured with `scripts/fe_magnetism.py` and left disagreeing if it does not.

## The change

1. **`xc.pbe_xc(rho_up, rho_dn, grad terms)`** in `electrons/xc.py`, spin-polarised, reusing
   `pw92_correlation`. It returns the energy density and the partial derivatives needed for the
   potential (∂e/∂ρσ and ∂e/∂|∇ρ|² terms). Tested against finite differences of its own energy,
   against the uniform-gas limit (it must equal LDA when ∇ρ = 0), and against published reference
   values for the hydrogen and helium atoms (a check, not an input).
2. **The potential needs a divergence:** v = ∂e/∂ρ − ∇·(∂e/∂∇ρ). Each solver supplies its own
   gradient: finite differences on the radial log grid; FFT on the periodic and molecular grids,
   done on the fine XC grid where `xc_grid > 1`.
3. **A functional switch, LDA the default.** `PeriodicDFT(functional="lda"|"pbe")`, the same on
   `RadialAtom`, `pseudo.generate` and `SCFSolver`. With `"lda"` everything is bit for bit what it
   is today. That is required, because copper's labels are being produced with LDA right now.
4. **Pseudopotentials per functional:** a PBE crystal must use pseudopotentials generated with
   PBE (mixing them gives errors of the size we are trying to fix). Cache names become
   `v6_Z26.pkl` (LDA, unchanged) and `v6_pbe_Z26.pkl`. Each PBE pseudopotential passes the same
   ghost and transferability checks and has its own `_checks.json`. D1's cap and D2's reference
   atom apply unchanged.
5. **Per element, like the grid:** the functional becomes part of an element's settings
   (`dataset.GRID` → element settings), recorded on every label and in its cache name when it is
   not LDA. Aluminium and copper stay LDA unless someone relabels them.

## What it costs

- **Work:** xc.py (the functional and its tests), four call sites, the radial divergence, and the
  torch path. Estimated at a few sessions; the radial atom goes first, because pseudopotential
  generation depends on it.
- **Grid:** GGA potentials are sharper than LDA's where the density varies fast. Copper's egg-box
  came from its partial core in LDA, and gradients of that core will be sharper still. Each element
  needs its egg-box and strain-energy convergence redone under PBE (`scripts/fe_grid.py` already
  does this for iron). Budget for a finer h or `xc_grid = 3`.
- **Near the nucleus and in the tails:** the reduced gradient s = |∇ρ|/ρ^{4/3} blows up where ρ → 0.
  PBE's enhancement factor stays bounded (≤ 1.804), but the potential's divergence term needs care
  on the log grid and at the density floor. This is a known source of noisy GGA potentials.
- **Time per SCF step:** extra FFTs for ∇ρ and the divergence (about 3 + 3 per step on the XC
  grid), small next to the eigensolver.
- **Results:** none change unless someone asks for PBE. The atoms rung's ionisation energies and
  molecules stay LDA; a PBE column can be added to `validation/run.py` later.

## Alternatives

- **Stay with LDA** and report iron's structure disagreeing. That's honest, but it caps the
  materials rung at elements LDA happens to get right.
- **A meta-GGA (r²SCAN).** It is also constraint-based and usually better than PBE for magnetic
  3d metals' volumes, but it overestimates iron's moment. It also needs the kinetic-energy density
  τ in every solver, and its potential is not a local multiplicative one in the usual (generalised
  Kohn–Sham) form, which is a much larger change. PBE first; r²SCAN can follow on the same switch.
- **DFT+U or hand corrections:** they add a fitted parameter, which the project rules out.

## Reproduce / success tests

- Unit: `pbe_xc` equals `lda_xc` at ∇ρ = 0, and its potential matches finite differences of
  E_xc on the radial, periodic and molecular grids.
- Atoms: the PBE helium and neon total energies against published PBE all-electron values, to
  within the radial grid's error.
- Iron: `scripts/fe_magnetism.py` with `functional="pbe"`, on a PBE iron pseudopotential whose own
  E(V) check (`pp_check.py eos`) has a minimum. Report the phase order whatever it is.
