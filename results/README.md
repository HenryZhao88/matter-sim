# Derived results, shared between machines

Caches stay local (`.cache/`, gitignored, megabytes of intermediate state). The small,
final numbers live here so either machine can read them without copying anything:

| File | Written by | Read by |
|---|---|---|
| `al_crystal.json` | `engine/crystal/eos.py` (periodic DFT: lattice constant, bulk modulus, elastic constants, cohesive energy) | the Materials workspace, `validate --part materials` |
| `al_results.json` | `engine/materials/experiments.run_all` (expansion, heat capacity, melting point, latent heat) | same |
| `al_eam_errors.json` | the fitting pipeline (how well the learned potential reproduces DFT) | the chain shown in the viewer |
| `al_eam_*.npz` | `engine/materials/eam.EAM.save` (the fitted potential itself) | molecular dynamics on either machine |
| `hadron_spectrum.json` | `engine/lattice/hadrons.spectrum` | the Lattice QCD tab, validation |

Rules:

- Say in the commit message which machine produced a result and how long it took. A number
  from a 6-configuration run and one from a 16-configuration run are not the same number.
- Anything above ~5 MB does not belong here; keep it in `.cache/` and say so.
- These are outputs, never inputs to the physics. Nothing in `engine/` may read a file here
  to decide what the answer should be.
