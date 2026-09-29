# Derived results, shared between machines

Caches stay local (`.cache/`, gitignored, megabytes of intermediate state). The small,
final numbers live here so every machine can read them without copying anything:

| File | Written by | Read by |
|---|---|---|
| `al_crystal.json` | `engine/crystal/eos.py` (periodic DFT: lattice constant, bulk modulus, elastic constants, cohesive energy) | the Materials workspace, `validate --part materials` |
| `al_results.json` | `engine/materials/experiments.run_all` (expansion, heat capacity, melting point, latent heat) | same |
| `al_eam_errors.json` | the fitting pipeline (how well the learned potential reproduces DFT) | the chain shown in the viewer |
| `al_eam_*.npz` | `engine/materials/eam.EAM.save` (the fitted potential itself) | molecular dynamics on either machine |
| `hadron_spectrum.json` | `engine/lattice/hadrons.spectrum` | the Lattice QCD tab, validation |
| `al_melting_1200.json`, `al_melting_96000.json` | `scripts/melting_large.py` (melting by coexistence, 1 200 and 96 000 atoms, GPU MD) | validation |
| `al_fcc8_reference.json` | an 8-atom float64 aluminium label, shipped as positions and numbers | checks of the fp32 DFT path |
| `cu_eos.json`, `cu_eos_pbe.json` | `scripts/eos.py 29` (copper's own lattice constant and bulk modulus, LDA / PBE) | training-set construction, validation |
| `cu_crosscheck.json` | `scripts/cross_check.py 29` (fp32 labels recomputed in float64) | anyone judging copper's labels |
| `cu_md_snapshots.json` | `scripts/label_md.py 29 ...` (hot 8-atom configurations the seed potential visits; kept because the dynamics cannot be regenerated bit for bit elsewhere) | copper labelling |
| `cu_eam_seed.*`, `cu_eam_final.*` | `scripts/seed_fit.py 29 6.704 [w] seed|final` (copper's learned potential and its errors: seed from crystal labels, final with MD labels too) | molecular dynamics |
| `fe_magnetism.json`, `fe_magnetism_pbe.json` | `scripts/fe_magnetism.py` (iron's phases: bcc/fcc, non-magnetic, ferro- and antiferromagnetic; LDA / PBE) | validation |

Rules:

- Say in the commit message which machine produced a result and how long it took. A number
  from a 6-configuration run and one from a 16-configuration run are not the same number.
- Anything above ~5 MB does not belong here; keep it in `.cache/` and say so.
- These are outputs, never inputs to the physics. Nothing in `engine/` may read a file here
  to decide what the answer should be.
