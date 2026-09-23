# Where the work stands

A snapshot for whoever (or whatever) picks this up next, on either machine.

## The state of the ladder

| Rung | State |
|---|---|
| Particles and forces | working: collisions, decays, confinement, parton showers, running α_s, proton collisions |
| Lattice QCD | working: confinement, deconfinement, and hadron masses (pion as a Goldstone boson) |
| Electrons and nuclei | working: H–Kr, molecules, truth mode (neural-network wavefunction) |
| Materials | **in progress**: the learned potential is being refitted on converged DFT labels |
| Everyday matter | waiting on the rung above: the code is written, the numbers are not in |

## The one unfinished thread: aluminium

The chain is DFT on small cells → a learned interatomic potential → molecular dynamics of a few
hundred atoms → a continuum 1 cm³ block. Everything is implemented. What is missing is a good
potential, for a reason worth remembering:

**The first training labels were not converged in k-points.** They used a mesh that varied with
cell size, which left errors of ~30 meV/atom that differed between configurations. No potential
can fit numbers that disagree with each other, and the symptom was a bulk modulus of 44 GPa
against DFT's 86, with the fit getting *worse* as more data arrived. Fitted to the fcc data
alone (internally consistent) the same model reaches ~2 meV/atom and 92 GPa, which is how the
data were convicted rather than the method.

The labels are now computed at a uniform 45 bohr k-spacing with 0.01 Ha smearing (converged to
<1 meV/atom), and the superseded ones are parked in `.cache/materials/dft_coarse_k/` — do not mix
them in; every new label records its own `kspacing` and `T_e`.

To finish:

1. Let `.cache/materials/dft/` reach 89 labels (a scratch script drives
   `engine.materials.dataset.label_all`; it skips whatever is already cached).
2. Fit: `fit()` then `refine()` from `engine/materials/eam.py`, weighted by `e_scale_eV=0.3`.
   Check against DFT: a₀ 3.955 Å, B 84.5 GPa, C11 124, C12 65, bcc−fcc 108 meV.
3. Measure: `engine.materials.experiments.run_all` writes `.cache/materials/al_results.json`
   (thermal expansion, heat capacity, melting point by solid–liquid coexistence, latent heat).
4. That file plus `al_crystal.json` is what the Materials workspace and
   `matter-sim validate --part materials` read.

## Watch out for

- **Memory, not cores, bounds the DFT pool.** Each worker holds the projectors for every
  k-point (~1.5 GB at this mesh). Eight workers once exhausted a 16 GB machine and drove it
  41 GB into swap. `label_all` caps at 5 workers and recycles children.
- **Orphaned workers.** Killing the parent leaves the pool alive. `pkill -f multiprocessing`.
- **Every long job caches per item** so a crash or a flat battery costs only the item in flight.
- **Element availability is earned.** A pseudopotential beyond argon is offered only after it
  passes a ghost-state check and a transferability check; the verdicts live in
  `.cache/pseudo/v4_Z*_checks.json` and `matter-sim pseudos` rebuilds them. Scandium currently
  fails (212 meV against a 150 meV limit) and is greyed out.

## Cross-platform (Mac + Windows/NVIDIA)

`mlx` is Apple-only and is still listed as a hard dependency, so `uv sync` fails on Windows.
Four places use it: `engine/core/backend.py` (already falls back to NumPy),
`engine/lattice/hadrons.py` (GPU Wilson–Dirac solver), `engine/materials/eam.py` (fitting) and
`engine/truth/vmc.py` (neural-network wavefunction). The plan is one backend covering MLX on
Apple, CUDA (PyTorch) on the NVIDIA machine, and NumPy everywhere as the fallback. The periodic
DFT in `engine/crystal/periodic.py` is plain NumPy on the CPU and is the real bottleneck — it is
the piece with the most to gain from a CUDA port.

## Conventions

- Run things with `env -u VIRTUAL_ENV uv run ...` (macOS) / `uv run ...`.
- Fast tests: `uv run pytest -m "not slow"`. The slow ones run real physics.
- Nothing about the answers is hard-coded: prefilters must be true symmetries, and every
  comparison with experiment lives in `validation/run.py`, which prints simulated vs measured.
