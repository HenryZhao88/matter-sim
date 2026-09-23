# Where the work stands

A snapshot for whoever (or whatever) picks this up next, on either machine.

## The state of the ladder

| Rung | State |
|---|---|
| Particles and forces | working: collisions, decays, confinement, parton showers, running α_s, proton collisions |
| Lattice QCD | working: confinement, deconfinement, and hadron masses (pion as a Goldstone boson) |
| Electrons and nuclei | working: H–Kr, molecules, truth mode (neural-network wavefunction) |
| Materials | working: melting, expansion, heat capacity from learned forces |
| Everyday matter | working: the 1 cm³ block, built from the per-atom properties above |

## Aluminium: done, and what it cost

The chain runs end to end: DFT on small cells → a learned interatomic potential → molecular
dynamics of a few hundred atoms → a continuum 1 cm³ block. Results are in `results/`
(`al_results.json`, `al_crystal.json`, `al_eam_errors.json`, `al_eam_aluminium.npz`) and in the
README table. Melting 852 K against 933.5 measured; latent heat 70 meV/atom against 111, which
fails its validation check and is reported failing.

Four faults had to be fixed first, and every one of them produced plausible-looking numbers
before it was found:

1. **Training labels not converged in k-points**, with a mesh that varied by cell size: ~30
   meV/atom of inconsistency between configurations. No potential can fit numbers that disagree
   with each other; the bulk modulus came out 44 GPa against DFT's 86. Labels now use a uniform
   45 bohr spacing with 0.01 Ha smearing and record their own settings; superseded ones are in
   `.cache/materials/dft_coarse_k/` and must never be mixed in.
2. **A fitted density that crossed zero.** Where neighbours summed to ρ≈0 the embedding
   energy's √ρ slope reached −114,000 eV and threw atoms across the box at ordinary geometries.
   Density coefficients are now non-negative by construction, and the slope is floored.
3. **No short-range repulsion**: the basis vanishes below 3.6 bohr, which training never
   sampled. Screened nuclear (ZBL) repulsion is spliced in below 3.4 bohr.
4. **A barostat that boiled the metal into vacuum** while reporting a melting point of 961 K
   that looked convincing. Limited to 0.02% volume per step; runs abort if the cell runs away;
   coexistence is done at fixed volume.

The lesson worth keeping: in this rung a wrong answer arrives looking like a right one. Check
the state, not just the number — volume per atom, minimum separation, whether the thing is
still a condensed metal.

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

- **Truth-mode error bars cover sampling, not training.** `evaluate()` takes its ± from the
  spread of the 512 walkers' time-averaged energies (independent Markov chains), which is honest
  for the energy of *that* trained wavefunction. A different seed trains a different
  wavefunction; measured on He and H₂ (five seeds, CUDA), that adds ~2 mHa of seed-to-seed
  spread no single run's ± can include. At the default `evaluate(12, 5)` the ± is ~4.5 mHa and
  sampling dominates; on long evaluations the ± falls to 1–2 mHa and the training spread does
  not, so there the ± understates how far another run could land. For a claim about the method
  rather than one wavefunction, run several seeds. (The previous estimate, from 12 correlated
  block means, was unstable: 1.5 mHa on a run whose walker spread says 4.0.)

## Cross-platform (Mac + Windows/NVIDIA)

`engine/core/accel.py` answers what a machine can compute on: MLX on Apple silicon, CUDA (or
MPS) through PyTorch, NumPy always as the reference. `mlx` installs only on Apple silicon and
`torch` is the optional `gpu` extra, so the project installs everywhere. Truth mode and the
lattice Wilson–Dirac solver each have MLX and PyTorch implementations: the Dirac operators and
propagators agree with the float64 NumPy reference to 1e-4 (a test), and the two VMC ports give
energies that agree within the seed-to-seed spread below. On the RTX 4050 the VMC runs no faster
than the CPU at 512 walkers (kernel-launch bound, ~23 s for He); at 8192 walkers CUDA is 7× the
CPU per walker, so on a CUDA box Truth mode should scale walkers, not iterations. Still to do: the periodic DFT in `engine/crystal/periodic.py` is plain NumPy on the
CPU and is the real bottleneck — labelling 89 configurations took about five hours. It is the
piece with the most to gain from a CUDA port, and the open question is whether single precision
holds energies to ~1 meV/atom, since a consumer NVIDIA card runs double precision at 1/64 speed
and Metal has no double precision at all.

## Conventions

- Run things with `env -u VIRTUAL_ENV uv run ...` (macOS) / `uv run ...`.
- Fast tests: `uv run pytest -m "not slow"`. The slow ones run real physics.
- Nothing about the answers is hard-coded: prefilters must be true symmetries, and every
  comparison with experiment lives in `validation/run.py`, which prints simulated vs measured.
