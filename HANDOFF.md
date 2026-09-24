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
- **A short truth-mode run can land below the exact energy, and that is noise, not bias.** It
  looks alarming — a variational energy cannot be below the true ground state — but the estimate
  is a mean over a finite sample, and the local-energy distribution has a long low tail (the
  network has no electron–nucleus cusp), so short runs scatter downward more often than a
  Gaussian would. The check is a long re-evaluation of the same wavefunction, not an argument.
  He seed 1 on MLX: `evaluate(12, 5)` gives −2.9091 ± 0.0053 (1.0σ below exact) and
  `evaluate(200, 20)` on that same wavefunction gives −2.9018 ± 0.0010, 1.9σ above. On CUDA, ten
  wavefunctions re-evaluated 400× longer averaged +1.4 mHa (He) and +1.3 mHa (H₂), none below by
  more than 0.9σ. Training-time clipping cannot cause it either: `evaluate` never clips, and
  clipping in `train` only changes *which* wavefunction you end up with — every wavefunction's
  true energy is at or above exact. Quote a truth-mode number from a long evaluation, or from
  several seeds; a single short run below exact means the run was short.

## Cross-platform (Mac + Windows/NVIDIA)

`engine/core/accel.py` answers what a machine can compute on: MLX on Apple silicon, CUDA (or
MPS) through PyTorch, NumPy always as the reference. `mlx` installs only on Apple silicon and
`torch` is the optional `gpu` extra, so the project installs everywhere. Truth mode and the
lattice Wilson–Dirac solver each have MLX and PyTorch implementations: the Dirac operators and
propagators agree with the float64 NumPy reference to 1e-4 (a test), and the two VMC ports give
energies that agree within the seed-to-seed spread below. On the RTX 4050 the VMC runs no faster
than the CPU at 512 walkers (kernel-launch bound, ~23 s for He); at 8192 walkers CUDA is 7× the
CPU per walker, so on a CUDA box Truth mode should scale walkers, not iterations. The periodic DFT
now has a single-precision GPU path, `engine/crystal/periodic_torch.py` (complex64 eigensolver
on the device, every energy and force sum in float64). Against the float64 NumPy path (which
reproduces the Mac's cached labels to 1e-11 on Windows): a production 4-atom label (fcc Al, 6×6×6 k,
h = 0.3, T_e = 0.01) to 0.0007 meV/atom and 1.4e-6 Ha/bohr, an 8-atom one (3×6×6 k) to 0.0021
meV/atom and 4.8e-6 Ha/bohr. A 4-atom label with forces takes 85 s on the RTX 4050 against 1121 s
for NumPy on that laptop's CPU (13×); an 8-atom one 148 s, peaking at 0.6 GB of VRAM and 3.6 GB of
committed host memory (1.2 GB touched), so on a 7.8 GB machine run one at a time. **CUDA only:**
on Apple's MPS the float64 work falls back to the CPU and the path is 8.7× *slower* than NumPy,
so the Mac should keep labelling with NumPy.

Two things to know about it:
- **Its eigensolver needs soft locking.** Without it, bands that have converged to complex64's
  rounding floor keep feeding that noise back as search directions, and LOBPCG returns
  eigenvalues far below the spectrum (−91 Ha on fcc Al); the SCF then never converges yet still
  returns an energy, several meV off. Fixed, with tests that fail without the fix.
- **Forces are what set its accuracy, not energies.** Energies are second order in the
  eigenvectors' error and forces first order, so a solver that stops a little early shows up only
  in the forces: with the locking floor at 30 ε₃₂‖H‖, energies agreed to 0.0065 meV/atom while
  forces were 1.4e-4 Ha/bohr off (2.9e-4 on the 8-atom cell), in a structured (100) pattern, and
  unmoved by tighter SCF tolerances, float64 occupations or a float64 force routine. The floor is
  now 1 ε₃₂‖H‖ (+40% time). Judge any change to this path by its forces.

Labels are refused, not cached, when the SCF does not converge (`dataset.NotConverged`).

## Conventions

- Run things with `env -u VIRTUAL_ENV uv run ...` (macOS) / `uv run ...`.
- Fast tests: `uv run pytest -m "not slow"`. The slow ones run real physics.
- Nothing about the answers is hard-coded: prefilters must be true symmetries, and every
  comparison with experiment lives in `validation/run.py`, which prints simulated vs measured.
