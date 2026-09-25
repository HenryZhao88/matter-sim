# matter-sim — working agreement for agents

Several agents work on this repository, one at a time, on different machines. This file is how
they talk to each other: read it before you start, update it before you stop. It replaces
copy-pasting messages through the human.

**Do not trust this file on its own.** It is written by agents and can be stale or wrong. The
repository is the truth: `git log --oneline`, the diffs, the tests, `validation/run.py`. If
something here disagrees with the code, the code wins — and fix this file.

## What the project is

A first-principles simulator of matter, from particle collisions up to a block of metal you
could hold. The whole point is that **only the laws are written into the code**: the Standard
Model Lagrangian as structure (gauge group, representations, the Higgs potential, Yukawa
couplings), its low-energy form for atoms (Schrödinger, Coulomb, Pauli), and a short list of
measured constants (α, G_F, m_Z, α_s, fermion masses, CKM, and each nucleus's charge and mass).

Everything else must **emerge**: particles and their charges, the W and Z masses, decays and
lifetimes, confinement, shell structure, Hund's rule, chemical bonds, crystal structures,
melting. Nothing about an outcome may be hard-coded, fitted to experiment, or nudged.

Comparisons with experiment live in `validation/run.py`, which prints simulated against
measured side by side. Experimental numbers are used to **check**, never as input. When a
result disagrees with nature, say so and leave it disagreeing.

## How we work

- **Pull before you start** (`git pull --rebase`), **push when you stop**. Never force-push.
- **Small commits with real messages.** Say what changed and why; if a number moved, say by how
  much. End commit messages with the attribution lines the session tells you to use.
- **Don't loosen a bound to make a test pass.** If a tolerance fails, report the measured value.
  A bound that was chosen for a reason is evidence; widening it destroys the evidence.
- **Verify before claiming.** Run the thing. Quote the output. "Should work" is not a result.
- **Say when something is unverified**, skipped, or still running. Half the bugs below were
  found because someone reported an awkward number instead of a tidy one.
- **Stay out of files another agent is actively editing**; say in the log what you are touching.
- **Long jobs must cache per item** and be resumable: machines here run out of memory and
  battery. Bound worker pools by RAM, not cores. Killing a pool's parent leaves orphans
  (`pkill -f multiprocessing`).

## Machines

| | Mac (M4, 16 GB) | Windows (Ryzen 5, 7.8 GB, RTX 4050 6 GB) |
|---|---|---|
| Accelerator | MLX (Metal) | CUDA through PyTorch |
| Best at | float64 reference work, MLX paths, the viewer | fp32 DFT labelling: **~12× NumPy** (Al label 95 s vs 1121 s) |
| Avoid | fp32 DFT (MPS is 8.7× *slower* than NumPy here) | anything needing >4 GB RAM; it gets killed |

`engine/core/accel.py` decides what a machine uses: MLX, else CUDA, else MPS, else NumPy.
`MATTER_SIM_ACCEL=torch` forces the torch path (that is how the Mac tests MPS).

## Current state

| Rung | State |
|---|---|
| Particles and forces | working: collisions, decays, confinement, parton showers, running α_s, pp at 13.6 TeV |
| Lattice QCD | working: confinement, deconfinement, hadron masses (pion as Goldstone boson, κ_c = 0.1695 vs 0.1694 published) |
| Electrons and nuclei | working: H–Kr (Sc fails its own check and is greyed out), molecules, truth mode (VMC, MLX and PyTorch) |
| Materials | working for **aluminium**: melting 852 K (measured 933.5), expansion, heat capacity. GPU molecular dynamics (`md_torch.py`) reaches 10⁶ atoms, tested equal to `md.py`, not yet used by the experiments. Spin-polarised periodic DFT (NumPy): iron comes out magnetic, 2.16 μB/atom |
| Everyday matter | working: the 1 cm³ block, 6.3 × 10²² atoms, 2.83 g, 2.29 kJ to melt |

Results both machines can read are in `results/`. `HANDOFF.md` carries the detail and the
hard-won lessons; read it too.

## In flight

- **Copper, blocked on grid convergence (Mac, running now).** Aluminium's h = 0.3 bohr does not
  resolve copper's 3d states, and the failure is severe. Sliding the whole crystal by half a
  grid step — which cannot change any physical energy — moves the computed energy by
  **−335 meV/atom at h = 0.30** and **−15.7 meV/atom at h = 0.26**, with spurious forces of
  2.7e-2 and 4.4e-3 Ha/bohr on a perfect crystal where every force is exactly zero. For scale,
  the whole aluminium fit achieved 6.5 meV/atom. h = 0.22, 0.19, 0.16 are running
  (`scratchpad/eggbox.py`, ~2 min per point; that script is local to the Mac and not in
  the repo — commit it with the results so the other machine can rerun it). **Do not label copper until this lands**, and
  re-time one copper label afterwards: copper needs ~3× aluminium's grid points and has 11
  valence electrons against 3, so aluminium's 8-minute label is not the right estimate.
- **Copper's grid error is its partial core, and a finer grid does not cure it (Linux cloud,
  2026-09-25, `scripts/eggbox.py`, committed).** Independent egg-box: fcc Cu, a = 6.83, 4³ k, the
  crystal slid half a grid step along x. Forces stay zero (a half step keeps atoms on grid mirror
  planes, so this test cannot reproduce the Mac's forces; its method must differ):

  | h (bohr) | 0.30 | 0.26 | 0.22 | 0.19 |
  |---|---|---|---|---|
  | current partial core, meV/atom | +31.7 | +7.2 | +5.3 | +8.2 |
  | partial core off (diagnostic only) | | −0.21 | | |
  | softer partial core (factor 1), meV/atom | −2.7 | +0.54 | +0.15 | |

  With the core on, energy terms swing by ±500 meV/atom and nearly cancel; a full-step shift and a
  new random start both give 0.00. Cause: `pseudo._partial_core` keeps the true core outside the
  radius where it exceeds **2×** the valence density, which for copper keeps 11 core electrons with
  4% of their Fourier weight beyond h = 0.30's cutoff (iron: 9 electrons, 100× smoother, egg-box
  0.09 meV at h = 0.30). Using **1×** keeps 6.8 electrons, is ghost-free, and the engine's own
  transferability check improves slightly (51.5 meV against 52.7; limit 150); 0.5× gives 45.6.
  **Proposed:** soften the partial core (a construction choice, not a fit) and pick copper's h
  from the egg-box with it; h ≈ 0.26 looks sufficient and fits both machines' memory. Not done
  here: it changes every Z ≥ 19 pseudopotential (bump `CACHE_VERSION`, rerun the atoms
  validation) and the pseudopotential is the Mac's area. Decide there.
- **Memory decides which h is usable (Windows, 2026-09-24).** Both DFT paths keep every
  k-point's wavefunctions resident (fp32: on the GPU in complex64; NumPy: in RAM in complex128,
  plus every k-point's projectors). For copper's 8-atom cells (3×7×7 k → 98 after time reversal,
  50 bands):

  | h (bohr) | grid | fp32 wavefunctions (6 GB GPU) | NumPy wavefunctions + projectors (RAM) |
  |---|---|---|---|
  | 0.22 | 64×32×32 | 2.4 GB | ~4.8 + 3.1 GB |
  | 0.19 | 72×36×36 | 3.4 GB | ~6.8 + 4.5 GB (tight on 16 GB) |
  | 0.16 | 90×48×48 | **7.6 GB: does not fit** | ~15 + 10 GB: **does not fit on the Mac either** |

  4-atom cells need about half. So h = 0.16 needs the wavefunctions streamed per k-point (or
  stored as G-sphere coefficients only) before either machine can label 8-atom copper with it,
  and h = 0.19 is the finest spacing the current code can do on both.
- **The fp32 path holds at copper's finer grids (Windows, measured).** Displaced 4-atom Cu, 2×2×2 k,
  fp32 against NumPy float64: at h = 0.19, −0.147 meV/atom and 1.1e-5 Ha/bohr (fp32 27 s, NumPy
  317 s). At h = 0.16 fp32 converged in 44 s; the NumPy comparison there was stopped by the
  low-memory guard and is still owed. The same cell's largest force is 0.109 Ha/bohr at h = 0.30
  and 0.0157 at h = 0.19 — independent support for the egg-box numbers above.
- **Iron: spin-polarised periodic DFT exists now (NumPy path only; Linux cloud, 2026-09-25).**
  `PeriodicDFT(spin=True, moments=...)`: LSDA, one Fermi level, so the magnetisation is free; the
  starting moment is only a push. Unpolarised results are unchanged bit for bit. Aluminium pushed
  to 1 μB/atom relaxes to 3e-5; bcc iron (a = 5.30 bohr) settles at **2.16 μB/atom** (measured
  2.22), ferromagnetic **397 meV/atom** below non-magnetic. Converged: h = 0.24 agrees with 0.20 to
  0.01 meV and 1e-4 μB (h = 0.30 is fine for the moment but 39 meV/atom off in absolute energy);
  k = 8 is ~3 meV and 0.02 μB from 12. `PeriodicDFTTorch` refuses `spin=True` (not ported).
  `scripts/fe_magnetism.py` (bcc/fcc × non-magnetic/FM/AFM, 7 volumes, h = 0.24) was running
  when this was written; the validation rows read its `results/fe_magnetism.json`. Expect plain
  LDA to get iron's structure wrong (it is known to favour close-packed non-magnetic iron): if it
  does, record it and leave it.

## What would help most

1. **Finish copper**: pick h from the egg-box numbers, run the equation of state on the Mac in
   float64 for copper's own lattice constant (it must come from our DFT, not from experiment),
   then label on the Windows machine with `solver="torch"`, with ~5% NumPy cross-checks.
2. **Iron, nickel, cobalt** on the new spin-polarised DFT: finish `fe_magnetism.py` if the
   results file is missing or partial (it resumes from `.cache/fe_magnetism`; fcc AFM is the slow phase);
   port spin to `periodic_torch.py` before labelling a magnetic metal on the GPU.
3. **Use the GPU molecular dynamics.** `engine/materials/md_torch.py` (`EAMForceFieldTorch`,
   `MDTorch`) matches `md.py` to 1e-10 and reproduces its seeded trajectories; on the RTX 4050 a
   full step is 84 ms at 256 000 atoms and 321 ms at 10⁶ (float64, 2.9 GB VRAM). Nothing uses it
   yet: `experiments.py` still runs a few hundred atoms. Next: melting by coexistence at 10⁵
   atoms (smaller finite-size error than today's 1 500), then grains, vacancies and a crack.
   Possible 2–4× more from float32 pair arithmetic, if it passes the same test.
4. **One-loop amplitudes** in the particle rung; the running coupling is in, the loops are not.

## Lessons that cost real time

Every one of these produced a *plausible* number before it was caught. Check `git log` for the
commits; the messages carry the measurements.

- **Training labels must be mutually consistent, not just accurate.** DFT labels with a k-mesh
  that varied by cell size left ~30 meV/atom of inconsistency; the fitted bulk modulus came out
  44 GPa against DFT's 86, and the fit got *worse* as more data arrived.
- **A learned potential extrapolates into nonsense.** The fitted electron density went negative,
  so the embedding energy's √ρ slope hit −114,000 eV and threw atoms across the box — at
  perfectly ordinary geometries. Densities are non-negative by construction now.
- **Physics the data never sampled is still physics.** No training configuration had atoms
  closer than 3.6 bohr, so the potential had no repulsion there and atoms passed through each
  other. Screened nuclear (ZBL) repulsion is spliced in below 3.4 bohr.
- **A thermostat or barostat will happily destroy the system.** A barostat allowed to change the
  cell 0.5% per step boiled the metal into vacuum, and the run reported a melting point of
  961 K against the measured 933.5. It was the sign of numerical noise.
- **fp32 eigensolvers break quietly.** LOBPCG fed converged bands' rounding noise back as search
  directions; float64 found the lowest band at −0.06 Ha, fp32 returned −91 Ha, and the SCF
  merely "did not converge" before returning a believable energy 4 meV/atom off.
- **A cache key is not a portable name.** The same configuration hashed differently on the two
  machines (unpinned pickle protocol), and unwrapped positions meant a lattice-vector shift
  changed the key.
- **A variational energy below the exact answer is usually short-run noise**, not a bug: check
  by re-evaluating the *same* wavefunction far longer before theorising. (One agent theorised;
  the other's test was right.)

## Keeping this file honest

When you finish a stint, update **Current state**, **In flight**, and **What would help most**,
and add a dated line to the log. Keep it short: this file is a handover, not a diary. Delete
anything that is no longer true rather than appending a correction.

## Log

- **2026-09-25, Linux cloud container (Claude; 4 cores, 15 GB, no GPU, ephemeral).** Checked
  this file against the code (it held; HANDOFF's copper note was stale, fixed). Built
  spin-polarised periodic DFT and measured iron's magnetism. Found copper's grid error is its
  partial core (numbers in In flight) and proposed a softer core; did not change `pseudo.py`.
  Lesson: two NumPy jobs on 4 cores each start 4 BLAS threads and run **~10× slower** together
  (a 90 s copper pair took 900 s); run one heavy job at a time or set `OMP_NUM_THREADS`.
- **2026-09-24, Windows (Claude).** Did not label copper. Measured copper at finer grids on the fp32
  path (holds at h = 0.19; h = 0.16 NumPy check owed) and which spacings fit in memory (h = 0.16
  fits on neither machine for 8-atom cells). Built GPU molecular dynamics (`md_torch.py`: forces
  and integrator, tested equal to `md.py`; 10⁶ atoms at 321 ms/step). Also this stint: wrap-
  invariant cache keys with free migration, the per-tag cross-check machinery
  (`scripts/cross_check.py`), `md_snapshots(Z=, mass_amu=)`, and the copper fixes to the fp32
  eigensolver (floor from the nonlocal term's real size, complex128 overlaps). This laptop's
  low-memory guard now stops even ~2 GB jobs when idle memory sits near 2 GB: run one job at a
  time, and expect to be asked before a rerun.
- **2026-09-24, Mac (Claude).** Wrote this file. Copper grid convergence running: aluminium's
  h = 0.3 is badly unconverged for copper (numbers above). Verified the Windows session's
  wrap-invariant cache keys against the real cache (71/71 configurations found), the full fast
  suite (108 passed, including the three files that machine cannot reach), and its fp32 DFT on
  MPS including slow tests (12 passed, tightened bounds hold). Decisions recorded: copper before
  iron; fp32 acceptable for copper labels with provenance and cross-checks; fix the cache key
  with a migration.
- **2026-09-23/24, Windows (Claude).** Truth mode and the Wilson–Dirac solver ported to PyTorch;
  fp32 periodic DFT (17× NumPy on CUDA) with the eigensolver fix, fp32-aware SCF tolerance, and
  a labeller that refuses non-converged results; wrap-invariant cache keys with migration.
- **2026-09-22/23, Mac (Claude).** Aluminium end to end: 89 converged DFT labels, learned
  potential, melting/expansion/heat capacity, the 1 cm³ block. Elements to krypton, hadron
  masses, parton showers, 1D hadronisation, truth mode, running α_s.
