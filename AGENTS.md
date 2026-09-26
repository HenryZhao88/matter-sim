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

| | Mac (M4, 16 GB) | Windows laptop (Ryzen 5, 7.8 GB, RTX 4050 6 GB) | **Downstairs PC** (Ryzen 9 5900X 12-core, 32 GB, RTX 3080 Ti 12 GB) | Linux cloud container (4 cores, 15 GB, no GPU) |
|---|---|---|---|---|
| Accelerator | MLX (Metal) | CUDA through PyTorch | CUDA through PyTorch | NumPy |
| Best at | float64 reference work, MLX paths, the viewer | fp32 DFT labelling: **~12× NumPy** (Al label 95 s vs 1121 s) | the heavy jobs: copper labelling on the GPU, iron's fcc scans on the CPU at the same time | short checks, watched by the human; ephemeral, so push everything |
| Avoid | fp32 DFT (MPS is 8.7× *slower* than NumPy here) | long GPU jobs from a Claude Code session: idle RAM is ~2 GB and the low-memory guard stops them (copper's 8-atom labels included) | nothing found yet. Measured: copper 4-atom labels (fp32) 658–1102 s, 15–42 SCF iterations, ~4.6 GB VRAM but only 21–35 % GPU use (under investigation), with iron's fcc scan on 9 CPU threads alongside; 8-atom untimed so far | anything over a few hours |

`engine/core/accel.py` decides what a machine uses: MLX, else CUDA, else MPS, else NumPy.
`MATTER_SIM_ACCEL=torch` forces the torch path (that is how the Mac tests MPS).

## Current state

| Rung | State |
|---|---|
| Particles and forces | working: collisions, decays, confinement, parton showers, running α_s, pp at 13.6 TeV |
| Lattice QCD | working: confinement, deconfinement, hadron masses (pion as Goldstone boson, κ_c = 0.1695 vs 0.1694 published) |
| Electrons and nuclei | working: H–Kr (Sc fails its own check and is greyed out), molecules, truth mode (VMC, MLX and PyTorch) |
| Materials | working for **aluminium**: melting 852 K (measured 933.5), expansion, heat capacity. GPU molecular dynamics (`md_torch.py`) reaches 10⁶ atoms, tested equal to `md.py`, not yet used by the experiments. Spin-polarised periodic DFT built; iron's pseudopotential fixed (D1), its grid not yet converged. **Copper**: grid converged, DFT lattice constant 3.548 Å; not yet labelled |
| Everyday matter | working: the 1 cm³ block, 6.3 × 10²² atoms, 2.83 g, 2.29 kJ to melt |

Results both machines can read are in `results/`. `HANDOFF.md` carries the detail and the
hard-won lessons; read it too.
**`DECISIONS.md` holds open decisions that change shared physics: read it, comment there
(dated, with your machine), and don't act on an open one until it is decided.**

## In flight

- **Copper: grid chosen, lattice constant computed, ready to label (Mac, 2026-09-25).** The grid
  error was copper's partial core density going through the nonlinear LDA on the plain grid, not
  its 3d states. `PeriodicDFT(xc_grid=2)` evaluates XC on a 2× grid and cuts the half-step egg-box
  from +15.6 to +0.56 meV/atom at h = 0.19. Copper's grid is **h = 0.19, xc_grid = 2**
  (`dataset.GRID[29]`). Against h = 0.16 it holds displacement energies to 0.1 meV/atom, strain
  energies to 1.2, forces to 2e-4 Ha/bohr; h = 0.22 misses strain by 9 meV/atom. Our own lattice
  constant is **a₀ = 3.548 Å = 6.704 bohr**, B = 168 GPa (`results/cu_eos.json`, k = 10³ — k = 12 moves a volume difference by 0.075 meV/atom — fit
  residuals ≤ 0.45 meV/atom). Build copper's training set around 6.704 bohr, not the measured
  3.615 Å. The labeller refuses an element with no `GRID` entry, and copper's cache names include
  its grid. **First copper label done (Windows, 2026-09-25):** the compressed fcc-volume cell
  (a × 0.90, 8³ k → 256, h 0.19, xc_grid 2) in 732 s on fp32, 15 SCF iterations, peak 2.0 GB VRAM
  and 1.3 GB RAM. The 8-atom attempt was stopped by Claude Code's low-memory guard before its first
  memory sample, so it is untimed. On Windows, "commit" tracks GPU memory one for one (a 4-atom
  label shows ~5.6 GB committed with 1.3 GB actually in RAM): watch the working set, not commit.
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
- **Spin-polarised periodic DFT exists (Linux cloud, 2026-09-25; NumPy path only).**
  `PeriodicDFT(spin=True, moments=...)`: LSDA, one Fermi level, so the magnetisation is free; the
  starting moment is only a push. Works with `xc_grid`. Unpolarised results are unchanged bit for
  bit (checked against main for Cu with xc_grid = 2 and Al). Aluminium pushed to 1 μB/atom relaxes
  to 3e-5 μB. `PeriodicDFTTorch` refuses `spin=True` (not ported).
- **Iron's pseudopotential collapses in the solid — blocks every iron number (Linux cloud).**
  Non-magnetic bcc Fe, h = 0.24, xc_grid = 2, 6³ k: E rises **+17.7, +17.2, +17.6 mHa/atom** per
  8 bohr³ from V = 56 to 80 (no minimum; at h = 0.30 it is still concave out to V = 170). The
  generator stretched the s-local radius to 1.5× base (r_s = 3.45 bohr) to avoid ghosts, because
  that candidate scores best on the *free-atom* test; the solid never enters that test. Its own
  ghost-free r_s = 2.30 candidate at the same settings: −20.9, −4.1, +4.8 (minimum V ≈ 69,
  a ≈ 2.75 Å); r_s = 2.88: minimum a ≈ 2.77 Å (coarse). No ghost band in the solid: the two
  potentials' band structures agree to 0.02 Ha, so the error is in the energy's volume
  dependence. `scripts/pp_check.py` reproduces all of it. **Lesson: a pseudopotential is not
  validated until a solid's E(V) has been looked at.** Copper's (1.25×) is fine, as its EOS shows.
  **Fixed (D1, applied 2026-09-25, Mac):** the stretch is capped at 1.25× and the pseudopotential
  cache is v5, so every machine regenerates on first use (a few minutes for Ti–Fe; the rest come
  back identical). Iron is now the 1.25× s-local potential with a minimum near a ≈ 2.75 Å. V, Cr
  and Mn changed too and are **not checked in a solid**. Ti and (on the Mac) V now fail their
  free-atom check. **Pseudopotentials for V/Mn differ between machines:** `DECISIONS.md` D2.

## What would help most

### Downstairs PC: start here (new machine, 2026-09-26)

You are the agent on the downstairs PC (Ryzen 9 5900X, 32 GB, RTX 3080 Ti). Read this file,
`DECISIONS.md` and `HANDOFF.md` first. Then:

1. **Set up.** `git pull`, then `uv sync --extra gpu` (the lock pulls CUDA PyTorch on Windows and
   Linux, see `pyproject.toml`). Check the GPU is seen:
   `uv run python -c "from engine.core.accel import describe; print(describe())"` should name the
   3080 Ti. Run the fast tests: `uv run pytest -m "not slow"`. Add a log line saying you've started.
2. **Label copper (item 1 below) on the GPU.** `uv run python scripts/label_crystal.py 29 6.704 torch 1`
   (71 configurations, resumable, 1 worker: the GPU does the work). Time the first 4-atom and the
   first 8-atom label and put the numbers in the Machines table. 8-atom cells are estimated at
   ~3.4 GB of GPU memory, which fits in 12 GB. The laptop could never run them.
3. **At the same time, on the CPU: iron's fcc phases (item 2).** The spin-polarised code is
   NumPy-only, so it doesn't compete with the GPU job:
   `OMP_NUM_THREADS=3 uv run python scripts/fe_magnetism.py 0.20 8 3 fcc-nonmagnetic,fcc-ferromagnetic,fcc-antiferromagnetic`
   (3 workers × 3 threads leaves cores for the GPU job; bound workers by RAM, and a 4-atom
   AFM point may need ~3 GB). The Linux container is running the bcc phases at the same grid;
   `git pull` before you finish and the script keeps their points in `results/fe_magnetism.json`.
   Commit that file as phases complete.
4. After both: `scripts/cross_check.py 29` (float64, ~5 %), the copper seed fit and MD snapshots
   (item 1's last bullet), then `uv run matter-sim validate --part materials` for iron's rows.

1. **Label copper's crystal set — open, needs a machine with more memory than the Windows laptop.**
   The human is finding an agent/machine for it; whoever takes it, say so in the log first.
   - Command: `uv run python scripts/label_crystal.py 29 6.704 <solver> [workers]` — 71
     configurations (`initial_configurations(29, 6.704)` without sc/disordered), resumable through
     the cache, grid from `dataset.GRID` (h 0.19, xc_grid 2), provenance on every label.
   - Solver: `torch` on a CUDA GPU (fp32, measured against float64 at h 0.19: −0.147 meV/atom,
     1.1e-5 Ha/bohr; run it with 1 worker, the GPU does the work). `numpy` anywhere else (float64;
     bound workers by RAM). Not `torch` on a Mac: MPS is slower than NumPy.
   - Cost, measured (Windows, fp32, RTX 4050): a 4-atom label 732 s, peak 2.0 GB VRAM, 1.3 GB RAM.
     8-atom cells (16 of the 71) are **untimed**: the only attempt was stopped by the low-memory
     guard. Estimated from sizes: ~3.4 GB VRAM on fp32; ~11 GB RAM per worker on NumPy.
   - Caches are per machine and not in git. The one finished label (the compressed fcc-volume cell,
     key `0f5146c079ed2647`) exists only in the Windows laptop's `.cache/materials/dft/`; another
     machine will simply recompute it.
   - After it: `scripts/cross_check.py 29` in float64 (~5%, per tag), a seed fit, then
     `md_snapshots(..., Z=29, mass_amu=63.546)` with temperatures chosen for copper (the defaults
     were aluminium's).
2. **Iron, then nickel and cobalt**, on the spin-polarised DFT. Iron is unblocked (D1 applied). Its
   grid is **h = 0.20, xc_grid = 2** (`scripts/fe_grid.py`, Linux: within 1.1 meV/atom of h = 0.16
   on volume energy, FM − NM and egg-box, 1e-4 μB on the moment; xc_grid moves it only 0.4 meV;
   below h ≈ 0.2 the volume energy scatters ±1.5 meV with no trend, as HANDOFF describes).
   `scripts/fe_magnetism.py` runs the scan; phases can be split across machines (bcc on the Linux
   container, fcc on the downstairs PC). Nickel and cobalt then need their own E(V) check
   (`scripts/pp_check.py eos`) and grid. Port spin to `periodic_torch.py` before labelling a
   magnetic metal on the GPU.
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
- **The obvious suspect is not always the cause.** Copper's grid error looked like unresolved 3d
  states; it was the partial core density in the XC functional. Wavefunction cutoffs did nothing,
  removing the core correction took 15.6 meV/atom to 0.2. Switch pieces off one at a time.
- **A variational energy below the exact answer is usually short-run noise**, not a bug: check
  by re-evaluating the *same* wavefunction far longer before theorising. (One agent theorised;
  the other's test was right.)

## Keeping this file honest

When you finish a stint, update **Current state**, **In flight**, and **What would help most**,
and add a dated line to the log. Keep it short: this file is a handover, not a diary. Delete
anything that is no longer true rather than appending a correction.

## Log

- **2026-09-26, Downstairs PC (Claude).** Started. Set up (uv was not installed: `pip install --user
  uv`, run as `python -m uv`); `describe()` names the RTX 3080 Ti; fast suite 113 passed, 3 skipped
  (542 s). **Taking copper labelling** (`label_crystal.py 29 6.704 torch 1`) and **iron's fcc
  phases** (`fe_magnetism.py 0.20 8 3 fcc-*`, OMP 3), both running here. Copper's first label is
  the laptop's cell (`0f5146c079ed2647`): E = −241.485978 Ha against the laptop's −241.485975
  (0.017 meV/atom), but 42 SCF iterations and 1102 s here against 15 and 732 s there. **Not a
  one-off:** the first four labels took 42, 15, 35, 39 iterations (1102, 658, 1038, 768 s), and the
  GPU sits at 21–35 % use with 4.6 of 12 GB, CPU at 39 %. At 15 iterations this GPU is barely faster
  than the laptop's (658 vs 732 s), so something besides the GPU sets the pace. Profiling one label
  now; unexplained so far. At this rate the 71 labels take ~20–30 h. fcc non-magnetic iron done:
  a₀ = 6.443 bohr, B = 328 GPa. fcc ferromagnetic: the starting moment collapses to zero at V = 60,
  64, 68 (E equal to non-magnetic); 16–36 min per spin point with 3 workers. Fixed `fe_magnetism.py`
  dropping another machine's phases when it rewrote the results file.
- **2026-09-26, Linux cloud container (Claude).** D1 applied by the Mac; iron's grid measured
  (`scripts/fe_grid.py`: h = 0.20, xc_grid = 2) and the bcc phases of `fe_magnetism.py` started
  here (running when this was written; results land in `results/fe_magnetism.json`). Added D2
  data for Linux. Wrote the downstairs PC's start-up instructions (What would help most).
- **2026-09-25, Windows (Claude).** Reviewed D1 independently (DECISIONS.md): agree with the 1.25×
  cap on the strength of iron's E(V), reproduced here to 0.01 mHa; the core-overlap reason does not
  hold (1.25× still overlaps by 1.06 bohr in bcc Fe, copper by 0.76). Added Windows data to D2: the
  reference atom converges here for Ti–Fe, V/Mn match Linux, Fe at 1.5× matches the Mac, so there
  is a second cross-machine difference.
- **2026-09-25, Mac (Claude).** Decided and applied D1 at the human's request, after reproducing
  iron's E(V) here (to 0.01 mHa). Pseudopotentials regenerated as v5: only Ti–Fe changed, copper
  bit for bit identical. Opened D2: the generator gives different V/Mn (Ti, Fe) pseudopotentials
  on different machines, probably from the unconverged all-electron reference atom.

- **2026-09-25, Linux cloud container (Claude; 4 cores, 15 GB, no GPU, ephemeral).** Checked
  this file against the code (held). Built spin-polarised periodic DFT (merged with main's
  `xc_grid`). Found iron's generated pseudopotential collapses in the solid. Independently found
  copper's egg-box was its partial core (the Mac fixed it the same week with `xc_grid`; my
  `scripts/eggbox.py` remains). Lesson: two NumPy jobs on 4 cores each start 4 BLAS threads and
  run **~10× slower** together (a 90 s copper pair took 900 s); one heavy job at a time.
- **2026-09-25, Windows (Claude).** Handed copper labelling back: this laptop cannot run the 8-atom
  labels under Claude Code's low-memory guard. Instructions for any machine are under "What would
  help most" #1. Nothing is running here.
- **2026-09-25, Windows (Claude).** Started copper labelling: 1 of 71 crystal configurations done
  (732 s, fp32, h 0.19 / xc_grid 2); the first 8-atom label was stopped by the low-memory guard.
  Added `scripts/label_crystal.py` (element-generic, resumable). Found that on Windows committed
  memory tracks GPU allocations, so RAM caps must use the working set. Fast crystal and materials
  tests pass on CUDA with the Mac's xc_grid changes (27 passed).
- **2026-09-25, Mac (Claude).** Copper unblocked. Found the grid error's cause (partial core ×
  nonlinear LDA; diagnosed by turning the pieces off one at a time) and added `xc_grid`, which is
  bit-for-bit the old code at 1. Converged copper's grid (h = 0.19, xc_grid = 2) and computed its
  lattice constant (3.548 Å, B 168 GPa). Grids are now per element in `dataset.GRID`; there is a
  resumable `scripts/eos.py`, and copper appears in `validation/run.py` (no pass mark). Fixed the
  GPU-MD tests on the Mac (MPS has no float64 → CPU). Recorded, not fixed: below h ≈ 0.19 the
  absolute energy scatters by a few meV/atom with grid size, from the grid products |ψ|² and V·ψ
  (see HANDOFF.md). Fast suite 117 passed.

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
