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
| Particles and forces | working: collisions, decays, confinement, parton showers, running α_s, pp at 13.6 TeV; one loop: lepton g−2 (a = 0.0011614 vs 0.0011597 measured) and α's running from α(0) |
| Lattice QCD | working: confinement, deconfinement, hadron masses (pion as Goldstone boson, κ_c = 0.1695 vs 0.1694 published) |
| Electrons and nuclei | working: H–Kr (Sc and Ti fail their own checks and are greyed out), molecules, LDA and PBE, truth mode (VMC, MLX and PyTorch) |
| Materials | working for **aluminium**: melting 852 K (measured 933.5), expansion, heat capacity. GPU molecular dynamics (`md_torch.py`, 10⁶ atoms) now runs the experiments (`engine="torch"`): melting in a 96 000-atom box gives **898 K** (1 200 atoms: 852–867 K). **Iron** (spin-polarised DFT, all five phases): bcc FM a = 2.794 Å (2.866), B = 218 GPa (170), 2.12 μB net (2.22); plain LDA puts fcc ~40 meV/atom below bcc FM (known LDA error, left disagreeing). **With PBE** (D3, built): bcc FM comes out lowest (fcc +140 meV/atom), a = 2.892 Å, B = 146 GPa, 2.39 μB net; all five iron checks pass. **Copper**: DFT lattice constant 3.548 Å (PBE 3.668), 89 labels, learned potential done; its experiments wait for element-generic MD |
| Everyday matter | working: the 1 cm³ block, 6.3 × 10²² atoms, 2.83 g, 2.29 kJ to melt |

Results both machines can read are in `results/`. `HANDOFF.md` carries the detail and the
hard-won lessons; read it too.
**`DECISIONS.md` holds open decisions that change shared physics: read it, comment there
(dated, with your machine), and don't act on an open one until it is decided.**

## In flight

- **Copper (done: melts at 1210 K, measured 1358).** Grid h = 0.19, xc_grid 2 (`dataset.GRID[29]`;
  the error at coarser grids was the partial core in LDA, not the 3d states). Our DFT lattice
  constant is **6.704 bohr = 3.548 Å** (LDA; PBE 3.668). Training set: 89 labels (71 crystal + 18 MD
  snapshots), 6 recomputed in float64 (worst 0.62 meV/atom). Final potential
  `results/cu_eam_final.*`: 14 meV/atom on unseen configurations, a₀ 3.553 Å. Experiments
  (`results/cu_results.json`, downstairs): T_melt 1209.8 K (1202–1217), latent heat 118.8 meV/atom
  (measured 137.4), melting expansion 4.7 %; all three validation rows pass.
- **Memory decides which h is usable.** Both DFT paths keep every k-point's wavefunctions resident.
  For copper's 8-atom cells (98 k-points, 50 bands): h = 0.19 needs 3.4 GB on fp32 and ~11 GB on
  NumPy; h = 0.16 (7.6 GB / ~25 GB) fits nowhere until wavefunctions are streamed per k-point or
  stored as G-sphere coefficients. The fp32 path against float64 at h = 0.16 is still owed.
- **Iron (magnetism done; PBE training set being labelled downstairs, 71 configurations).** Spin-polarised DFT runs on NumPy and the GPU
  (`PeriodicDFT(spin=True, moments=...)`, `PeriodicDFTTorch` too). Iron's pseudopotential was fixed
  by D1 (cap 1.25×), grid h = 0.20, xc_grid 2 under LDA and PBE. **LDA gets the structure wrong, PBE
  right** (`results/fe_magnetism_pbe.json`). `dataset.GRID[26]` now labels iron with PBE, spin-polarised
  from a 3 μB push (`label_crystal.py 26 5.4655 torch 1 bcc`; the bcc set is
  `initial_configurations(..., structure="bcc")`). First label: 2.32 μB/atom, 23 iterations.
  Open: PBE iron's a₀ 2.892 Å may be ~2% large. It is not the stretched s radius (a harder one
  gives 2.934); the next suspect is the frozen 3s3p semicore.
- **Pseudopotentials are v6** (D1 cap, D2 annealed reference atom). Every machine regenerates on
  first use; only Ti–Ni differ from v4. Sc (212 meV) and Ti (265) fail their checks. Linux's
  solid E(V) checks (`2b63d1f`: V, Cr, Mn, Co, Ni all have a minimum) were on v5. On the Mac, v6
  changed V and Mn to what Linux already had, but **Ni changed (49.9 → 129.8 meV)**: redo Ni's E(V)
  on v6 before using it. **Lesson: a pseudopotential is not validated until a solid's E(V) has been
  looked at.**

## What would help most

### Work queue by machine (updated 2026-09-26, Linux)

Take an item, say so in the log, and move it to the log when done. Items are ordered by value.

**Downstairs PC** (running: iron's PBE crystal set on the GPU, ~20–30 min a label, so ~1–2 days):
- **Iron's training set**, then its float64 cross-check (spin-polarised NumPy: slow, one at a time),
  a seed fit (`scripts/seed_fit.py 26 5.4655`), bcc MD snapshots (`md_snapshots` is fcc-only today)
  and the final fit. If the Mac changes iron's pseudopotential (the 2 % size question), the labels are
  stale: each records `pseudo_version`.
- Candidates after: cross-check copper's three MD tags (~7 h each on the CPU, one at a time).

**Windows (item 3 owner), a request from downstairs — done 2026-09-29, run half-finished:**
copper's experiments now run (`4212c63`, `7d852f4`): `md.fcc_state(a, n, mass_amu)`, every
experiment takes `mass_amu`/`engine`, and `run_all(..., schedule="auto")` scales every temperature
from the potential's own `temperature_scale` (a crystal heated in 50 K steps until its order goes;
Al 1450 K, Cu 1900 K; fractions justified by aluminium superheating 1.6× its coexistence melting
point). Aluminium's defaults are unchanged. `scripts/element_experiments.py Cu
results/cu_eam_final.npz 63.546 20 20 60` does it resumably. On the Windows laptop it got through
the scale, the thermal curve (a = 3.5546 Å at 133 K → 3.5898 Å at 950 K) and the first
coexistence point (1425 K: melts) before Claude Code's low-memory guard stopped it; the same
command resumes (5 coexistence points × ~6.5 min + latent heat left). Its cache is on the Windows
laptop only — another machine starts over (~45 min on a GPU). Copper's validation rows exist with
pass marks committed before any result (aluminium's tolerances).

**Mac** (float64 reference work):
- ~~PBE~~ done 2026-09-28 (D3): crystals, atoms, pseudopotentials, molecules, the GPU path, and
  labels carry the functional (`dataset.GRID[...]["functional"]`, in the cache name when not LDA).
  Open on this line:
  - **Iron's size.** PBE bcc iron comes out 2.892 Å; all-electron PBE is usually quoted near 2.83,
    so ours is ~2% large (LDA: 2.794 vs ~2.75). **Not the stretched s radius:** the harder
    r_s = 2.31 candidate gives 2.934 Å, larger still. Calibration on copper: ours is 3.548 (LDA) and
    3.668 Å (PBE) against all-electron ~3.52 and ~3.63, so ~0.8–0.9% large for both functionals.
    Iron's excess is about twice that. Next suspect: the frozen 3s3p semicore (8 valence electrons,
    core correction only). A 16-electron iron pseudopotential would test it, but the generator
    doesn't do semicore states. The literature values here are from memory; verify them before
    quoting.
  - **PBE molecules are slow:** relaxing water took 4,242 s against LDA's 378 (six full-grid FFTs
    per XC call, and more SCF iterations near convergence, where the density change floors at
    ~5e-6 while the energy converges). Profile before relying on it.
- ~~D2~~ done 2026-09-26 (pseudopotential cache v6; see DECISIONS.md).

**Linux cloud container** (short checks): both items done 2026-09-26 (see the log). Next candidates:
Co and Ni magnetism with `fe_magnetism.py`-style scans (Ni fcc FM, Co fcc FM as a first step; hcp Co
needs an orthorhombic hcp cell), or the fcc-FM metastability (does a finer moment start find the
lower state?).

**Any machine:**
- **GPU molecular dynamics at scale** (item 3), and **one-loop amplitudes** (item 4).

1. **Label copper's crystal set — labelled on the downstairs PC (2026-09-26, 16 h on the RTX 3080 Ti):
   66 of 71 converged, 5 refused** (fcc-volume at 1.00, 1.075 and 1.10 × a₀, two fcc-strain), from an
   fp32 eigensolver fault since fixed (`1cdae28`); the 5 are being relabelled with the fix, which
   agrees with the existing labels to 0.0002 meV/atom.
   - Command: `uv run python scripts/label_crystal.py 29 6.704 <solver> [workers]` — 71
     configurations (`initial_configurations(29, 6.704)` without sc/disordered), resumable through
     the cache, grid from `dataset.GRID` (h 0.19, xc_grid 2), provenance on every label.
   - Solver: `torch` on a CUDA GPU (fp32, measured against float64 at h 0.19: −0.147 meV/atom,
     1.1e-5 Ha/bohr; run it with 1 worker, the GPU does the work). `numpy` anywhere else (float64;
     bound workers by RAM). Not `torch` on a Mac: MPS is slower than NumPy.
   - Cost, measured (fp32): 4-atom 430–600 s on the RTX 3080 Ti when the CPU is free (732 s on the
     laptop's 4050), 8-atom 1100–1900 s, 2-atom bcc 330–540 s. An 8-atom cell holds ~11.6 GB of
     the 3080 Ti's 12 GB (allocator cache included).
   - Caches are per machine and not in git: all 66 copper labels are in the downstairs PC's
     `.cache/materials/dft/`. The fit and MD snapshots should run there.
   - Cross-check done: 6 float64 recomputations (one per tag), worst −0.62 meV/atom (the most
     compressed fcc-volume cell: single precision at high compression, since the fixed code reproduces
     it to 0.005 meV/atom; the rest ≤0.17) and
     2.9e-5 Ha/bohr, none over budget (`results/cu_crosscheck.json`).
   - Seed fit done (`results/cu_eam_seed.json`: a₀ 3.554 Å vs DFT 3.548, B 169 vs 168 GPa); 18 MD
     snapshots at 870/1450/2040 K in `results/cu_md_snapshots.json`, being labelled. Final
     potential `results/cu_eam_final.*` (see the log).
2. **Iron, then nickel and cobalt**, on the spin-polarised DFT. Iron is unblocked (D1 applied). Its
   grid is **h = 0.20, xc_grid = 2** (`scripts/fe_grid.py`, Linux: within 1.1 meV/atom of h = 0.16
   on volume energy, FM − NM and egg-box, 1e-4 μB on the moment; xc_grid moves it only 0.4 meV;
   below h ≈ 0.2 the volume energy scatters ±1.5 meV with no trend, as HANDOFF describes).
   `scripts/fe_magnetism.py` runs the scan; **all five phases are done** (bcc on Linux, fcc
   downstairs; 35/35 points converged). `matter-sim validate --part materials` gives 4/5 iron
   checks. The miss is plain LDA putting fcc about 40 meV/atom below bcc ferromagnetic, left
   disagreeing. Caveat: the per-start fits span moment collapses and metastable points (see the
   Linux work queue). Nickel and cobalt then need their own E(V) check
   (`scripts/pp_check.py eos`) and grid. Spin now runs on the GPU too (`PeriodicDFTTorch(spin=True)`,
   checked against NumPy on magnetic iron); the labeller does not pass `spin` yet.
3. **GPU molecular dynamics at scale — started (Windows, 2026-09-26).** `md.make_md(engine="torch")`
   runs the experiments on `md_torch.MDTorch` (tested equal to `md.py`); `scripts/melting_large.py`
   gives aluminium's melting point by coexistence at **898.4 K in a 96 000-atom box** (bracket
   890.6–906.2; 6 × ~290 s on the RTX 4050) against 851.6 K (Mac) and 867.2 K (Windows) at 1 200
   atoms, measured 933.5. The small box is biased low, not only noisy: at 875 K it melted and the
   large box froze. `results/al_melting_96000.json`; validation shows it as an extra row, while the
   headline melting point and the continuum block still use the 1 200-atom value — whether to
   switch them (and redo the latent heat at scale) is open. Next: grains, vacancies, a crack;
   float32 pair arithmetic for 2–4× if it passes the same tests.
4. **One-loop amplitudes — started (Windows, 2026-09-26).** `engine/particles/loops.py` does loops
   the way `amplitudes.py` does trees (numeric Dirac algebra, Feynman parameters, nothing written
   in). The QED vertex gives the lepton g−2: a = 0.00116141 for e and μ alike (α/2π to 1e-9);
   measured 0.00115965 (e) and 0.00116592 (μ), the rest being higher orders. Vacuum polarisation
   runs α from α(0) (new measured input `ALPHA_0`): leptons give 1/α(m_Z) = 132.7 against 127.95,
   the gap being hadronic vacuum polarisation, which perturbation theory cannot give (quark loops
   at Lagrangian masses land at 127.8, shown without a pass mark). Next: loops with W/Z/Higgs
   (the W-mass shift, the weak part of a_μ), and QCD corrections that need real emission (R ratio).

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

- **2026-09-30 01:30, Downstairs PC (Claude).** **Iron's magnetic labels: Pulay mixing can land on the
  non-magnetic state.** Refused cells (#9 at 1.10 a₀; #10, #11 4-atom thermal cells near equilibrium)
  were traced on the GPU: the cell's moment rose and then collapsed (#11: 3.2 → 0.2 μB), damped or not,
  with 22 or 28 bands. Pulay (DIIS) finds stationary points whether stable or not, and the non-magnetic
  state is always one. Mixing the magnetisation linearly until dρ < 0.1 e, then Pulay (`PeriodicDFT.mix_m
  = "hybrid"`), reaches the ferromagnet: #11 converges in 61 iterations to 2.22 μB/atom, 0.10 Ha per
  cell *below* the collapsed state. `dataset.label` now runs every spin-polarised label that way
  (`b8c08c3`), and gives spin runs bands for the majority spin, ⌈(valence + Σ|push|)/2⌉ + 6: the 8-atom
  cells had 38 for ~41 majority electrons. It also refuses a label whose top band holds electrons.
  Iron labelling restarted (9 two-atom labels kept: ferromagnetic, moment rising 1.51 → 2.81 μB with
  volume). Also: `md_snapshots(structure="bcc")` (copper's snapshots regenerate bit for bit); copper's
  MD tags are being cross-checked in float64 on the CPU.
  **Checked, not a problem:** the committed scans' fcc-FM points whose top band is full (occupation
  ~1). With 28 bands instead of 22, V = 80 PBE gives the same energy and moment to 1e-6.
  **Open, for whoever owns `fe_magnetism.py`:** the zero-moment fcc-FM points (LDA V ≤ 72, PBE V ≤ 64)
  and the metastable points ran on Pulay mixing and may be that stationary point rather than the
  lowest state. Rerunning them with `mix_m="hybrid"` would tell. Not claimed either way.
  **Lesson: a converged spin-polarised SCF is not proof of the right magnetic state.** Check the moment
  against volume, and prefer the lower-energy state.- **2026-09-29 17:45, Downstairs PC (Claude).** **Copper's experiments done here** (rerun from scratch, ~45 min
  on the 3080 Ti): melts at 1209.8 K (measured 1357.8), latent heat 118.8 meV/atom, melting
  expansion 4.7 %, all three rows pass; materials validation 12/13. **The labeller does magnetic metals
  and bcc**: `GRID` entries may carry a starting moment (in the cache name; each label records the
  moment it ends with and its `pseudo_version`), and `initial_configurations(structure="bcc")`. The fcc
  path is unchanged: all 94 Al and 94 Cu cache names checked identical. Iron's PBE set (71 labels)
  started on the GPU.
- **2026-09-29, Downstairs PC (Claude).** Taking (1) **copper's experiments**, rerun from the start here
  (scripts/element_experiments.py; the Windows cache is not portable), and (2) **the labeller for
  magnetic metals** (spin/moments through dataset), aimed at labelling iron with PBE on the GPU.
  Touching ngine/materials/dataset.py and scripts/label_crystal.py.
- **2026-09-29, Windows (Claude).** Copper's experiments made element-generic (downstairs request);
  copper's run half-done and stopped by the low-memory guard (see the Windows queue entry).
- **2026-09-28, Mac (Claude), second stint.** Labels carry their functional. PBE for molecules
  (`SCFSolver(functional="pbe")`, water relaxes to 0.972 Å / 104.45°; LDA 0.973 Å / 105.07°;
  measured 0.958 Å / 104.5°). The GPU path checked with PBE and spin (magnetic iron vs float64:
  same moment, 0.05 meV/atom). Copper under PBE: a₀ 3.668 Å, B 150 GPa (`results/cu_eos_pbe.json`).
  Iron's harder pseudopotential makes iron larger (2.934 Å), so the stretch is not why iron is big.
  Ran the **full slow suite** on the merged tree: 25/26. The failure was the downstairs fp32 fix
  (b) on MPS (complex128 cast on the device), fixed in `04c5f38`; now 26/26.

- **2026-09-28, Mac (Claude).** Built PBE (D3, decided by the human): `xc.pbe_xc`, and a
  `functional` switch through the radial atom, pseudopotential generation and cache, and the
  periodic solver (so the GPU path too). LDA is bit for bit unchanged. PBE atoms match published
  all-electron PBE (He −2.892935, Ne −128.866404, Ar −527.346034 Ha). Iron with PBE: grid h = 0.20,
  xc_grid 2 holds (vs 0.16: 1.1 meV/atom on the volume energy, 0.03 on FM − NM). Phase scan widened
  to V = 60–92 bohr³ (PBE's bcc minimum is at 81.6, too near the old edge of 84). **bcc
  ferromagnetic lowest, fcc +140 meV/atom**; a = 2.892 Å, B = 146 GPa, 2.39 μB net. The validation's
  iron moment row now compares the *net* moment with the measured net 2.22 μB. It had compared the
  absolute ∫|m| (LDA 2.163 → 2.123 net; PBE 2.507 → 2.385); bound unchanged, both pass.

- **2026-09-27 13:30, Downstairs PC (Claude).** **fp32 fix (b) done: LOBPCG's Rayleigh–Ritz grams run in
  complex64 on CUDA** (`gram_dtype`, default complex64 on CUDA, complex128 elsewhere; labels
  record which). Profile first: 76 % of the GPU's busy time in an SCF step was those complex128 grams.
  A copper 4-atom label goes **379 → 207 s, same 20 iterations** (486 s before fix (a): 2.35× overall);
  8-atom ~470 s (1100–1900 in production). The first version failed on perfect fcc Cu (#5: 60
  iterations, 19 meV/atom off). A complex64 gram over N points is good to ε·√N, not ε, so the
  ε drop threshold admitted noise and the bands lost orthonormality (1e-6..5e-3). Fixed by dropping
  directions below ε·√N and by orthonormalising each solve's start in complex128. Then **all 6
  cross-checked copper labels** agree with their stored labels to ≤0.038 meV/atom and ≤2.7e-5 Ha/bohr,
  and with float64 as well as complex128 does (worst −0.617 meV/atom, 2.4e-5); #5 converges in 17. New
  slow test on perfect copper at h = 0.25, checked to fail with the broken threshold (it does not at
  h = 0.35). Fast suite 118 passed; single-precision tests 10 + 2 passed.
- **2026-09-27 08:10, Downstairs PC (Claude).** Copper's float64 cross-check done (16.6 h serial):
  6/6 within budget, worst −0.62 meV/atom and 2.9e-5 Ha/bohr; bcc 0.001, strain 0.014, thermal
  −0.16/−0.17, fcc-8 −0.11. The −0.62 is the most compressed cell, whose label is the one the
  old LOBPCG took 42 iterations over. It is within budget, so it was left in. This machine's
  queue is empty; nothing is running here.
- **2026-09-26 22:45, Downstairs PC (Claude).** **Copper's training set is complete: 89 DFT labels, none
  refused** (71 crystal + 18 MD snapshots at 870/1450/2040 K, ~17 min each on the 3080 Ti). **Final
  potential** `results/cu_eam_final.*`: a₀ 3.553 Å (DFT 3.548), B 181 GPa (DFT 168; 8 % stiff),
  bcc − fcc 37 meV/atom (labels 42). On the MD snapshots it is within 1.6–2.8 meV/atom and ≤0.09 eV/Å.
  Its error sits in the heavily displaced fcc-8 cells (1.0 eV/Å rms, pairs at 3.2 bohr, forces to
  10.8 eV/Å), which dominate the test numbers (14 meV/atom, 0.82 eV/Å). `eam.fit` gained a
  PyTorch Adam stage for machines without MLX (before, they went straight to least squares from a
  rough guess). Copper's experiments wait on Windows (request above). Killed non-essential desktop
  apps at the human's request: Claude Code's low-memory guard had fired with 1 GB free (the float64
  8-atom cross-check holds ~12.7 GB).
- **2026-09-26 16:30, Downstairs PC (Claude).** **The refused copper labels and the iteration spread
  were one fp32 fault, now fixed** (`1cdae28`): LOBPCG kept Rayleigh–Ritz directions down to 1e-10
  of the largest overlap, below complex64's resolution. On label #5 one kept direction (1.2e-10)
  broke orthonormality (8e-5) and put the lowest band at −45.5 Ha (true +0.16); the SCF never
  recovered. The drop threshold is now ε of the block precision. With the fix, #5 converges in 17
  iterations, #1 takes 16 (42 before) and #33 takes 22 (56 before), and a recomputed label agrees
  with its stored one to 0.0002 meV/atom. Timed fix (a): 486 → 384 s on one label, same iterations.
  **`eam.py` is per element now**: z, r_min, core_in and core_out live on the model, aluminium's are
  the defaults (its saved model is bit for bit unchanged), and `short_range_for` sets them by
  aluminium's rule. Copper's data reaches 3.20 bohr, inside aluminium's splice, and the
  repulsion had used Z = 13. Seed fit: test 3.5 meV/atom, 0.12 eV/Å; it misses one 3.20-bohr cell's
  8.9 eV/Å force by 6.6. **All 5 refused cells relabelled with the fix: copper's crystal set is 71/71
  converged.** MD snapshots labelling next (~8 h of GPU). Cross-check 1/6 after 1.6 h; its NumPy worker
  holds 8.6 GB, and Claude Code's low-memory guard fired at 3.8 GB free (the 8-atom check will need more).
- **2026-09-26 15:10, Downstairs PC (Claude).** **Copper labelling finished**: 66 converged, 5 refused
  (#5, #8, #9 fcc-volume at 1.00/1.075/1.10 × a₀; #36, #44 fcc-strain), 57 690 s in all. **GPU spin
  verified:** all 8 `single_precision` tests pass on the final code (slow included). Displaced magnetic
  bcc Fe (h 0.35, 2³ k, xc_grid 2): GPU against NumPy −0.064 meV/atom, max |ΔF| 1.9e-5 Ha/bohr,
  moment 1.9228 μB/atom on both, 19 SCF iterations each.
- **2026-09-26, Linux cloud container (Claude).** Iron's lowest-crystal row now ranks each
  structure's lower envelope (fcc non-magnetic lowest, bcc FM +44 meV/atom; still a miss, LDA's).
  E(V) checks, v5 potentials, non-magnetic, h 0.24/xc 2/6³ k: Ni fcc a ≈ 3.49 Å (3.524), Co fcc 3.45,
  Cr bcc 2.84 (2.91), V bcc 3.00 (3.03), Mn bcc 2.76; all convex with a minimum, so D1's condition
  for V, Cr, Mn is met (numbers in DECISIONS.md D1).
- **2026-09-26, Windows (Claude).** Items 3 and 4 started (see "What would help most"): GPU MD runs
  the experiments; aluminium melts at 898 K in a 96 000-atom box. One-loop QED in
  `engine/particles/loops.py`: lepton g−2 and the running of α, with validation rows.
- **2026-09-26, Mac (Claude).** Fixed D2. The reference atom's 4s/3d sloshing made its SCF
  converge by chance, after an iteration count rounding decides (Mac: V 432, Mn 814, Ni 838, Ti
  347; the cap is 400). `pseudo.reference_atom` anneals the smearing when the plain SCF fails, and
  the cache is now v6. Mac V/Mn now equal the other machines (77.0, 58.0 meV). Ni moved 49.9 →
  129.8 meV (still passes; its solid E(V) is unchecked, now on v6). Every other element is bit for
  bit unchanged, copper included.
- **2026-09-26 12:30, Downstairs PC (Claude).** Copper labelling 59/71 (57 converged, **2 refused**:
  #36 and #44, fcc-strain, SCF not converged in 60 iterations; a rerun retries them). **8-atom
  cells timed: 1102–1865 s, 22–28 SCF iterations** (#50–59, fp32). Once iron's scan stopped sharing
  the CPU, the 4-atom labels fell to 430–600 s and ~20 iterations (35–42 before). So part of the slowness
  was CPU contention, not only complex128. **During 8-atom labels the job holds ~11.6 of 12 GB
  of GPU memory** (the allocator's cache included; the 3.4 GB estimate was for wavefunctions), so
  nothing else fits on the GPU: a benchmark started alongside it hung in `eigh` and was stopped.
  Took the fp32 speed-up and the spin port (`44fc2a9`): fix (a) (`small_device="cpu"` in
  `lobpcg_dev`) and `PeriodicDFTTorch(spin=True)`. Fast `single_precision` tests pass on CUDA (5,
  the new spin one included). **Still owed:** a timing A/B of fix (a) on a full label, and the slow
  tests on the final code (`test_single_precision_spin_polarised_iron_matches_numpy`, the copper
  ones). The copper run in progress uses the code it started with.
- **2026-09-26, Windows (Claude).** Taking "Any machine": item 3 (GPU molecular dynamics at scale,
  starting with melting by coexistence at ~10⁵ atoms) and item 4 (one-loop amplitudes). Touching
  `engine/materials/md*.py`, `experiments.py` and the particle rung; will log what else.
- **2026-09-26, Linux cloud container (Claude).** Checked iron's fcc results: 35/35 points
  converged. Found metastable magnetic points (fcc FM at V = 76 is 13 meV/atom above non-magnetic
  with 1.04 μB), so the per-start fits mix branches and the lowest-phase ranking (5–9 meV spread)
  is within that noise; the robust fact is fcc ≈ 40 meV below bcc FM. Replaced the downstairs
  start-up section (done) with a work queue by machine. D2: the hashes differ even with the same
  recipe, so compare by max |Δv_ion|.
- **2026-09-26 02:45, Downstairs PC (Claude).** Iron's fcc phases done (21/21 converged, 16–58 min per
  spin point): fcc FM moment zero for V ≤ 72, 1.0/2.5/2.6 μB at 76/80/84; layered AFM ±0.6–1.9 μB
  from V = 68 and the lowest fcc state for V = 72–80. `validation/run.py`'s lowest-phase row now
  says when a "magnetic" phase's moment returned to zero (it had printed fcc-ferromagnetic as the
  winner with 0.0001 μB). Materials validation: iron 4/5, as Linux found. D2 data added. Copper
  labelling still running (15/71 at 02:33).
- **2026-09-26, Downstairs PC (Claude).** Started. Set up (uv was not installed: `pip install --user
  uv`, run as `python -m uv`); `describe()` names the RTX 3080 Ti; fast suite 113 passed, 3 skipped
  (542 s). **Taking copper labelling** (`label_crystal.py 29 6.704 torch 1`) and **iron's fcc
  phases** (`fe_magnetism.py 0.20 8 3 fcc-*`, OMP 3), both running here. Copper's first label is
  the laptop's cell (`0f5146c079ed2647`): E = −241.485978 Ha against the laptop's −241.485975
  (0.017 meV/atom), but 42 SCF iterations and 1102 s here against 15 and 732 s there. **Not a
  one-off:** the first four labels took 42, 15, 35, 39 iterations (1102, 658, 1038, 768 s), and the
  GPU sits at 21–35 % use with 4.6 of 12 GB, CPU at 39 %. At 15 iterations this GPU is barely faster
  than the laptop's (658 vs 732 s). At this rate the 71 labels take ~20–30 h. **Profiled** (py-spy on
  the running label, 2 min, 5948 samples): 80 % in `_eig_dev`, and 60 % at the two `torch.linalg.eigh`
  calls in `lobpcg_dev` (`periodic_torch.py:320` 49 %, where the GPU syncs, so it absorbs the queued
  work before it; `:323` 11 %). The queued work there is the Rayleigh–Ritz `gram`, which runs in
  **complex128 on the GPU**: 46 ms for a 150-row block over 36³ points on the 3080 Ti, against 1.6 ms
  in complex64 (29×), and a 150² complex128 `eigh` is 11–16 ms on the GPU against 2–5 ms on the CPU.
  GeForce cards run float64 at 1/64 rate, which is why the 3080 Ti barely beats the 4050 (both
  GeForce). The complex128 is deliberate (commit `747153f`; see the fp32 lesson below), so it is
  unchanged. Possible fixes, each needing the fp32-vs-NumPy comparison before production use:
  (a) the small `eigh`s on the CPU, which changes no precision; (b) the grams as split complex64 products
  accumulated in complex128 (error-free splitting), several times cheaper than complex128 on
  GeForce. The 15-vs-42 SCF iteration spread is separate and still unexplained. fcc non-magnetic iron done:
  a₀ = 6.443 bohr, B = 328 GPa. fcc ferromagnetic: the starting moment collapses to zero at V = 60,
  64, 68 (E equal to non-magnetic); 16–36 min per spin point with 3 workers. Fixed `fe_magnetism.py`
  dropping another machine's phases when it rewrote the results file.
- **2026-09-26, Linux cloud container (Claude).** D1 applied by the Mac; iron's grid measured
  (`scripts/fe_grid.py`: h = 0.20, xc_grid = 2) and the bcc phases of `fe_magnetism.py` run
  here and **done**, merged with downstairs' fcc in `results/fe_magnetism.json`: bcc FM a = 2.794 Å,
  B = 218 GPa, 2.163 μB, FM − NM −336 meV/atom; fcc NM lies 45 meV below bcc FM (plain LDA's
  known error, left as is). fcc AFM still to come from downstairs. Added D2
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
