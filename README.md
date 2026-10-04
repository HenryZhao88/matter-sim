# matter-sim

A first-principles simulator of matter, from particle collisions up to a block of metal you could
hold. The only physics written into the code is the laws themselves:

- **The Standard Model Lagrangian**, written as structure: the gauge group
  SU(3)×SU(2)×U(1), how each field transforms, the Higgs potential and the Yukawa couplings.
- **For atoms**, its low-energy form: the Schrödinger equation for electrons and nuclei, with
  Coulomb's law and Pauli exclusion.
- **Measured constants**: 18 of them for particles (α, G_F, m_Z, α_s, the fermion and Higgs
  masses, the CKM mixing matrix), and each nucleus's charge and mass for atoms.

Everything else has to **emerge**, and none of it is written anywhere in the code:
- particles and their charges;
- the W and Z masses;
- which collisions make what;
- decays and lifetimes;
- confinement;
- shell structure and Hund's rule;
- chemical bonds, molecular shapes and magnetism;
- crystal structures, lattice constants, ferromagnetism in iron, melting.

It runs on Apple Silicon (Metal GPU through [MLX](https://github.com/ml-explore/mlx)) and on
NVIDIA GPUs (CUDA through PyTorch, the optional `gpu` extra), with NumPy as the reference
everywhere, and shows everything live in a browser.

```bash
uv run matter-sim                           # opens the viewer at http://127.0.0.1:8765
uv run matter-sim validate                  # recomputes every result below (~1.5 h)
uv run matter-sim validate --part particles # just the particle rung (~1 min)
```

The first launch builds the viewer (needs Node.js) and derives the decay tables (a few minutes,
once). The first proton collision fetches the proton's measured structure (0.5 MB) and computes
870 parton–parton tables on every core (about 30 minutes on an M4, once). After that, everything
is cached.

Several agents work on this repository from different machines. `AGENTS.md` is how they hand
work to each other: what the project is, what it stands on, what is in flight, and the mistakes
already paid for. `DECISIONS.md` holds choices that change shared physics, with the evidence.
`HANDOFF.md` carries the longer technical notes. Read them before changing anything.

## The ladder of scales

The scale bar at the top of the viewer switches between the rungs.

| Rung | What is simulated | How |
|---|---|---|
| **Particles and forces** | collisions, decays, confinement | Standard Model amplitudes; lattice gauge theory |
| **Electrons and nuclei** | atoms and molecules | quantum electrons (DFT/Hartree–Fock), moving nuclei |
| **Materials** | metal crystals and up to 10⁶ atoms in motion, melting and expanding | periodic DFT (spin-polarised, LDA or PBE); forces learned from that DFT, then molecular dynamics on the GPU |
| **Everyday matter** | a cubic centimetre of aluminium | a continuum built from the per-atom properties measured above |

## What emerges: particles and forces

Most numbers are tree-level (lowest order); the gaps to experiment are the known size of the
loop corrections that level leaves out. The first one-loop results are at the end of the table.

| Result | Simulated | Experiment | Notes |
|---|---|---|---|
| Photon mass after symmetry breaking | 0 | 0 | a massless photon falls out of the Higgs mechanism |
| W boson mass | 79.83 GeV | 80.37 GeV | predicted from α, G_F, m_Z |
| Quark and lepton charges | +⅔, −⅓, −1, 0 | same | from weak isospin and hypercharge |
| Muon lifetime | 2.192 μs | 2.197 μs | μ → e ν ν̄ through a virtual W |
| Z → invisible | 20.6% | 20.0% | counts three neutrino families |
| R = hadrons/μμ at 10 GeV | 3.58 | 11/3 below the b threshold | three quark colours |
| e⁺e⁻ → W⁺W⁻ at 200 GeV | 19.2 pb | ≈17 pb | falls at high energy only because the gauge structure is exact |
| gg → gg, uū → gg | analytic to 10⁻⁹ | QCD textbook | triple and quartic gluon vertices |
| Top quark decays before forming hadrons, bottom does not | yes | yes | width vs Λ_QCD, both computed |
| Pair creation, string breaking (1D QED) | yes | Schwinger mechanism | exact real-time evolution |
| Lattice QCD plaquette at β = 6.0 | 0.5938 | 0.5937 | pure gauge |
| Quark potential | rises linearly | confinement | string tension from Wilson loops |
| Polyakov loop | jumps near β ≈ 5.7 | 5.69 | quark–gluon plasma transition |
| pp → t t̄ at 13.6 TeV | 777 pb | ≈ 900 pb | partons measured, collision computed (leading order) |
| pp → Z/γ* → μμ | 1.27 nb | ≈ 2.0 nb | leading order, scattering angle cut at cos θ* = ±0.95 |
| W⁺/W⁻ production ratio | 1.36 | ≈ 1.3 | because the proton is uud |
| Colour factors C_F, C_A | 4/3, 3 | same | computed from the SU(3) generators, not typed in |
| α_s(10 GeV) from α_s(m_Z) | 0.173 | 0.178 | one loop, β₀ from those group constants |
| Jets: gluons radiate more than quarks | 1.2–1.7× the particles | C_A/C_F = 2.25 at high energy | parton shower from the splitting functions |
| Pion mass² vs quark mass (lattice) | straight line to zero at κ = 0.1695 | 0.1694 published | the pion is a Goldstone boson |
| Rho mass in the chiral limit | 0.555 /a | ≈ 0.56 published | stays heavy while the pion vanishes |
| 1D hadronisation | charges end up screened | string breaks into mesons | a flying pair, evolved exactly |
| Electron g − 2 (one loop) | 0.0011614 | 0.0011597 | the QED vertex from the same Feynman rules; equals α/2π to 1e-9; the α² term is missing |
| Muon g − 2 (one loop) | 0.0011614 | 0.0011659 | the same number: mass-independent at one loop; the rest is α², hadronic and weak loops |
| 1/α at m_Z from α(0), lepton loops | 132.7 | 127.95 | **fails**, as it should: the rest is hadronic vacuum polarisation, not perturbative |

### The three particle tools

- **Collider.** Choose beams (e⁺e⁻, μ⁺μ⁻, γγ, quarks, gluons, or protons at 13.6 TeV) and
  collide them. For protons, a trigger keeps only the rare collisions that make leptons, photons,
  W, Z, top quarks or the Higgs, which is about 1 in 10⁵. It draws them from the exact conditional
  distribution, the same way a real detector trigger selects its events.
  - Every final state is tried with amplitudes built from the vertices, so if the Lagrangian
    forbids an outcome, its rate comes out exactly zero.
  - Unstable particles decay by their computed branching ratios and lifetimes.
  - A detector view bends charged tracks in a 3.8 T field and shows missing momentum.
- **1D world.** A universe with one space dimension (QED in 1+1D on a lattice), evolved exactly.
  Strong fields tear pairs out of the vacuum, strings between charges snap, and colliding
  mesons merge and scatter.
- **Lattice QCD.** Gluons on a 4D spacetime grid. It measures the energy between two quarks
  (confinement) and heats the field until confinement switches off.

### Where it stops

- **Hadronisation in 3D.** Fast quarks and gluons now radiate: a parton shower built from the
  Altarelli–Parisi splitting functions, with colour factors and the running coupling taken from
  the group itself, follows them down to about 1 GeV. Below that, confinement takes over, and
  turning partons into hadrons in real time is not something any known classical algorithm does
  from first principles (the sign problem). In one space dimension there is no such barrier, and
  the "Hadronisation" experiment in the 1D world shows the whole process exactly: a pair flies
  apart, the field string between them snaps, and the fragments come out as neutral mesons.
- **Proton structure.** The quark and gluon content of the proton (parton distributions, CT14 LO)
  is measured input, just like the particle masses. Only hard proton collisions are generated,
  where partons meet with more than 50 GeV. Most of the 100 mb pp cross section is soft,
  non-perturbative scattering, and it is not shown.
- **Mostly tree level.** One-loop QED is in (`engine/particles/loops.py`: the vertex correction
  and vacuum polarisation), but collision and decay amplitudes are still tree level: no weak or
  QCD loops yet, so H → gg and H → γγ are missing and the Higgs's branching ratios are two-body
  tree level only. The strong coupling does run with the collision energy (one loop, with the
  quark thresholds read off the model), so the leading effect of loops on rates is there.

## What emerges: atoms and molecules

| Result | Simulated | Experiment | Notes |
|---|---|---|---|
| Hydrogen Lyman-α / Balmer-α | 121.5 / 656.1 nm | 121.6 / 656.3 nm | exact Hamiltonian |
| Ionisation energies H–Ar | within 0.65 eV | NIST | peaks at He, Ne, Ar; dips at Li, B, O, Na, Al, S |
| Filling order and Hund's rule | C, N, O: 2, 3, 2 unpaired | same | found by minimising energy |
| H₂ bond / binding energy | 0.768 Å / 4.89 eV | 0.741 Å / 4.75 eV | forms from two separate atoms |
| H₂ with parallel spins | repels | unbound | Pauli exclusion |
| H₃⁺ | equilateral, 0.908 Å | equilateral, 0.873 Å | started as a bent chain |
| Water | 105.2°, 0.973 Å | 104.5°, 0.957 Å | started at 150° |
| Ammonia | pyramid, 107.9° | 106.7° | started nearly flat |
| Methane | 109.3–109.6° | 109.47° | started lopsided |
| N₂ | 1.101 Å | 1.098 Å | started stretched |
| H₂S | 91.7° | 92.1° | 13° flatter than water: the periodic trend |
| NaCl | 2.377 Å | 2.361 Å | ionic bond |
| O atom, O₂, Al₂ | 2 unpaired spins each | triplets | started spin-paired |
| Copper's configuration | 4s¹ 3d¹⁰ | 4s¹ 3d¹⁰ | the textbook exception, unprompted |
| Helium, exact Hamiltonian (truth mode) | −2.897 ± 0.004 Ha | −2.9037 | Hartree–Fock gives −2.8617 |
| H₂, exact Hamiltonian (truth mode) | −1.170 ± 0.004 Ha | −1.1745 | no functional, no basis set |

The "LDA" approximation for electron correlation accounts for most of the remaining gaps. In
LDA, water is about 105° and H₂ about 0.765 Å. The gradient-corrected PBE functional is also
available (`functional="pbe"`, with its own pseudopotentials); with it water relaxes to 104.45°
and 0.972 Å. PBE relaxations are about 10× slower than LDA's for now.

**Truth mode** removes that approximation for small systems: a neural-network wavefunction
(FermiNet-style, trained on the GPU) is varied against the exact many-electron Hamiltonian, with
the electron–electron cusps built in and antisymmetry from determinants. It is slower by orders
of magnitude and limited to a handful of electrons, so it serves as the reference the cheap
methods are measured against.

**Which elements are available.** Hydrogen and helium are simulated all-electron. Lithium to
krypton use pseudopotentials the engine derives from its own all-electron atoms, each one
checked for ghost states and for transferability across ionised and promoted configurations
before it is offered. Everything up to argon agrees to within 0.06 eV. The fourth row is looser
(4 meV for zinc, 53 for copper, 146 for cobalt). Scandium (212 meV) and titanium (265 meV)
currently fail their own check against the 150 meV limit, so they stay greyed out rather than
quietly giving wrong answers. Iron's first pseudopotential passed every atomic check but
collapsed in the solid; the generator now caps how far it softens a channel (`DECISIONS.md`, D1).
Relativity is not in the atom solver yet, which matters most for the heaviest of these.

## What emerges: a piece of metal

Aluminium, by the chain the ladder is for. Periodic DFT computes the crystal; a potential is
fitted to nothing but that DFT (89 configurations: volumes, strains, thermal disorder, a
supercell, bcc, and snapshots of hot liquid); a few hundred atoms then move on those forces;
and the per-atom properties they produce define a block you could hold.

| Result | Simulated | Experiment | Notes |
|---|---|---|---|
| Lattice constant (0 K) | 3.955 Å | 4.046 Å at 293 K | LDA under-estimates; this is the DFT value the potential reproduces to 0.4 mÅ |
| Bulk modulus | 84.5 GPa | 76 GPa | from the DFT equation of state |
| Elastic constants C₁₁, C₁₂, C₄₄ | 124, 65, 39 GPa | 107, 61, 28 | strained periodic cells |
| bcc above fcc | 108 meV/atom | fcc is the stable one | the learned potential agrees to 1 meV |
| Melting point | 852 K | 933.5 K | solid and liquid in one box of 1 200 atoms; whichever grows, wins |
| Melting point, 96 000 atoms (GPU) | 898 K | 933.5 K | the small box is biased low; the block below still uses 852 K |
| Volume change on melting | 3.2% | 6.5% | |
| Latent heat of fusion | 70 meV/atom | 111 meV/atom | **the weakest number here** |
| Thermal expansion | 27.3 ×10⁻⁶/K | 23.1 | classical nuclei |
| Specific heat at 293 K | 0.97 J/g·K | 0.897 | classical nuclei: no quantum freeze-out |
| Density at 293 K | 2828 kg/m³ | 2699 | from the lattice constant and the nuclear mass |
| Speed of sound, longitudinal | 6815 m/s | 6420 | from the elastic constants |
| 1 cm³ block | 6.3 × 10²² atoms, 2.83 g | 2.70 g | |
| Energy to melt that block from room temperature | 2.29 kJ | ≈2.9 kJ | |

The learned potential reproduces the DFT it was trained on to 6.5 meV per atom on configurations
it never saw. The gaps above are mostly the gap between LDA and nature, which the atoms rung
already has, carried upward — plus, in the latent heat, a real weakness of this potential.

**What that took, and what it cost.** The training labels have to be converged in k-points to
a few meV per atom, and *consistently* across cells: the first attempt used a mesh that varied
with cell size, leaving ~30 meV/atom of inconsistency, and no potential can fit numbers that
disagree with each other — the bulk modulus came out at 44 GPa against DFT's 86. Three further
faults only showed up as dynamics that exploded: a fitted density that crossed zero (an electron
density cannot), no repulsion at distances the training data never sampled, and a barostat that
boiled the metal into vacuum while reporting a plausible-looking melting point. Each is fixed in
the physics rather than papered over; `engine/materials/` says where.

## What emerges: copper and iron

The same chain, started on two more metals. Copper's crystal comes from its own DFT (never from
the measured lattice constant), its training set is labelled (89 configurations, 6 recomputed
in float64 to check the GPU labels: worst 0.62 meV/atom), and its learned potential is fitted
and measured in a 96 000-atom box. Iron needed spin-polarised DFT, since its structure comes from
its magnetism; its potential is trained on 86 PBE labels and melts in a 93 750-atom box. The
potential has no magnetism of its own, so it stays bcc up to melting (real iron is fcc from 1185 to
1667 K and melts from bcc again).

| Result | Simulated | Experiment | Notes |
|---|---|---|---|
| Copper lattice constant, LDA / PBE | 3.548 / 3.668 Å | 3.603 Å (0 K) | static lattice; LDA short and PBE long, as usual |
| Copper bulk modulus, LDA / PBE | 168 / 150 GPa | 142 GPa | |
| Copper's learned potential | a₀ 3.553 Å, bcc − fcc 37 meV | DFT: 3.548 Å, 42 meV | 14 meV/atom on configurations it never saw |
| Copper melting point | 1210 K | 1358 K | solid–liquid coexistence, 96 000 atoms |
| Copper latent heat of fusion | 118.8 meV/atom | 137.4 meV/atom | melting expansion 4.7 % |
| Copper thermal expansion at 293 K | 8.9 × 10⁻⁶/K | 16.5 × 10⁻⁶/K | classical nuclei |
| Iron chooses to be a magnet | yes: bcc FM − NM = −581 meV/atom (PBE) | ferromagnetic | the starting moment is only a push; aluminium gives it back |
| Iron's crystal, LDA | fcc non-magnetic, 44 meV below bcc | bcc ferromagnetic | **wrong**: plain LDA's known failure for iron, left disagreeing |
| Iron's crystal, PBE | **bcc ferromagnetic**, fcc 140 meV above | bcc ferromagnetic | the gradient correction fixes it; nothing tuned |
| Iron lattice constant (bcc FM), LDA / PBE | 2.794 / 2.892 Å | 2.866 Å | possibly ~2% large for a pseudopotential reason: under investigation (AGENTS.md) |
| Iron bulk modulus, LDA / PBE | 218 / 146 GPa | 170 GPa | |
| Iron magnetic moment, LDA / PBE | 2.12 / 2.39 μB | 2.22 μB | net moment at the computed lattice constant |
| Iron's learned potential (PBE labels) | a₀ 2.901 Å, fcc − bcc 97 meV | DFT: 2.892 Å, 100 meV | 5.8 meV/atom on configurations it never saw |
| Iron melting point | 2058 K | 1811 K | **too high**: the potential over-favours the solid |
| Iron latent heat of fusion | 176.9 meV/atom | 143.1 meV/atom | melting expansion 13.6 % (measured ~3.4 %); 204 without the solid run nearest T_m |
| Iron thermal expansion at 293 K | 7.9 × 10⁻⁶/K | 11.8 × 10⁻⁶/K | classical nuclei, no magnetism in the potential |

## How it works

**Standard Model** (`engine/particles/`):
- `model.py` diagonalises the gauge-boson mass matrix from |D_μ⟨H⟩|². The W, Z and photon,
  and every coupling, fall out of that.
- `amplitudes.py` builds tree-level amplitudes for any process with Berends–Giele recursion
  over the vertex list, using explicit spinors, polarisations and colours.
- `decays.py` tries every final state to find the open channels.
- `events.py` generates collisions.
- `hadron.py` combines those collisions with the measured proton structure.

**Lattice gauge theory** (`engine/lattice/`):
- `schwinger.py`: QED in 1+1D (Kogut–Susskind fermions, Gauss's law), evolved by Krylov
  exponentiation in the zero-charge sector.
- `qcd.py`: the SU(3) Wilson action with a Cabibbo–Marinari heatbath.
- `hadrons.py`: quarks in that gluon field through the Wilson–Dirac operator, inverted on the
  GPU with conjugate gradients from smeared sources. Hadron masses come from how fast the
  quark correlations fade in time.

**Electrons** (`engine/electrons/`):
- Solved on a 3D grid with spectral operators, so kinetic energy and open-boundary Coulomb
  interactions are exact for the grid.
- The electron–electron interaction uses LDA (exact uniform-gas exchange plus the Perdew–Wang
  correlation fit to quantum Monte Carlo of the uniform electron gas), PBE (the generalised
  gradient approximation, whose constants all come from exact conditions), Hartree–Fock, or a
  single-electron exact mode.
- Nuclei feel exact Hellmann–Feynman forces.

**Core electrons** (`engine/atoms/`):
- A spherical all-electron solver (Numerov on a log grid) computes every atom from H to Kr,
  starting from a Thomas–Fermi atom so heavy shells stay bound on the way to convergence.
- The engine builds Troullier–Martins pseudopotentials from those atoms, with d projectors and
  a partial core density where core and valence overlap.
- Each one is checked for ghost states (the separable form's spectrum against the semilocal
  one) and for transferability across ionised and promoted configurations. The local channel is
  chosen automatically as the first that passes both.

**Crystals and materials** (`engine/crystal/`, `engine/materials/`):
- `periodic.py`: DFT with Bloch waves, symmetry-reduced k-meshes, Ewald sums and
  Hellmann–Feynman forces; spin-polarised (the magnetisation is free, one Fermi level), LDA or
  PBE, with exchange–correlation on a finer grid where a sharp core density needs it (copper).
  `periodic_torch.py` runs the same thing with a single-precision eigensolver on a GPU, checked
  against the float64 path.
- `eam.py`: an embedded-atom potential whose functions are learned from this engine's own DFT
  energies and forces (MLX autodiff, then least squares), never from experiment.
- `md.py`, `experiments.py`: molecular dynamics with that potential, and the measurements a
  laboratory would make on the metal: expansion, heat capacity, melting by solid–liquid
  coexistence, latent heat. `md_torch.py` runs it on a GPU (10⁶ atoms, tested step for step
  against `md.py`).
- `dataset.py`: DFT training labels, each on its element's converged grid and functional, cached
  by content with a name that is the same on every machine.
- `continuum.py`: the centimetre block assembled from those per-atom properties.

**Truth mode** (`engine/truth/`): a neural-network wavefunction varied against the exact
many-electron Hamiltonian (variational Monte Carlo on the GPU), for the small systems where the
answer is known exactly.

**Precision.** The GPU runs in 32-bit. "Check precision" in the viewer re-solves the state in
64-bit on the CPU and reports the difference.

## Layout

```
engine/particles/  Standard Model: model, amplitudes, decays, events, protons
engine/lattice/    1D real-time QED, lattice QCD
engine/electrons/  3D quantum electrons (SCF, functionals, eigensolver)
engine/atoms/      radial atoms and pseudopotentials
engine/crystal/    periodic DFT, equations of state, elastic constants
engine/materials/  learned interatomic forces, molecular dynamics, the continuum block
engine/truth/      neural-network wavefunctions (variational Monte Carlo)
engine/nuclei/     relaxation and molecular dynamics
server/            engine threads + WebSocket streaming
viewer/            TypeScript + three.js viewer
validation/        the experiment-vs-simulation checks
tests/             pytest (fast: `uv run pytest -m "not slow"`)
```

## Roadmap

1. **Loop corrections**: one-loop QED is in (g − 2, the running of α). Next are the weak and QCD
   loops, to close the remaining percent-level gaps and add loop-induced processes such as H → γγ.
2. **Baryons on the lattice**: the pion and rho are there; the proton needs three-quark
   correlators and much more statistics. Lighter quarks and quark loops are a matter of compute.
3. **A relativistic atom solver**: scalar-relativistic radial equations, which the fourth row
   wants and anything heavier needs.
4. **More metals and alloys**: copper and iron run end to end; iron's potential needs more
   liquid-like training data (it melts ~250 K too high). Then nickel and cobalt, and alloys.
5. **Grains, defects and cracks**: the GPU molecular dynamics reaches 10⁶ atoms; the experiments
   that need that size are next.
6. **Semicore electrons in the pseudopotentials**: the next suspect for iron's possibly-large
   lattice constant (the stretched s radius has been ruled out).
