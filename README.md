# matter-sim

A first-principles simulator of matter, from particle collisions up to molecules. The only
physics written into the code is the laws themselves:

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
- chemical bonds, molecular shapes and magnetism.

It runs on Apple Silicon (Metal GPU through [MLX](https://github.com/ml-explore/mlx)) and
shows everything live in a browser.

```bash
uv run matter-sim                           # opens the viewer at http://127.0.0.1:8765
uv run matter-sim validate                  # recomputes every result below (~1.5 h)
uv run matter-sim validate --part particles # just the particle rung (~1 min)
```

The first launch builds the viewer (needs Node.js) and derives the decay tables (a few minutes,
once). The first proton collision fetches the proton's measured structure (0.5 MB) and computes
870 parton–parton tables on every core (about 30 minutes on an M4, once). After that, everything
is cached.

## The ladder of scales

The scale bar at the top of the viewer switches between the rungs.

| Rung | What is simulated | How |
|---|---|---|
| **Particles and forces** | collisions, decays, confinement | Standard Model amplitudes; lattice gauge theory |
| **Electrons and nuclei** | atoms and molecules | quantum electrons (DFT/Hartree–Fock), moving nuclei |
| **Materials** | hundreds of atoms in motion | forces learned from this project's own DFT, then molecular dynamics |
| **Everyday matter** | a cubic centimetre of aluminium | a continuum built from the per-atom properties measured above |

## What emerges: particles and forces

All numbers are tree-level (lowest order). The gaps to experiment are the known size of the
loop corrections that level leaves out.

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
- **Tree level only.** Amplitudes have no loop corrections yet. H → gg and H → γγ are loop
  processes, so the Higgs's branching ratios are two-body tree level only. The strong coupling
  does run with the collision energy (one loop, with the quark thresholds read off the model), so
  the leading effect of loops on rates is there.

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
LDA, water is about 105° and H₂ about 0.765 Å.

**Truth mode** removes that approximation for small systems: a neural-network wavefunction
(FermiNet-style, trained on the GPU) is varied against the exact many-electron Hamiltonian, with
the electron–electron cusps built in and antisymmetry from determinants. It is slower by orders
of magnitude and limited to a handful of electrons, so it serves as the reference the cheap
methods are measured against.

**Which elements are available.** Hydrogen and helium are simulated all-electron. Lithium to
krypton use pseudopotentials the engine derives from its own all-electron atoms, each one
checked for ghost states and for transferability across ionised and promoted configurations
before it is offered. Everything up to argon agrees to within 0.06 eV; the fourth row is
looser (4 meV for zinc, 147 meV for titanium) and scandium currently fails its own check at
212 meV, so it stays greyed out rather than quietly giving wrong answers. Relativity is not in
the atom solver yet, which matters most for the heaviest of these.

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
  correlation fit to quantum Monte Carlo of the uniform electron gas), Hartree–Fock, or a
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
  Hellmann–Feynman forces.
- `eam.py`: an embedded-atom potential whose functions are learned from this engine's own DFT
  energies and forces (MLX autodiff, then least squares), never from experiment.
- `md.py`, `experiments.py`: molecular dynamics with that potential, and the measurements a
  laboratory would make on the metal: expansion, heat capacity, melting by solid–liquid
  coexistence, latent heat.
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

1. **Loop corrections**: one-loop amplitudes, to close the remaining percent-level gaps and add
   loop-induced processes such as H → γγ. The running of α_s is already in; the amplitudes are not.
2. **Baryons on the lattice**: the pion and rho are there; the proton needs three-quark
   correlators and much more statistics. Lighter quarks and quark loops are a matter of compute.
3. **A relativistic atom solver**: scalar-relativistic radial equations, which the fourth row
   wants and anything heavier needs.
4. **More metals and alloys**: the learned-forces pipeline is aluminium-specific only in its
   training set; other elements need their own labelled configurations.
5. **Bigger dynamics**: the molecular dynamics runs a few hundred atoms in Python. On the GPU,
   with neighbour lists in MLX, 10⁵–10⁶ atoms is reachable, which is where grains, defects and
   cracks start to appear.
