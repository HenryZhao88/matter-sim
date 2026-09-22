# matter-sim

A first-principles simulator of matter. The only physics written into the code is:

- the Schrödinger equation for electrons and nuclei (kinetic energy + Coulomb's law),
- Pauli exclusion (electrons as antisymmetric fermions),
- measured inputs: fundamental constants, each nucleus's charge and mass.

Everything else has to **emerge**: orbitals, shell structure, Hund's rule, chemical bonds,
molecular shapes, magnetism, spectra. Nothing in the code names a bond, an orbital shape, a
bond angle, or a filling rule.

It runs on Apple Silicon, using the Metal GPU through [MLX](https://github.com/ml-explore/mlx),
and shows the simulation live in a browser.

```bash
uv run matter-sim            # opens the viewer at http://127.0.0.1:8765
uv run matter-sim validate   # recomputes every result below from scratch
```

First run only: `cd viewer && npm install && npm run build`.

## What emerges

All numbers come from `uv run matter-sim validate` on an M4. "Method" is what this level of
theory (LDA) is known to give; the gap to experiment belongs to the approximation, not the code.

| Result | Simulated | Experiment | Notes |
|---|---|---|---|
| Hydrogen Lyman-α / Balmer-α | 121.5 / 656.1 nm | 121.6 / 656.3 nm | exact Hamiltonian, infinite nuclear mass |
| Ionisation energies H–Ar | within ~0.6 eV | NIST | peaks at He, Ne, Ar; dips at Li, B, O, Na, Al, S |
| Filling order and Hund's rule | C, N, O: 2, 3, 2 unpaired | same | found by minimising energy |
| H₂ bond | 0.77 Å | 0.741 Å | forms spontaneously from two atoms |
| H₂ with parallel spins | repels | unbound | Pauli exclusion, never coded |
| H₃⁺ | equilateral triangle | equilateral | started as a bent chain |
| Water | 104.6°, 0.972 Å | 104.5°, 0.957 Å | started at 150° |
| O₂ | magnetic (2 unpaired spins) | paramagnetic | started with every spin paired |

## How it works

Matter is modelled as a ladder of scales. Each rung is derived from the one below it, not
hard-coded:

| Rung | What's simulated | Status |
|---|---|---|
| Quarks and nuclei | nuclear structure | Not simulated: nuclei enter as measured charge and mass, which is all chemistry needs |
| **Electrons and nuclei** | quantum electrons, moving nuclei | **This release** |
| Materials | thousands to millions of atoms | Next: forces learned from rung-2 calculations |
| Everyday matter | a cubic centimetre of aluminium (6×10²² atoms) | The long goal |

**Electrons** (`engine/electrons/`) are solved on a uniform 3D grid with spectral (FFT)
operators. That makes the kinetic energy exact for the grid, and open-boundary Coulomb
interactions exact. The electron–electron term uses one of:

- **LDA**: spin-polarised density functional theory. Exchange is the exact uniform-gas result;
  correlation is the Perdew–Wang fit to quantum Monte Carlo of the uniform electron gas. No
  molecular data goes in.
- **Hartree–Fock**: exact exchange with zero fitted constants.
- **Single electron**: no electron–electron term, which is exact for one-electron systems.

**Nuclei** feel exact Hellmann–Feynman forces (checked against finite differences) and either
relax to the nearest stable shape (FIRE) or move in real time with their real masses
(Born–Oppenheimer molecular dynamics).

**Core electrons** (`engine/atoms/`) of Li–Ne are handled by pseudopotentials that the engine
**derives itself**:

1. A spherical all-electron atom solver (Numerov on a log grid) computes each atom
   essentially exactly for the model.
2. Troullier–Martins norm-conserving pseudopotentials are built from those atoms.
3. They are checked against the all-electron atom in ionised and excited configurations, and
   agree to about 0.02 eV.

H and He are simulated with bare point nuclei.

**Precision.** The GPU runs in 32-bit, because Metal has no 64-bit floats. "Check precision"
in the viewer re-solves the current state in 64-bit on the CPU and reports the difference.

**Resolution.** The grid spacing is the main approximation. Every quantity converges
systematically as it shrinks ("Draft", "Standard", "Fine"). Better hardware means finer grids
and bigger systems, with no new rules.

## Layout

```
engine/       physics (no UI): core/ grid + backends, electrons/ SCF, nuclei/ motion, atoms/ radial + pseudo
server/       engine thread + WebSocket streaming
viewer/       TypeScript + three.js viewer (cyanotype density rendering, measurement rails)
validation/   the experiment-vs-simulation checks
tests/        pytest (fast: `uv run pytest -m "not slow"`, full: `uv run pytest`)
legacy/       the original project, untouched
```

## Roadmap

1. **Truth mode**: neural-network variational Monte Carlo (FermiNet/Psiformer-style) for
   small molecules. It converges to the exact many-electron answer and removes LDA's
   approximation where compute allows.
2. **Heavier elements**: pseudopotentials beyond neon, with d-channel projectors and nonlinear
   core corrections, up to aluminium and beyond.
3. **Periodic crystals**: Bloch k-points, so aluminium's crystal structure, lattice constant and
   elastic constants can emerge.
4. **Rung 3 (materials)**: train machine-learned interatomic potentials on rung-2 data, then
   simulate 10⁴–10⁶ atoms. That's where melting, grain boundaries and everyday properties begin.
