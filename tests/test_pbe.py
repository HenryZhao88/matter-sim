"""PBE, the gradient-corrected functional: the functional itself, the atom, the pseudopotential and
the crystal, each checked against something it must satisfy."""

import numpy as np
import pytest

from engine.electrons.xc import lda_xc, pbe_xc


def _densities(n=300, seed=0):
    rng = np.random.default_rng(seed)
    ru = 10 ** rng.uniform(-3, 1, n)
    rd = ru * rng.uniform(0.05, 1, n)
    suu = (ru ** (4 / 3) * rng.uniform(0, 3, n)) ** 2
    sdd = (rd ** (4 / 3) * rng.uniform(0, 3, n)) ** 2
    sud = np.sqrt(suu * sdd) * rng.uniform(-1, 1, n)
    return ru, rd, suu, sud, sdd


def test_pbe_is_lda_where_the_density_is_uniform():
    ru, rd, *_ = _densities()
    z = np.zeros_like(ru)
    e, vu, vd, *_ = pbe_xc(ru, rd, z, z, z)
    el, vlu, vld = lda_xc(ru, rd)
    assert np.abs(e - el).max() < 1e-12 * np.abs(el).max()
    assert np.abs(vu - vlu).max() < 1e-12 and np.abs(vd - vld).max() < 1e-12


@pytest.mark.parametrize("k", range(5))
def test_pbe_derivatives_are_those_of_its_energy(k):
    """Each partial derivative against a central difference of the energy density. The step is
    1e-4 relative: at 1e-5 the difference quotient's own rounding reaches 4e-4 on |∇ρ↓|² (the error
    falls as the step grows, the mark of rounding rather than of a wrong derivative)."""
    args = list(_densities())
    res = pbe_xc(*args)
    h = 1e-4 * np.abs(args[k]) + 1e-12
    up = [a.copy() for a in args]
    dn = [a.copy() for a in args]
    up[k] += h
    dn[k] -= h
    fd = (pbe_xc(*up)[0] - pbe_xc(*dn)[0]) / (2 * h)
    scale = np.abs(res[k + 1]) + 1e-9 * np.abs(res[0]) / (np.abs(args[k]) + 1e-12)
    assert np.max(np.abs(fd - res[k + 1]) / scale) < 1e-4


@pytest.mark.parametrize("Z,published", [(2, -2.892935), (4, -14.629947), (10, -128.866404)])
def test_pbe_atoms_match_published_all_electron_energies(Z, published):
    """He, Be and Ne total energies with PBE, against the published all-electron PBE values (used
    only to check). The same grid gives LDA its own reference energies to similar accuracy."""
    from engine.atoms.radial import RadialAtom
    r = RadialAtom(Z, spin=0.0, functional="pbe").solve()
    assert r.converged
    assert r.energy == pytest.approx(published, abs=2e-5)


def test_pbe_pseudopotential_reproduces_its_atom():
    """Built from the PBE atom and unscreened with PBE: the pseudo-atom gives back the reference
    eigenvalues, and passes the same ghost and transferability checks as LDA's."""
    from engine.atoms import species
    from engine.atoms.pseudo import verify
    pp = species.pseudopotential(13, "pbe")
    assert pp.functional == "pbe"
    row = verify(pp)[0]
    for l in row["eps_ae"]:
        assert row["eps_ps"][l] == pytest.approx(row["eps_ae"][l], abs=1e-6)
    assert species._checks_path(13, "pbe").exists()


def test_pbe_crystal_forces_are_the_slope_of_its_energy():
    """The gradient term enters the potential through a spectral divergence; if that were not the
    derivative of the energy, forces and energy slopes would part. Aluminium, PBE: 1e-7 Ha/bohr."""
    from engine.crystal.periodic import Crystal, PeriodicDFT, cubic
    rng = np.random.default_rng(3)
    base = cubic("fcc", 7.6, 13)
    pos = base.positions + rng.normal(0, 0.15, base.positions.shape)

    def run(p):
        return PeriodicDFT(Crystal(base.cell, base.charges, p), h=0.35, kmesh=1, symmetry=False,
                           functional="pbe").run(forces=True, tol=1e-9)
    r0 = run(pos)
    eps = 0.01
    for atom, axis in ((1, 0), (0, 2)):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, axis] += eps
        pm[atom, axis] -= eps
        slope = (run(pp).free_energy - run(pm).free_energy) / (2 * eps)
        assert r0.forces[atom, axis] == pytest.approx(-slope, abs=5e-6)


@pytest.mark.slow
def test_spin_polarised_pbe_forces_on_iron():
    """Iron, spin-polarised PBE with its partial core on the 2x XC grid: forces against the energy's
    slope (measured 7e-6 and 1.2e-5 Ha/bohr)."""
    from engine.crystal.periodic import Crystal, PeriodicDFT, cubic
    rng = np.random.default_rng(3)
    for _ in range(2):                                   # the same draws as the measurement
        rng.normal(0, 0.15, (4, 3))
    base = cubic("bcc", 5.3, 26)
    pos = base.positions + rng.normal(0, 0.15, base.positions.shape)

    def run(p):
        return PeriodicDFT(Crystal(base.cell, base.charges, p), h=0.35, kmesh=1, symmetry=False, xc_grid=2,
                           functional="pbe", spin=True, moments=2.0).run(forces=True, tol=1e-9)
    r0 = run(pos)
    assert r0.converged and abs(r0.moment) > 1.0
    eps = 0.01
    for atom, axis in ((1, 0), (0, 2)):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, axis] += eps
        pm[atom, axis] -= eps
        slope = (run(pp).free_energy - run(pm).free_energy) / (2 * eps)
        assert r0.forces[atom, axis] == pytest.approx(-slope, abs=1e-4)


def test_pbe_hydrogen_on_the_3d_grid_approaches_the_radial_atom():
    """The molecular solver's PBE (spectral gradients) against the radial solver's (finite differences
    on a log grid): the gap at h = 0.3 is the grid's, the same as LDA's (measured 2.24e-3 vs 2.02e-3)."""
    from engine.atoms.radial import RadialAtom
    from engine.core.backend import get_backend
    from engine.core.grid import Grid
    from engine.electrons.scf import SCFSolver
    from engine.system import System
    g = Grid(14.0, 0.3, get_backend("numpy"))
    gap = {}
    for f in ("lda", "pbe"):
        r = SCFSolver(g, System([1], [[0, 0, 0]]), functional=f).run()
        assert r.converged
        gap[f] = r.energy - RadialAtom(1, spin=1, functional=f).solve().energy
    assert 0 < gap["pbe"] < 3e-3
    assert abs(gap["pbe"] - gap["lda"]) < 5e-4


@pytest.mark.slow
def test_pbe_molecule_forces_are_the_slope_of_its_energy():
    """Water with PBE pseudopotentials, forces against the energy's slope (measured 3.4e-5 Ha/bohr;
    LDA at the same settings 3.9e-5)."""
    from engine.core.backend import get_backend
    from engine.core.grid import Grid
    from engine.electrons.scf import SCFSolver
    from engine.system import System
    g = Grid(12.0, 0.25, get_backend("numpy"))

    def run(d):
        s = SCFSolver(g, System([8, 1, 1], [[0, 0, 0], [d, 0.2, 0], [-0.5, 1.7, 0]]), functional="pbe")
        return s.run(max_iter=120, tol_rho=1e-5, tol_e=1e-9), s
    r, s = run(1.8)
    F = s.forces()
    eps = 0.01
    slope = (run(1.8 + eps)[0].free_energy - run(1.8 - eps)[0].free_energy) / (2 * eps)
    assert F[1, 0] == pytest.approx(-slope, abs=1e-4)
