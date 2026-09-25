"""Crystals, learned forces and the parton shower."""

import numpy as np
import pytest
from conftest import requires_torch

from engine.crystal.periodic import Crystal, PeriodicDFT, crystal_symmetries, cubic, ewald


def test_ewald_reproduces_the_madelung_constant():
    E, F = ewald(cubic("sc", 1.0, 13))                 # Z_val = 3 point charges, simple cubic
    assert E == pytest.approx(-1.76012 * 9 / (2 * (3 / (4 * np.pi)) ** (1 / 3)), rel=1e-5)
    assert np.abs(F).max() < 1e-10


def test_same_crystal_in_two_cells_has_the_same_ion_energy():
    a = 7.5
    L = np.array([a / np.sqrt(2), a / np.sqrt(2), a])
    rotated = Crystal(L, [13, 13], np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * L)
    assert ewald(cubic("fcc", a, 13))[0] / 4 == pytest.approx(ewald(rotated)[0] / 2, abs=1e-10)


def test_fcc_has_full_cubic_symmetry():
    assert len(crystal_symmetries(cubic("fcc", 7.5, 13))) == 48


@pytest.mark.slow
def test_periodic_forces_match_energy_slope():
    rng = np.random.default_rng(3)
    base = cubic("fcc", 7.6, 13)
    pos = base.positions + rng.normal(0, 0.15, base.positions.shape)
    def run(p):
        return PeriodicDFT(Crystal(base.cell, base.charges, p), kmesh=3, symmetry=False).run(forces=True, tol=1e-8)
    r0 = run(pos)
    eps = 0.01
    for atom, axis in ((1, 0), (2, 2)):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, axis] += eps
        pm[atom, axis] -= eps
        slope = (run(pp).free_energy - run(pm).free_energy) / (2 * eps)
        assert r0.forces[atom, axis] == pytest.approx(-slope, rel=0.05, abs=2e-4)


def test_xc_grid_interpolates_exactly_and_returns_what_it_was_given():
    """The finer XC grid holds the same band-limited density, not an approximation of it."""
    s = PeriodicDFT(cubic("fcc", 7.6, 13), h=0.6, kmesh=1, xc_grid=2)
    L, N = s.c.cell[0], s.N[0]
    x = np.arange(N) * L / N
    f = 1.0 + 0.3 * np.cos(2 * np.pi * 3 * x / L)[:, None, None] * np.sin(2 * np.pi * 2 * x / L)[None, :, None] \
        + 0.1 * np.cos(2 * np.pi * 5 * x / L)[None, None, :]
    xf = np.arange(2 * N) * L / (2 * N)
    exact = 1.0 + 0.3 * np.cos(2 * np.pi * 3 * xf / L)[:, None, None] * np.sin(2 * np.pi * 2 * xf / L)[None, :, None] \
        + 0.1 * np.cos(2 * np.pi * 5 * xf / L)[None, None, :]
    assert np.abs(s._to_fine(f) - exact).max() < 1e-12
    assert np.abs(s._to_coarse(s._to_fine(f)) - f).max() < 1e-12


@pytest.mark.slow
def test_copper_forces_match_energy_slope_with_xc_on_a_finer_grid():
    """The core-correction force is taken on the XC grid, where the core density lives."""
    rng = np.random.default_rng(3)
    base = cubic("fcc", 6.83, 29)
    pos = base.positions + rng.normal(0, 0.15, base.positions.shape)
    def run(p):
        return PeriodicDFT(Crystal(base.cell, base.charges, p), h=0.35, kmesh=1, symmetry=False,
                           xc_grid=2).run(forces=True, tol=1e-8)
    r0 = run(pos)
    eps = 0.01
    for atom, axis in ((1, 0), (2, 2)):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, axis] += eps
        pm[atom, axis] -= eps
        slope = (run(pp).free_energy - run(pm).free_energy) / (2 * eps)
        assert r0.forces[atom, axis] == pytest.approx(-slope, rel=0.02, abs=5e-4)


@pytest.mark.slow
def test_xc_on_a_finer_grid_removes_copper_egg_box():
    """Sliding a perfect crystal cannot change its energy. On copper it did, by tens of meV/atom:
    the partial core density, sharp, run through the nonlinear LDA on the plain grid. Measured at
    h = 0.19, 2x2x2 k: +15.6 meV/atom (xc_grid=1) against +0.56 (xc_grid=2). Here (h = 0.26, Γ):
    5.83 against 0.135. The bound is half the 1 meV/atom a label may be off by."""
    a, H = 3.61 / 0.529177210903, 0.26
    def energy(shift, xg):
        c = cubic("fcc", a, 29)
        N = PeriodicDFT(c, h=H, kmesh=1).N[0]
        c.positions = c.positions + shift * (a / N) * np.array([1.0, 0.7, 0.3])
        return PeriodicDFT(c, h=H, kmesh=1, T_e=0.01, symmetry=False, xc_grid=xg).run().free_energy / 4 * 27211.386
    plain = abs(energy(0.5, 1) - energy(0.0, 1))
    fine = abs(energy(0.5, 2) - energy(0.0, 2))
    assert plain > 20 * fine
    assert fine < 0.5


@pytest.mark.slow
def test_aluminium_chooses_fcc():
    e = {k: PeriodicDFT(cubic(k, (115.0 * n) ** (1 / 3), 13), kmesh=8).run().energy / n
         for k, n in (("fcc", 4), ("bcc", 2), ("sc", 1))}
    assert e["fcc"] < e["bcc"] < e["sc"]


def test_antiferromagnetic_order_lowers_the_symmetry():
    """Layered antiferromagnetic fcc (up planes and down planes alternating along z) is tetragonal:
    of the cube's 48 operations only the 16 that keep z as z map up-atoms onto up-atoms."""
    c = cubic("fcc", 6.8, 26)
    up_down = [(26, m) for m in (1, -1, -1, 1)]          # atoms at z = 0, ½, ½, 0
    assert len(crystal_symmetries(c, labels=up_down)) == 16


def test_spin_polarised_dft_without_a_starting_moment_is_the_unpolarised_one():
    """Both spins start from the same state, so they stay equal exactly, and every spin-resolved
    piece (occupations, energy terms, mixing) must add back up to the unpolarised calculation."""
    c = cubic("fcc", 7.6, 13)
    kw = dict(h=0.45, kmesh=2, T_e=0.01)
    ref = PeriodicDFT(c, **kw).run()
    got = PeriodicDFT(c, spin=True, **kw).run()
    assert got.converged and got.moment == 0 and got.abs_moment == 0
    assert got.free_energy == pytest.approx(ref.free_energy, abs=1e-9)


@pytest.mark.slow
def test_spin_polarised_dft_with_a_partial_core_on_a_fine_xc_grid():
    """Copper: the partial core counts half to each spin, on the finer XC grid; with no starting
    moment that must add back up to the unpolarised energy and forces."""
    rng = np.random.default_rng(7)
    base = cubic("fcc", 6.83, 29)
    c = Crystal(base.cell, base.charges, base.positions + rng.normal(0, 0.15, base.positions.shape))
    kw = dict(h=0.5, kmesh=1, T_e=0.01, symmetry=False, xc_grid=2)
    ref = PeriodicDFT(c, **kw).run(forces=True)
    got = PeriodicDFT(c, spin=True, **kw).run(forces=True)
    assert got.converged and got.moment == 0
    assert got.free_energy == pytest.approx(ref.free_energy, abs=1e-8)
    assert np.abs(got.forces - ref.forces).max() < 1e-6


@pytest.mark.slow
def test_aluminium_does_not_stay_magnetic():
    """Pushed to 1 Bohr magneton per atom, aluminium's electrons give the moment back: nothing in the
    code says which metals are magnets, so this has to come out of exchange against band energy."""
    c = cubic("fcc", 7.6, 13)
    kw = dict(h=0.4, kmesh=4, T_e=0.01)
    ref = PeriodicDFT(c, **kw).run()
    got = PeriodicDFT(c, spin=True, moments=1.0, **kw).run()
    assert got.converged
    assert got.abs_moment / 4 < 1e-3
    assert got.free_energy == pytest.approx(ref.free_energy, abs=1e-7)


@pytest.mark.slow
def test_spin_polarised_forces_match_energy_slope():
    """Magnetic iron, displaced: the force includes each spin's nonlocal term and the partial core's
    share of both spins' exchange-correlation potential."""
    rng = np.random.default_rng(5)
    base = cubic("bcc", 5.3, 26)
    pos = base.positions + rng.normal(0, 0.15, base.positions.shape)

    def run(p):
        return PeriodicDFT(Crystal(base.cell, base.charges, p), h=0.35, kmesh=2, T_e=0.01, symmetry=False,
                           spin=True, moments=2.5).run(forces=True, tol=1e-8)
    r0 = run(pos)
    assert r0.converged and r0.abs_moment / 2 > 1.0
    eps = 0.01
    for atom, axis in ((0, 0), (1, 2)):
        pp, pm = pos.copy(), pos.copy()
        pp[atom, axis] += eps
        pm[atom, axis] -= eps
        slope = (run(pp).free_energy - run(pm).free_energy) / (2 * eps)
        assert r0.forces[atom, axis] == pytest.approx(-slope, rel=0.05, abs=2e-4)


def test_colour_factors_come_from_the_group():
    from engine.particles.shower import group_constants
    g = group_constants()
    assert g["CF"] == pytest.approx(4 / 3) and g["CA"] == pytest.approx(3) and g["TR"] == pytest.approx(0.5)


def test_gluon_jets_radiate_more_than_quark_jets():
    from engine.particles.shower import shower
    rng = np.random.default_rng(1)
    ng = np.mean([len(shower("g", np.array([200.0, 0, 0, 200]), rng)) for _ in range(200)])
    nq = np.mean([len(shower("u", np.array([200.0, 0, 0, 200]), rng)) for _ in range(200)])
    assert ng > 1.3 * nq


@requires_torch
@pytest.mark.parametrize("wide", [None, "cpu"])
def test_single_precision_gpu_dft_matches_numpy(wide):
    """The complex64 device path (periodic_torch.py) against the NumPy reference on a real metal:
    energy within 0.1 meV/atom and forces within 5e-6 Ha/bohr, far inside the budget it has to
    meet (1 meV/atom, 1e-3 Ha/bohr). ``wide="cpu"`` keeps the float64 work on the CPU,
    as on Apple's MPS, which has no float64."""
    from engine.crystal.periodic_torch import PeriodicDFTTorch
    rng = np.random.default_rng(5)
    base = cubic("bcc", 6.1, 13)
    c = Crystal(base.cell, base.charges, base.positions + rng.normal(0, 0.2, base.positions.shape))
    kw = dict(h=0.45, kmesh=2, T_e=0.01, symmetry=False)       # coarse: a comparison, not a converged answer
    ref = PeriodicDFT(c, **kw).run(forces=True)
    got = PeriodicDFTTorch(c, wide_device=wide, **kw).run(forces=True)
    assert ref.converged and got.converged
    assert abs(got.free_energy - ref.free_energy) / 2 * 27.211386 < 1e-4
    # forces are first order in the eigenvectors' error, energies second: a solver floor 30x too
    # loose left 3.8e-5 here (1.4e-4 on a production label) while energies still agreed
    assert np.abs(got.forces - ref.forces).max() < 5e-6
    assert np.abs(ref.forces).max() > 1e-3                 # the displacement produces real forces


@requires_torch
def test_single_precision_scf_converges_on_degenerate_bands():
    """Perfect fcc Al on one k-point: heavily degenerate bands, where the complex64 eigensolver once
    fed its own rounding noise back as search directions, produced eigenvalues of −91 Ha, and left
    the SCF unconverged after 60 iterations, 4 meV/atom off. It must converge, and agree."""
    from engine.crystal.periodic_torch import PeriodicDFTTorch
    c = cubic("fcc", 7.6, 13)
    ref = PeriodicDFT(c, h=0.4, kmesh=2).run()
    got = PeriodicDFTTorch(c, h=0.4, kmesh=2).run()
    assert ref.converged and got.converged
    assert abs(got.free_energy - ref.free_energy) / 4 * 27.211386 < 0.1


@requires_torch
def test_single_precision_eigensolver_is_stable_far_past_convergence():
    """Run LOBPCG 200 iterations on a converged problem: the eigenvalues must stay on the float64 ones."""
    import torch
    from engine.crystal.periodic_torch import PeriodicDFTTorch
    from engine.electrons.xc import lda_xc
    c = cubic("fcc", 7.6, 13)
    n, t = PeriodicDFT(c, h=0.4, kmesh=2), PeriodicDFTTorch(c, h=0.4, kmesh=2)
    rho = np.full(n.N, c.valence / c.volume)
    rx = rho + n.rho_core
    Veff = n.Vloc + n._hartree(rho) + lda_xc(rx / 2, rx / 2)[1]
    k = n.kpts[0]
    B, _, E, _ = n._projectors(k)
    rng = np.random.default_rng(0)
    U0 = rng.standard_normal((n.n_bands, n.Ntot)) + 1j * rng.standard_normal((n.n_bands, n.Ntot))
    ref, _ = n._eig(k, Veff, B, E, U0.copy(), 60)
    Bt, Et, _ = t._projectors_dev(0, k)
    V32 = torch.tensor(Veff.reshape(1, -1).astype(np.float32), device=t.dev)
    got, _ = t._eig_dev(k, V32, Bt, Et, torch.tensor(U0.astype(np.complex64), device=t.dev), 200)
    assert np.abs(got - ref).max() < 1e-5


@requires_torch
@pytest.mark.slow
@pytest.mark.parametrize("xc_grid", [1, 2])
def test_single_precision_dft_on_a_transition_metal(xc_grid):
    """Copper: d projectors, a p-channel Kleinman–Bylander energy of 469 Ha, and a nonlocal energy that
    is most of the total. The eigensolver floor once scaled with the projector term's operator norm
    (3e4 Ha here), locked the bands 500x too early and left forces 2.9e-3 Ha/bohr off (3.4e-5 now).
    The energy agrees to ~0.1-0.4 meV/atom, not aluminium's 0.001: single precision's relative error
    times copper's much larger energy terms."""
    from engine.crystal.periodic_torch import PeriodicDFTTorch
    rng = np.random.default_rng(7)
    base = cubic("fcc", 6.83, 29)
    c = Crystal(base.cell, base.charges, base.positions + rng.normal(0, 0.15, base.positions.shape))
    kw = dict(h=0.45, kmesh=1, T_e=0.01, symmetry=False, xc_grid=xc_grid)
    ref = PeriodicDFT(c, **kw).run(forces=True)
    got = PeriodicDFTTorch(c, **kw).run(forces=True)
    assert ref.converged and got.converged
    assert abs(got.free_energy - ref.free_energy) / 4 * 27.211386 < 1e-3
    assert np.abs(got.forces - ref.forces).max() < 3e-4
