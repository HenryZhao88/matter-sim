"""Crystals, learned forces and the parton shower."""

import numpy as np
import pytest

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


@pytest.mark.slow
def test_aluminium_chooses_fcc():
    e = {k: PeriodicDFT(cubic(k, (115.0 * n) ** (1 / 3), 13), kmesh=8).run().energy / n
         for k, n in (("fcc", 4), ("bcc", 2), ("sc", 1))}
    assert e["fcc"] < e["bcc"] < e["sc"]


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
