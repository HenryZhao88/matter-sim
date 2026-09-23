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


@requires_torch
@pytest.mark.parametrize("wide", [None, "cpu"])
def test_single_precision_gpu_dft_matches_numpy(wide):
    """The complex64 device path (periodic_torch.py) against the NumPy reference on a real metal:
    energy within 0.1 meV/atom and forces within 1e-4 Ha/bohr, ten times tighter than the budget
    it has to meet (1 meV/atom, 1e-3 Ha/bohr). ``wide="cpu"`` keeps the float64 work on the CPU,
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
    assert np.abs(got.forces - ref.forces).max() < 1e-4
    assert np.abs(ref.forces).max() > 1e-3                 # the displacement produces real forces
