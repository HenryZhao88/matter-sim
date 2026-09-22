import numpy as np
import pytest

from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.electrons.external import NuclearField, ion_ion
from engine.electrons.xc import lda_xc, pw92_correlation


@pytest.mark.parametrize("rs,expected", [(1.0, -0.0598), (2.0, -0.0448), (5.0, -0.0282), (10.0, -0.0186)])
def test_pw92_unpolarized_matches_published_values(rs, expected):
    ec, _, _ = pw92_correlation(np.array([rs]), np.array([0.0]))
    assert ec[0] == pytest.approx(expected, abs=3e-4)


def test_exchange_unpolarized_formula():
    rs = 2.0
    rho = 3 / (4 * np.pi * rs ** 3)
    e, _, _ = lda_xc(np.array([rho / 2]), np.array([rho / 2]))
    ec, _, _ = pw92_correlation(np.array([rs]), np.array([0.0]))
    ex_per_electron = e[0] / rho - ec[0]
    assert ex_per_electron == pytest.approx(-0.458165 / rs, rel=1e-5)


@pytest.mark.parametrize("ru,rd", [(0.05, 0.02), (0.3, 0.0001), (0.001, 0.0008), (0.2, 0.2)])
def test_xc_potential_is_derivative_of_energy(ru, rd):
    eps = 1e-7
    e0, vu, vd = lda_xc(np.array([ru]), np.array([rd]))
    eu, _, _ = lda_xc(np.array([ru + eps]), np.array([rd]))
    ed, _, _ = lda_xc(np.array([ru]), np.array([rd + eps]))
    assert vu[0] == pytest.approx((eu[0] - e0[0]) / eps, rel=1e-4)
    assert vd[0] == pytest.approx((ed[0] - e0[0]) / eps, rel=1e-4)


def test_ion_ion_forces_are_minus_gradient():
    Z = [1, 8, 1]
    P = np.array([[1.4, 1.1, 0.0], [0.0, 0.0, 0.0], [-1.4, 1.1, 0.2]])
    _, F = ion_ion(Z, P)
    eps = 1e-6
    for i in range(3):
        for a in range(3):
            Pp = P.copy(); Pp[i, a] += eps
            Pm = P.copy(); Pm[i, a] -= eps
            fd = -(ion_ion(Z, Pp)[0] - ion_ion(Z, Pm)[0]) / (2 * eps)
            assert F[i, a] == pytest.approx(fd, rel=1e-5, abs=1e-8)


def test_point_nucleus_potential_is_coulomb_away_from_nucleus():
    g = Grid(L=16.0, h=0.2, backend=get_backend("numpy"))
    R = (0.13, -0.07, 0.05)  # deliberately off-grid
    v = NuclearField(g).potential([3], [R])
    r = g.r_from(R)
    # Pointwise the band-limited potential rings near the grid cutoff; its
    # average over a shell must still be exactly Coulomb.
    for lo, hi in [(1.0, 2.0), (2.0, 4.0)]:
        shell = (r > lo) & (r < hi)
        assert abs(np.mean(v[shell] + 3 / r[shell])) < 4e-3


def test_nuclear_force_matches_finite_difference_of_energy():
    g = Grid(L=10.0, h=0.25, backend=get_backend("numpy"))
    nf = NuclearField(g)
    r = g.r_from((0.4, 0.0, 0.0))
    rho = np.exp(-r ** 2)  # fixed density
    R = np.array([[0.1, 0.2, -0.1]])
    F = nf.forces([3], R, rho)
    eps = 1e-4
    for a in range(3):
        Rp = R.copy(); Rp[0, a] += eps
        Rm = R.copy(); Rm[0, a] -= eps
        ep = np.sum(nf.potential([3], Rp) * rho) * g.dV
        em = np.sum(nf.potential([3], Rm) * rho) * g.dV
        assert F[0, a] == pytest.approx(-(ep - em) / (2 * eps), rel=1e-4, abs=1e-7)
