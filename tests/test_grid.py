import numpy as np
import pytest
from scipy.special import erf

from engine.core.backend import get_backend
from engine.core.grid import Grid, fft_friendly


def test_fft_friendly_sizes():
    assert fft_friendly(7) == 8
    assert fft_friendly(62) == 64
    assert fft_friendly(71) == 72
    assert fft_friendly(97) == 100


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_kinetic_of_plane_wave_is_half_k_squared(backend):
    g = Grid(L=10.0, h=0.25, backend=get_backend(backend))
    X, Y, Z = g.coords
    k = 2 * np.pi * 3 / g.L
    psi = np.sin(k * X) * np.cos(2 * k * Y)
    t = g.backend.to_numpy(g.kinetic(g.backend.asarray(psi)))
    expected = 0.5 * (k ** 2 + (2 * k) ** 2) * psi
    assert np.max(np.abs(t - expected)) < 1e-4


@pytest.mark.parametrize("backend", ["numpy", "mlx"])
def test_hartree_of_gaussian_matches_analytic_open_boundary(backend):
    g = Grid(L=16.0, h=0.2, backend=get_backend(backend))
    s = 0.8
    r = g.r_from((0.3, -0.2, 0.1))
    rho = np.exp(-(r ** 2) / (2 * s ** 2)) / (2 * np.pi * s ** 2) ** 1.5
    v = g.backend.to_numpy(g.hartree(g.backend.asarray(rho)))
    with np.errstate(invalid="ignore", divide="ignore"):
        exact = np.where(r > 1e-12, erf(r / (np.sqrt(2) * s)) / r, np.sqrt(2 / np.pi) / s)
    inside = r <= g.L / 4
    assert np.max(np.abs(v - exact)[inside]) < 1e-3
    # Open boundaries: far from the charge the potential is 1/r, not periodic-image junk.
    corner = r > 0.45 * g.L
    assert np.max(np.abs(v - exact)[corner]) < 5e-3


def test_backends_agree():
    gn = Grid(L=8.0, h=0.25, backend=get_backend("numpy"))
    gm = Grid(L=8.0, h=0.25, backend=get_backend("mlx"))
    rng = np.random.default_rng(0)
    f = rng.standard_normal(gn.shape)
    a = gn.backend.to_numpy(gn.kinetic(gn.backend.asarray(f)))
    b = gm.backend.to_numpy(gm.kinetic(gm.backend.asarray(f)))
    assert np.max(np.abs(a - b)) / np.max(np.abs(a)) < 1e-5
