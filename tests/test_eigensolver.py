import numpy as np
import pytest

from conftest import BACKENDS
from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.electrons.eigensolver import lobpcg, teter_preconditioner


@pytest.mark.parametrize("backend,tol", [(b, 1e-4 if b == "numpy" else 2e-3) for b in BACKENDS])
def test_harmonic_oscillator_levels(backend, tol):
    """V = r²/2 has exact levels 1.5, 2.5 (×3), 3.5 (×6)."""
    g = Grid(L=12.0, h=0.3, backend=get_backend(backend))
    b = g.backend
    r = g.r_from((0, 0, 0))
    V = b.asarray(0.5 * r ** 2)

    def H(X):
        return g.kinetic(X) + V * X

    rng = np.random.default_rng(1)
    X0 = b.asarray(rng.standard_normal((6,) + g.shape) * np.exp(-r ** 2 / 4))
    lam, X, _ = lobpcg(b, H, X0, teter_preconditioner(g), tol=1e-4, maxiter=80, n_check=4)
    np.testing.assert_allclose(lam[:4], [1.5, 2.5, 2.5, 2.5], atol=tol)
