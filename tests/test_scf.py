import numpy as np
import pytest

from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.electrons.occupations import fermi
from engine.electrons.scf import SCFSolver
from engine.system import System


def test_fermi_fills_degenerate_levels_evenly():
    f, _, _ = fermi(np.array([-1.0, -0.3, -0.3, -0.3, 0.5]), 2.0, 1e-3)
    np.testing.assert_allclose(f, [1, 1 / 3, 1 / 3, 1 / 3, 0], atol=1e-6)


def test_system_spin_counts():
    s = System([8], [[0, 0, 0]], multiplicity=3)  # O: 6 valence electrons (1s² is in the pseudopotential)
    assert (s.n_up, s.n_dn) == (4, 2)
    with pytest.raises(ValueError):
        System([8], [[0, 0, 0]], multiplicity=2)


def solve(Z, functional, backend="mlx", h=0.2, L=14.0, **kw):
    g = Grid(L, h, get_backend(backend))
    return SCFSolver(g, System([Z], [[0, 0, 0]], **kw), functional=functional).run()


def test_one_electron_hydrogen_is_exact():
    r = solve(1, "none")
    assert r.converged
    assert r.energy == pytest.approx(-0.5, abs=1e-3)


@pytest.mark.slow
def test_hydrogen_hartree_fock_is_exact():
    """For one electron, exact exchange cancels the self-repulsion exactly."""
    r = solve(1, "hf")
    assert r.energy == pytest.approx(-0.5, abs=1e-3)


def test_hydrogen_lsda_matches_reference():
    r = solve(1, "lda")
    assert r.converged
    assert r.energy == pytest.approx(-0.479, abs=2e-3)


def test_helium_lda_converges_to_reference_as_grid_refines():
    """Accuracy is limited only by resolution: finer grid -> closer to -2.835."""
    coarse = solve(2, "lda", h=0.2)
    fine = solve(2, "lda", h=0.15)
    assert coarse.converged and fine.converged
    assert fine.energy < coarse.energy
    assert fine.energy == pytest.approx(-2.835, abs=1.5e-2)


def test_float32_gpu_agrees_with_float64_cpu():
    a = solve(2, "lda", backend="mlx", h=0.3, L=12.0)
    b = solve(2, "lda", backend="numpy", h=0.3, L=12.0)
    assert a.energy == pytest.approx(b.energy, abs=2e-4)


def test_result_independent_of_starting_guess():
    g = Grid(12.0, 0.3, get_backend("mlx"))
    e = [SCFSolver(g, System([2], [[0, 0, 0]]), seed=s).run().energy for s in (0, 7)]
    assert e[0] == pytest.approx(e[1], abs=2e-4)
