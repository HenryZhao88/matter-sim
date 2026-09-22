import numpy as np
import pytest

from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.electrons.scf import SCFSolver
from engine.simulation import Params, Simulation
from engine.system import System


def h2(R):
    return System([1, 1], [[-R / 2, 0, 0], [R / 2, 0, 0]])


@pytest.mark.slow
def test_h2_force_is_minus_energy_slope():
    """Hellmann–Feynman forces must agree with finite differences of E(R)."""
    g = Grid(14.0, 0.2, get_backend("numpy"))

    def energy_and_force(R):
        s = SCFSolver(g, h2(R))
        res = s.run(tol_rho=1e-5, tol_e=1e-8)
        return res.free_energy, s.forces()

    R, dR = 1.6, 0.04
    Ep, _ = energy_and_force(R + dR)
    Em, _ = energy_and_force(R - dR)
    _, F = energy_and_force(R)
    slope = (Ep - Em) / (2 * dR)
    # Atom 1 sits at +R/2: moving it by δ changes R by δ, so F_x = -dE/dR.
    assert F[1, 0] == pytest.approx(-slope, rel=0.05, abs=2e-3)
    np.testing.assert_allclose(F[0], -F[1], atol=2e-3)  # Newton's third law emerges


@pytest.mark.slow
def test_h2_relaxes_to_a_bond():
    sim = Simulation(h2(2.2), Params(quality="standard", mode="relax"))
    for _ in range(60):
        sim.step()
        if sim.relaxed:
            break
    P = sim.system.positions
    R = float(np.linalg.norm(P[0] - P[1]))
    assert sim.relaxed
    assert 1.38 < R < 1.50  # LDA: 1.45 bohr; experiment 1.401


def test_dynamics_conserves_energy_roughly():
    sim = Simulation(h2(1.6), Params(quality="draft", mode="dynamics", dt=10.0))
    for _ in range(12):
        sim.step()
    trace = np.array(sim.energy_trace[1:])
    assert np.ptp(trace) < 2e-3
