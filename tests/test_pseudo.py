"""Pseudopotentials: derived from our own atoms, checked against them."""

import numpy as np
import pytest

from engine.atoms.pseudo import verify
from engine.atoms.radial import RadialAtom
from engine.atoms.species import pseudopotential
from conftest import DEFAULT_BACKEND
from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.electrons.scf import SCFSolver
from engine.simulation import Params, Simulation
from engine.system import System


@pytest.mark.parametrize("Z", [3, 6, 8, 10, 11, 13, 16, 18])
def test_reference_eigenvalues_are_reproduced_exactly(Z):
    pp = pseudopotential(Z)
    row = verify(pp)[0]
    for l in (0, 1):
        assert row["eps_ps"][l] == pytest.approx(row["eps_ae"][l], abs=2e-4)


@pytest.mark.parametrize("Z", [5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17])
def test_transferable_to_ionised_and_excited_configurations(Z):
    """Excitation energies of the pseudo-atom track the all-electron atom."""
    for row in verify(pseudopotential(Z))[1:]:
        assert row["dE_ps"] == pytest.approx(row["dE_ae"], abs=2e-3)  # ≈ 0.05 eV


def test_core_is_every_shell_below_the_outermost():
    assert pseudopotential(8).core == [(1, 0)]
    assert pseudopotential(13).core == [(1, 0), (2, 0), (2, 1)]
    assert pseudopotential(13).Z_val == 3


@pytest.mark.parametrize("Z,mult", [(6, 3), (8, 3)])
def test_3d_pseudo_atom_matches_radial_pseudo_atom(Z, mult):
    pp = pseudopotential(Z)
    ref = RadialAtom(Z, spin=mult - 1, grid=pp.grid, v_external=pp.v_ion, n_valence=pp.Z_val).solve()
    g = Grid(14.0, 0.3, get_backend(DEFAULT_BACKEND))
    res = SCFSolver(g, System([Z], [[0, 0, 0]], multiplicity=mult)).run()
    assert res.energy == pytest.approx(ref.energy, abs=5e-3)


@pytest.mark.slow
def test_water_bends_to_the_right_angle():
    th, d = np.radians(70.0), 1.1 / 0.529177
    P = [[0, 0, 0], [d * np.sin(th), d * np.cos(th), 0], [-d * np.sin(th), d * np.cos(th), 0]]
    sim = Simulation(System([8, 1, 1], P), Params(quality="standard", mode="relax"))
    for _ in range(80):
        sim.step()
        if sim.relaxed:
            break
    P = sim.system.positions
    a, b = P[1] - P[0], P[2] - P[0]
    angle = np.degrees(np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    assert sim.relaxed
    assert 102.5 < angle < 106.5                       # experiment 104.5°
    assert 0.95 < np.linalg.norm(a) * 0.529177 < 0.99  # experiment 0.958 Å


@pytest.mark.slow
def test_pseudo_forces_match_energy_slope():
    """Local + non-local (projector) forces equal −dE/dR for LiH."""
    g = Grid(16.0, 0.25, get_backend("numpy"))

    def run(R):
        s = SCFSolver(g, System([3, 1], [[-R / 2, 0.1, 0], [R / 2, 0.1, 0]]))
        res = s.run(tol_rho=1e-6, tol_e=1e-9)
        return res.free_energy, s.forces()

    R, dR = 3.2, 0.04
    Ep, _ = run(R + dR)
    Em, _ = run(R - dR)
    _, F = run(R)
    slope = (Ep - Em) / (2 * dR)
    assert F[1, 0] == pytest.approx(-slope, rel=0.05, abs=1e-3)
    assert F[0, 0] == pytest.approx(slope, rel=0.05, abs=1e-3)


def test_annealed_reference_atom_is_the_converged_one():
    """Manganese's 4s and 3d trade electrons for hundreds of iterations at the atom's fine
    smearing; the plain SCF settles only by chance (814 iterations on the Mac, under 400 on Linux
    and Windows), which gave different machines different pseudopotentials. Annealing the smearing
    reaches the same state every time: the energy and 4s/3d occupations the long plain run finds."""
    import math
    from engine.atoms.radial import RadialAtom, RadialGrid, settled
    grid = RadialGrid(r_min=2e-6 / math.sqrt(25), r_max=60.0, dx=0.004)
    res = settled(RadialAtom(25, spin=0.0, grid=grid))
    assert res.converged
    assert abs(res.energy - (-1148.46643671)) < 2e-8
    s4 = [2 * lv.occupation for lv in res.levels if lv.spin == 0 and lv.n == 4 and lv.l == 0][0]
    assert abs(s4 - 1.2441) < 1e-3
