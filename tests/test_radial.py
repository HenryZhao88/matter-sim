"""All-electron atoms: the periodic table's structure must emerge, not be put in."""

import numpy as np
import pytest

from engine.atoms.radial import RadialGrid, ground_state, radial_eigenstates
from engine.core.units import HARTREE_EV


def test_hydrogen_spectrum_is_exact():
    g = RadialGrid()
    for l in range(3):
        e, _ = radial_eigenstates(g, -1 / g.r, l, 3, 1.0)
        n = np.arange(l + 1, l + 4)
        np.testing.assert_allclose(e, -0.5 / n ** 2, atol=2e-6)


# NIST atomic reference data (LSD). NIST uses VWN correlation; this engine uses
# PW92, which differs by up to ~3 mHa for these atoms.
NIST_LSD = {1: -0.478671, 2: -2.834836, 3: -7.343957, 4: -14.447209, 8: -74.527013}


@pytest.mark.parametrize("Z", sorted(NIST_LSD))
def test_total_energy_matches_nist(Z):
    assert ground_state(Z).energy == pytest.approx(NIST_LSD[Z], abs=3e-3)


@pytest.mark.parametrize("Z,multiplicity,config", [
    (5, 2, "1s² 2s² 2p¹"),
    (6, 3, "1s² 2s² 2p²"),
    (7, 4, "1s² 2s² 2p³"),
    (8, 3, "1s² 2s² 2p⁴"),
    (11, 2, "1s² 2s² 2p⁶ 3s¹"),
])
def test_hunds_rule_and_filling_order_emerge(Z, multiplicity, config):
    a = ground_state(Z)
    assert a.multiplicity == multiplicity
    assert a.configuration() == config


@pytest.mark.slow
def test_ionization_energy_pattern_across_period_two():
    ie = {Z: (ground_state(Z, 1).energy - ground_state(Z).energy) * HARTREE_EV for Z in range(2, 12)}
    ie[1] = -ground_state(1).energy * HARTREE_EV
    assert ie[2] > ie[1] and ie[2] > ie[3]          # helium: closed shell
    assert ie[10] == max(ie[z] for z in range(3, 12))  # neon: closed shell
    assert ie[3] < ie[4] and ie[11] < ie[10]          # new shell starts
    assert ie[5] < ie[4]                              # boron: first p electron
    assert ie[8] < ie[7]                              # oxygen: first paired p electron
