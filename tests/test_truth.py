"""Truth mode: neural-network wavefunction from the Schrödinger equation alone."""

import numpy as np
import pytest
from conftest import requires_mlx


@requires_mlx
@pytest.mark.slow
def test_helium_variational_energy_close_to_exact():
    from engine.truth.vmc import VMC, Molecule
    m = Molecule([2.0], np.zeros((1, 3)), 1, 1)
    v = VMC(m, walkers=512, hidden=24, layers=2, dets=2)
    v.train(iters=150)
    E, err = v.evaluate(10, 5)
    assert -2.93 < E < -2.87          # exact −2.90372; Hartree–Fock −2.8617
