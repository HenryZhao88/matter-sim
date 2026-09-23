"""Truth mode: neural-network wavefunction from the Schrödinger equation alone."""

import numpy as np
import pytest
from conftest import requires_autodiff, requires_torch


@requires_autodiff
@pytest.mark.slow
def test_helium_variational_energy_close_to_exact():
    from engine.truth.vmc import VMC, Molecule
    m = Molecule([2.0], np.zeros((1, 3)), 1, 1)
    v = VMC(m, walkers=512, hidden=24, layers=2, dets=2)
    v.train(iters=150)
    E, err = v.evaluate(10, 5)
    assert -2.93 < E < -2.87          # exact −2.90372; Hartree–Fock −2.8617


@requires_torch
def test_torch_local_energy_matches_finite_differences():
    """The torch port's autodiff kinetic energy against float64 central differences of log ψ."""
    import torch
    from engine.truth.vmc_torch import VMC
    from engine.truth.molecule import Molecule
    mol = Molecule([1.0, 1.0], np.array([[0, 0, -0.7], [0, 0, 0.7]]), 1, 1)
    v = VMC(mol, walkers=3, hidden=16, layers=2, dets=2, device="cpu")
    kin = (v.local_energy(v.r) - v._potential(v.r)).numpy()
    net = v.net.double()
    with torch.no_grad():
        for w in range(3):
            x = v.r[w].double().reshape(-1)
            f = lambda y: float(net(y.reshape(-1, 3)))
            h, lap, g2 = 1e-4, 0.0, 0.0
            for k in range(x.numel()):
                e = torch.zeros_like(x)
                e[k] = h
                a, b, c = f(x + e), f(x), f(x - e)
                lap += (a - 2 * b + c) / h**2
                g2 += ((a - c) / (2 * h)) ** 2
            assert abs(kin[w] - (-0.5 * (lap + g2))) < 1e-3
