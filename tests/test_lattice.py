"""Lattice gauge theories: real-time QED in 1+1D and pure-gauge QCD in 4D."""

import numpy as np
import pytest

from engine.lattice.qcd import GaugeField, polyakov_scan, run_measurement
from engine.lattice.schwinger import SchwingerModel, run_scenario, vacuum_profile


def test_schwinger_hamiltonian_is_hermitian_and_gauss_law_holds():
    m = SchwingerModel(N=10, mass=0.3)
    assert abs(m.H - m.H.getH()).max() < 1e-12
    # zero total charge sector: the field vanishes past the last site
    last = m.background + m.q.sum(axis=1)
    assert np.allclose(last, 0)


def test_exact_evolution_conserves_energy():
    model, it = run_scenario("collision", N=12, mass=0.4, t_max=4.0, frames=9)
    E = [obs["energy"] for _, obs in it]
    assert max(E) - min(E) < 1e-10


def test_strong_field_creates_pairs_that_screen_it():
    model, it = run_scenario("pair_creation", N=12, mass=0.3, t_max=4.0, frames=21, strength=1.0)
    rows = list(it)
    first, later = rows[0][1], rows[-1][1]
    assert first["particles"] == pytest.approx(0.0, abs=1e-12)     # bare vacuum: nothing there
    assert max(r[1]["particles"] for r in rows) > 1.0              # pairs appear from nothing
    assert np.abs(later["field"]).mean() < np.abs(first["field"]).mean()   # and screen the field


def test_string_between_charges_breaks_for_light_matter():
    N, mass = 14, 0.4
    vac = vacuum_profile(N, mass)
    model, it = run_scenario("string_breaking", N=N, mass=mass, t_max=10.0, frames=21)
    rows = list(it)
    i, j = N // 4, N - 1 - N // 4
    f0 = np.abs(rows[0][1]["field"][i:j]).mean()
    f1 = np.abs(rows[-1][1]["field"][i:j]).mean()
    assert f1 < 0.8 * f0
    assert rows[-1][1]["particles"] - vac["particles"] > 0.5


def test_plaquette_matches_published_value():
    g = GaugeField(6, 6, 6.0, seed=4)
    vals = []
    for s in range(70):
        g.sweep()
        if s >= 30:
            vals.append(g.plaquette())
    assert np.mean(vals) == pytest.approx(0.5937, abs=0.004)


@pytest.mark.slow
def test_quark_potential_rises_linearly_confinement():
    res = [m for m in run_measurement(5.7, L=8, T=8, sweeps=80) if m["type"] == "qcd.result"][0]
    V = np.array(res["V"])
    assert np.all(np.diff(V) > 0)                     # always rising
    assert 0.08 < res["fit"]["sigma"] < 0.25          # a real string tension


@pytest.mark.slow
def test_deconfinement_at_high_temperature():
    (b1, p1, _), (b2, p2, _) = polyakov_scan([5.4, 6.1], L=6, Nt=4, sweeps=80)
    assert p2 > 3 * p1
