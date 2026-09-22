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


def test_wilson_dirac_is_gamma5_hermitian_and_gpu_matches_cpu():
    from engine.lattice.hadrons import GAMMA5, WilsonDirac, WilsonDiracGPU
    g = GaugeField(2, 4, 5.7, seed=3, hot=True)
    D = WilsonDirac(g, 0.15)
    rng = np.random.default_rng(0)
    a = rng.normal(size=D.shape + (4, 3)) + 1j * rng.normal(size=D.shape + (4, 3))
    b = rng.normal(size=D.shape + (4, 3)) + 1j * rng.normal(size=D.shape + (4, 3))
    # <a, D b> = <D† a, b> with D† = γ5 D γ5
    assert abs(np.vdot(a, D.apply(b)) - np.vdot(D.apply_dag(a), b)) < 1e-10 * np.abs(a).sum()
    G = WilsonDiracGPU(g, 0.15)
    x = np.zeros(D.shape + (3, 4, 1), np.complex64)
    x[..., 0] = np.swapaxes(b, -1, -2)
    y = np.array(G.apply(G.mx.array(x)))[..., 0]
    assert np.abs(np.swapaxes(y, -1, -2) - D.apply(b)).max() < 1e-4


@pytest.mark.slow
def test_pion_is_lighter_than_rho():
    from engine.lattice.hadrons import spectrum
    r = spectrum(beta=5.7, L=4, T=8, kappas=(0.150, 0.155, 0.158), n_configs=2, therm=20, spacing=5)
    assert all(p < q for p, q in zip(r["pion"], r["rho"]))
    assert r["pion"][0] > r["pion"][-1]                      # lighter quarks → lighter pion


def test_wilson_dirac_is_gamma5_hermitian_and_gpu_matches_cpu():
    from engine.lattice.hadrons import WilsonDirac, WilsonDiracGPU
    g = GaugeField(2, 4, 5.7, seed=3, hot=True)
    D = WilsonDirac(g, 0.15)
    rng = np.random.default_rng(0)
    a = rng.normal(size=D.shape + (4, 3)) + 1j * rng.normal(size=D.shape + (4, 3))
    b = rng.normal(size=D.shape + (4, 3)) + 1j * rng.normal(size=D.shape + (4, 3))
    # <a, D b> = <D† a, b> with D† = γ5 D γ5
    assert abs(np.vdot(a, D.apply(b)) - np.vdot(D.apply_dag(a), b)) < 1e-10 * np.abs(a).sum()
    G = WilsonDiracGPU(g, 0.15)
    x = np.zeros(D.shape + (3, 4, 1), np.complex64)
    x[..., 0] = np.swapaxes(b, -1, -2)
    y = np.array(G.apply(G.mx.array(x)))[..., 0]
    assert np.abs(np.swapaxes(y, -1, -2) - D.apply(b)).max() < 1e-4


@pytest.mark.slow
def test_pion_is_lighter_than_rho():
    from engine.lattice.hadrons import spectrum
    r = spectrum(beta=5.7, L=4, T=8, kappas=(0.150, 0.155, 0.158), n_configs=2, therm=20, spacing=5)
    assert all(p < q for p, q in zip(r["pion"], r["rho"]))
    assert r["pion"][0] > r["pion"][-1]                      # lighter quarks → lighter pion


def test_flying_pair_string_breaks_into_neutral_mesons():
    """Hadronisation in 1D: the charges flying apart are screened by pairs from the vacuum."""
    model, it = run_scenario("jet", N=14, mass=0.25, t_max=6.0, frames=7, strength=0.6)
    rows = [obs for _, obs in it]
    sep = [obs["charge"][: model.N // 2].sum() for obs in rows]
    assert sep[0] > 0.9                           # one unit of charge on each side at the start
    assert min(sep[3:]) < 0.5 * sep[0]           # string broke: each half is now nearly neutral
