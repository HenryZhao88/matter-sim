"""The Standard Model rung: everything checked here is computed from the Lagrangian."""

import math

import numpy as np
import pytest

from engine.particles.amplitudes import Amplitude
from engine.particles.process import (_avg_factor, beams, cross_section, leg, model, two_body,
                                      width_2body)


def test_symmetry_breaking_gives_massless_photon_and_heavy_w_z():
    m = model()
    assert m.vector("photon").mass == pytest.approx(0.0, abs=1e-6)
    assert m.vector("Z").mass == pytest.approx(91.1876, abs=1e-3)       # input
    assert m.vector("W1").mass == pytest.approx(80.37, rel=0.01)        # prediction (tree level)
    assert m.vector("W1").mass == m.vector("W2").mass


@pytest.mark.parametrize("name,q", [("u", 2 / 3), ("d", -1 / 3), ("e", -1), ("nu_e", 0), ("t", 2 / 3)])
def test_electric_charges_emerge_from_isospin_and_hypercharge(name, q):
    assert model().fermion(name).charge == pytest.approx(q, abs=1e-12)


def test_qed_pair_production_matches_textbook():
    m = model()
    alpha = m.e ** 2 / (4 * math.pi)
    sigma, _, _ = cross_section("e-", "e+", "mu-", "mu+", 10.0)
    assert sigma == pytest.approx(4 * math.pi * alpha ** 2 / (3 * 100) * 0.3893793721e9, rel=2e-3)


def _m2(names, rs=100.0, ct=0.3):
    legs = [leg(names[0], True), leg(names[1], True), leg(names[2], False), leg(names[3], False)]
    A = Amplitude(model(), legs)
    p1, p2, _ = beams(rs, 0, 0)
    p3, p4, _ = two_body(rs, 0, 0, ct, 0.4)
    M2 = float(np.sum(np.abs(A.evaluate([p1, p2, p3, p4])) ** 2)) * A.config_scale / _avg_factor(names[:2])
    s, t, u = rs * rs, -rs * rs / 2 * (1 - ct), -rs * rs / 2 * (1 + ct)
    return M2, s, t, u


@pytest.mark.slow
def test_gluon_self_interaction_gg_to_gg():
    gs = model().gs
    M2, s, t, u = _m2(["g", "g", "g", "g"])
    assert M2 == pytest.approx(gs ** 4 * 9 / 2 * (3 - t * u / s ** 2 - s * u / t ** 2 - s * t / u ** 2), rel=1e-9)


def test_quark_annihilation_to_gluons():
    gs = model().gs
    M2, s, t, u = _m2(["u", "u~", "g", "g"])
    ref = gs ** 4 * ((32 / 27) * (t * t + u * u) / (t * u) - (8 / 3) * (t * t + u * u) / s ** 2)
    assert M2 == pytest.approx(ref, rel=1e-9)


def test_gauge_cancellation_keeps_ww_production_finite():
    """Individual diagrams grow like s; only the full gauge structure makes σ fall."""
    s200, _, _ = cross_section("e-", "e+", "W+", "W-", 200.0, n_cos=48)
    s1000, _, _ = cross_section("e-", "e+", "W+", "W-", 1000.0, n_cos=48)
    s3000, _, _ = cross_section("e-", "e+", "W+", "W-", 3000.0, n_cos=48)
    assert 15 < s200 < 22                  # LEP2: ≈ 17 pb (tree level slightly higher)
    assert s3000 < s1000 < s200


def test_widths_and_charge_conservation():
    assert width_2body("Z", "e-", "e+") == pytest.approx(0.0839, rel=0.02)
    assert width_2body("W+", "e+", "nu_e") == pytest.approx(0.2265, rel=0.03)
    assert width_2body("W+", "e-", "nu_e~") == 0.0           # would violate charge
    assert width_2body("t", "b", "W+") == pytest.approx(1.46, rel=0.05)


def test_muon_lifetime_emerges_from_w_exchange():
    from engine.particles.decays import branching_ratios, lifetime_seconds
    assert lifetime_seconds("mu-") == pytest.approx(2.197e-6, rel=0.03)
    (products, br), = branching_ratios("mu-")
    assert set(products) == {"e-", "nu_e~", "nu_mu"}      # lepton flavour conserved, never imposed


def test_z_invisible_width_counts_three_neutrino_families():
    from engine.particles.decays import branching_ratios
    invisible = sum(b for p, b in branching_ratios("Z") if all(x.startswith("nu") for x in p))
    assert invisible == pytest.approx(0.200, abs=0.01)


def test_top_decays_before_confinement_but_bottom_does_not():
    from engine.particles.events import is_confined
    assert not is_confined("t")
    assert is_confined("b")


@pytest.mark.slow
def test_r_ratio_counts_quark_colours():
    from engine.particles.events import outcomes
    tab = outcomes("e-", "e+", 10.0)
    had = sum(r[2] for r in tab if r[0].rstrip("~") in ("u", "d", "s", "c", "b"))
    mumu = next(r[2] for r in tab if {r[0], r[1]} == {"mu-", "mu+"})
    assert had / mumu == pytest.approx(3.58, abs=0.08)     # 11/3 with the b-quark threshold


def test_conservation_filter_only_skips_true_zeros():
    """Final states the quantum-number filter skips really have zero amplitude."""
    from engine.particles.events import _nonzero, conserves
    for final in [("e-", "mu+"), ("u", "e+"), ("d~", "g"), ("nu_e", "nu_mu~")]:
        assert not conserves(("e-", "e+"), final)
        assert not _nonzero("e-", "e+", final[0], final[1], 200.0)
    assert conserves(("u", "d~"), ("W+", "g"))
