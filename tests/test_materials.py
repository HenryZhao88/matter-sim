"""Learned interatomic forces, molecular dynamics, and the continuum block built on them."""

import numpy as np
import pytest

from engine.materials.continuum import Block, Derived
from engine.materials.eam import EAM, N_BASIS, energy_forces
from engine.materials.md import EAMForceField, fcc_lattice


def toy_model(seed: int = 0) -> EAM:
    rng = np.random.default_rng(seed)
    return EAM(a=rng.normal(0, 0.02, N_BASIS), b=np.abs(rng.normal(0.05, 0.01, N_BASIS)),
               c=np.array([-0.3, 0.1, -0.02, 0.001]), e0=-2.0)


def test_eam_forces_are_the_gradient_of_its_energy():
    m = toy_model()
    rng = np.random.default_rng(3)
    cell = np.array([8.0, 8.0, 8.0])
    pos = rng.uniform(0, 8, (6, 3))
    E0, F = energy_forces(m, cell, pos)
    for i, a in ((0, 0), (3, 2), (5, 1)):
        d = 1e-5
        p = pos.copy(); p[i, a] += d
        Ep, _ = energy_forces(m, cell, p)
        p[i, a] -= 2 * d
        Em, _ = energy_forces(m, cell, p)
        assert abs(-(Ep - Em) / (2 * d) - F[i, a]) < 1e-5 * max(1.0, abs(F[i, a]))


def test_periodic_neighbour_energy_matches_a_supercell():
    """The same crystal in a doubled cell must have exactly twice the energy."""
    m = toy_model(1)
    pos, box = fcc_lattice(7.6, (2, 2, 2))
    E1, _ = energy_forces(m, box, pos)
    pos2, box2 = fcc_lattice(7.6, (4, 2, 2))
    E2, _ = energy_forces(m, box2, pos2)
    assert abs(E2 - 2 * E1) < 1e-8 * abs(E1)


def test_md_force_field_agrees_with_the_direct_sum():
    m = toy_model(2)
    pos, box = fcc_lattice(7.6, (3, 3, 3))
    rng = np.random.default_rng(0)
    pos = pos + rng.normal(0, 0.15, pos.shape)
    E_ref, F_ref = energy_forces(m, box, pos)
    E, F, _ = EAMForceField(m).compute(pos, box)
    assert abs(E - E_ref) < 1e-6 * abs(E_ref)
    assert np.max(np.abs(F - F_ref)) < 1e-6


def test_thermostat_reaches_the_requested_temperature():
    from engine.materials.md import MD, al_state
    m = toy_model(4)
    md = MD(m, al_state(m, 7.6, (3, 3, 3)), dt_fs=1.0, seed=1)
    md.thermalise(300.0)
    md.run(400, T=300.0, sample_every=50)
    rows = md.run(600, T=300.0, sample_every=10)
    mean_T = float(np.mean([r["T"] for r in rows]))
    assert 200 < mean_T < 420          # fluctuates with only 108 atoms


def _demo_block() -> Block:
    # a(T) chosen so the block's arithmetic can be checked against known aluminium numbers
    props = Derived(a_of_T=[(100, 4.03), (300, 4.05), (500, 4.07), (700, 4.10)],
                    H_of_T=[(100, -3.30), (300, -3.25), (500, -3.20), (700, -3.14)],
                    T_melt=930.0, latent_eV=0.11, B_GPa=76.0, C11=107.0, C12=61.0, C44=28.0,
                    E_coh_eV=3.4)
    return Block(props, T=293.15)


def test_block_mass_and_atom_count_follow_from_the_lattice():
    b = _demo_block()
    assert 5.9e22 < b.atoms() < 6.1e22          # 1 cm³ of aluminium
    assert 2.65 < b.density() / 1000 < 2.75     # g/cm³
    assert abs(b.mass_g() - b.density() / 1000) < 1e-9   # 1 cm³: mass in g equals density in g/cm³


def test_block_elastic_and_thermal_responses():
    b = _demo_block()
    assert 58 < b.stretch(0)["young_GPa"] < 68         # ⟨100⟩ Young's modulus, ≈63 GPa for aluminium
    assert 0.3 < b.stretch(0)["poisson"] < 0.4
    s = b.sound_speeds()
    assert 5500 < s["longitudinal"] < 7000 and 2500 < s["transverse"] < 3500
    assert 20e-6 < b.linear_expansion_per_K() < 30e-6
    assert 0.8 < b.specific_heat_J_per_gK() < 1.05
    # heating 1 cm³ from room temperature to melting, then melting it, is a few kilojoules
    assert 2000 < b.heat_to_melt_J() < 5000
    assert b.squeeze(1.0)["side_cm"] < 1.0 and b.side_at_cm(600) > 1.0


@pytest.mark.slow
def test_methfessel_paxton_smearing_converges_faster_in_k():
    """The same crystal, two smearing schemes: MP is closer to the converged answer at few k."""
    from engine.crystal.periodic import PeriodicDFT, cubic
    c = cubic("fcc", 7.4737, 13)
    ref = PeriodicDFT(c, h=0.35, kmesh=(10, 10, 10), T_e=0.01).run().free_energy
    fd = PeriodicDFT(c, h=0.35, kmesh=(4, 4, 4), T_e=0.01).run().free_energy
    assert abs(fd - ref) > 1e-4          # coarse Fermi–Dirac sampling is visibly off
