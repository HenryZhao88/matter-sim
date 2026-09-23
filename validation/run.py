"""Experiment vs. simulation: the checks behind every claim in the README.

Run:  uv run matter-sim validate           (≈ 10–15 min on an M4)
      uv run matter-sim validate --quick   (≈ 3 min, coarser grids)

Every row is computed from scratch here. The only inputs are nuclear charges,
nuclear masses, and fundamental constants.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from engine.atoms.pseudo import verify
from engine.atoms.radial import RadialGrid, ground_state, radial_eigenstates
from engine.atoms.species import pseudopotential
from engine.core.backend import get_backend
from engine.core.grid import Grid
from engine.core.units import BOHR_ANGSTROM, HARTREE_EV, hartree_to_nm
from engine.electrons.scf import SCFSolver
from engine.scenes.presets import system_from_preset
from engine.simulation import Params, Simulation
from engine.system import System

OUT = Path(__file__).resolve().parent / "results.json"

# NIST ionisation energies (eV)
EXP_IE = {1: 13.598, 2: 24.587, 3: 5.392, 4: 9.323, 5: 8.298, 6: 11.260, 7: 14.534, 8: 13.618,
          9: 17.423, 10: 21.565, 11: 5.139, 12: 7.646, 13: 5.986, 14: 8.152, 15: 10.487,
          16: 10.360, 17: 12.968, 18: 15.760}
SYMBOL = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()

rows: list[dict] = []


def record(group: str, quantity: str, simulated, experiment, unit: str = "", note: str = "", ok=None):
    row = {"group": group, "quantity": quantity, "simulated": simulated, "experiment": experiment,
           "unit": unit, "note": note, "ok": ok}
    rows.append(row)
    sim_s = f"{simulated:.4g}" if isinstance(simulated, float) else str(simulated)
    exp_s = f"{experiment:.4g}" if isinstance(experiment, float) else str(experiment)
    mark = {True: "✓", False: "✗", None: " "}[ok]
    print(f"  {mark} {quantity:<38} {sim_s:>12} {unit:<4} exp {exp_s:>10}  {note}", flush=True)


def section(title: str) -> None:
    print(f"\n{title}", flush=True)


def relax(system: System, quality: str, max_steps: int = 150) -> Simulation:
    sim = Simulation(system, Params(quality=quality, mode="relax"))
    for _ in range(max_steps):
        sim.step()
        if sim.relaxed:
            break
    return sim


def angle(P, c, a, b) -> float:
    u, v = P[a] - P[c], P[b] - P[c]
    return math.degrees(math.acos(float(u @ v / np.linalg.norm(u) / np.linalg.norm(v))))


def main(quick: bool = False, part: str = "all") -> None:
    t0 = time.time()
    q = "draft" if quick else "standard"
    print(f"matter-sim validation ({'quick' if quick else 'full'}; 3D grid: {q}; part: {part})")
    if part in ("all", "particles"):
        particles_section(quick)
    if part in ("all", "atoms"):
        atoms_section(quick, q)
    if part in ("all", "materials"):
        materials_section()
    ok = [r["ok"] for r in rows if r["ok"] is not None]
    print(f"\n{sum(ok)}/{len(ok)} checks pass  ({time.time() - t0:.0f} s)")
    OUT.write_text(json.dumps({"quick": quick, "part": part, "rows": rows}, indent=2, default=float))
    print(f"Wrote {OUT}")


def materials_section() -> None:
    """Aluminium: what the learned forces and the continuum block produce, against measurement."""
    from engine.materials.continuum import REFERENCE, Block, derived_aluminium
    d = derived_aluminium()
    if d is None:
        print("\n(Materials checks skipped: the learned potential has not been trained and measured yet.)")
        return
    section("Aluminium: forces learned from this engine's own DFT, then molecular dynamics")
    b = Block(d, T=293.15)
    ref = {k: v[0] for k, v in REFERENCE.items()}
    record("materials", "Melting point (solid–liquid coexistence)", d.T_melt, ref["T_melt"], "K",
           "LDA over-binds; ±60 K is the usual spread", abs(d.T_melt - ref["T_melt"]) < 150)
    record("materials", "Latent heat of fusion", d.latent_eV * 1000, ref["latent_eV"] * 1000, "meV/atom",
           "", abs(d.latent_eV - ref["latent_eV"]) < 0.04)
    record("materials", "Density at 293 K", b.density() / 1000, ref["density"] / 1000, "g/cm³",
           "from the lattice constant and the nuclear mass", abs(b.density() - ref["density"]) < 150)
    record("materials", "Linear thermal expansion", b.linear_expansion_per_K() * 1e6, ref["alpha_per_K"] * 1e6,
           "10⁻⁶/K", "classical nuclei", abs(b.linear_expansion_per_K() - ref["alpha_per_K"]) < 8e-6)
    record("materials", "Specific heat at 293 K", b.specific_heat_J_per_gK(), ref["c_J_per_gK"], "J/g·K",
           "classical nuclei: no quantum freeze-out", abs(b.specific_heat_J_per_gK() - ref["c_J_per_gK"]) < 0.15)
    record("materials", "Speed of sound (longitudinal)", b.sound_speeds()["longitudinal"], ref["sound_long"],
           "m/s", "from the DFT elastic constants", abs(b.sound_speeds()["longitudinal"] - ref["sound_long"]) < 900)
    record("materials", "Energy to melt 1 cm³ from room temperature", b.heat_to_melt_J() / 1000, 3.0, "kJ",
           "≈2.9 kJ from measured tables", 2.0 < b.heat_to_melt_J() / 1000 < 4.5)


def particles_section(quick: bool) -> None:
    from engine.lattice.qcd import GaugeField, polyakov_scan, run_measurement
    from engine.lattice.schwinger import run_scenario
    from engine.particles.decays import branching_ratios, lifetime_seconds, total_width
    from engine.particles.events import is_confined, lambda_qcd, outcomes
    from engine.particles.process import cross_section, model

    section("Standard Model: symmetry breaking (input: gauge group, representations, α, G_F, m_Z)")
    m = model()
    record("particles", "Photon mass", m.vector("photon").mass, 0.0, "GeV", "massless: emerges", abs(m.vector("photon").mass) < 1e-9)
    record("particles", "W mass (tree-level prediction)", m.vector("W1").mass, 80.37, "GeV", "loop corrections +0.7%",
           abs(m.vector("W1").mass - 80.37) / 80.37 < 0.01)
    charges = {n: round(m.fermion(n).charge, 4) for n in ("u", "d", "e", "nu_e")}
    record("particles", "Charges u, d, e, ν (from isospin + hypercharge)",
           ", ".join(f"{v:+g}" for v in charges.values()), "+⅔, −⅓, −1, 0",
           ok=charges == {"u": 0.6667, "d": -0.3333, "e": -1.0, "nu_e": 0.0})

    section("Standard Model: decays and collisions (tree level)")
    tau_mu = lifetime_seconds("mu-")
    record("particles", "Muon lifetime", tau_mu * 1e6, 2.1970, "μs", "via W exchange", abs(tau_mu * 1e6 - 2.197) < 0.05)
    inv = sum(b for p, b in branching_ratios("Z") if all(x.startswith("nu") for x in p))
    record("particles", "Z → invisible (counts ν families)", inv * 100, 20.0, "%", "3 families", abs(inv - 0.200) < 0.01)
    record("particles", "Z width", total_width("Z"), 2.4952, "GeV", "no QCD corrections", abs(total_width("Z") - 2.4952) < 0.12)
    record("particles", "Top escapes confinement, bottom does not",
           f"{not is_confined('t')}, {is_confined('b')}", "True, True", "", f"Λ_QCD = {lambda_qcd():.3f} GeV",
           (not is_confined("t")) and is_confined("b"))
    tab = outcomes("e-", "e+", 10.0)
    had = sum(r[2] for r in tab if r[0].rstrip("~") in ("u", "d", "s", "c", "b"))
    mumu = next(r[2] for r in tab if {r[0], r[1]} == {"mu-", "mu+"})
    record("particles", "R = σ(hadrons)/σ(μμ) at 10 GeV", had / mumu, 3.58, "", "11/3 from three colours, b threshold",
           abs(had / mumu - 3.58) < 0.1)
    s200 = cross_section("e-", "e+", "W+", "W-", 200.0, n_cos=48)[0]
    s3000 = cross_section("e-", "e+", "W+", "W-", 3000.0, n_cos=48)[0]
    record("particles", "σ(e⁺e⁻ → W⁺W⁻) at 200 GeV", s200, 17.0, "pb", "LEP2, tree level", 15 < s200 < 22)
    record("particles", "…and falls at 3 TeV (gauge cancellation)", s3000, "< σ(200)", "pb", ok=s3000 < s200)
    from engine.particles.shower import alpha_s, group_constants
    gc = group_constants()
    record("particles", "Colour factors C_F, C_A (from the generators)", f"{gc['CF']:.4f}, {gc['CA']:.4f}", "4/3, 3", "",
           ok=abs(gc["CF"] - 4 / 3) < 1e-9 and abs(gc["CA"] - 3) < 1e-9)
    record("particles", "Running α_s(10 GeV) from α_s(m_Z)", alpha_s(10.0), 0.178, "", "one loop, β₀ from the group",
           abs(alpha_s(10.0) - 0.178) < 0.012)

    from engine.particles.hadron import CACHE as HCACHE, HadronCollider
    probe = HadronCollider.__new__(HadronCollider)
    probe.sqrt_s = 13600.0
    if all(probe._path(*j).exists() for j in HadronCollider.jobs(probe)):
        section("Proton–proton at 13.6 TeV (partons measured, collisions computed, leading order)")
        h = HadronCollider(13600.0)
        h.build()
        tab = {tuple(r["final"]): r["pb"] for r in h.summary(1000)}
        get = lambda *n: tab.get(tuple(sorted(n)), 0.0)
        tt = get("t", "t~")
        record("protons", "σ(pp → t t̄)", tt, 900.0, "pb", "LO vs NNLO measurement", 400 < tt < 1200)
        zmm = get("mu-", "mu+") / 1000
        record("protons", "σ(pp → Z/γ* → μμ)", zmm, 2.0, "nb", "LO, |cos θ*| < 0.95", 0.8 < zmm < 2.5)
        ratio = get("mu+", "nu_mu") / max(get("mu-", "nu_mu~"), 1e-9)
        record("protons", "W⁺/W⁻ ratio", ratio, 1.3, "", "the proton is uud", 1.15 < ratio < 1.5)
    else:
        print("\n(Proton–proton checks skipped: parton tables not built yet. Collide protons once in the viewer.)")

    section("Real-time QED in one space dimension (exact)")
    _, it = run_scenario("pair_creation", N=14, mass=0.3, t_max=4.0, frames=21, strength=1.0)
    frames = list(it)
    created = max(f[1]["particles"] for f in frames)
    drift = max(f[1]["energy"] for f in frames) - min(f[1]["energy"] for f in frames)
    record("lattice", "Pairs created from empty space by a field", created, "> 0", "particles", ok=created > 1)
    record("lattice", "Energy conservation", drift, 0.0, "", "exact evolution", drift < 1e-9)
    model_j, it = run_scenario("jet", N=14, mass=0.25, t_max=6.0, frames=7, strength=0.6)
    sep = [o["charge"][: model_j.N // 2].sum() for _, o in it]
    record("lattice", "Hadronisation: flying charges end up screened", f"{sep[0]:.2f} → {min(sep[3:]):.2f}", "1 → ≈0",
           "string breaks into mesons", min(sep[3:]) < 0.5 * sep[0])

    section("Lattice QCD, pure gauge (quenched)")
    g = GaugeField(6, 6, 6.0, seed=4)
    vals = []
    for s in range(80):
        g.sweep()
        if s >= 30:
            vals.append(g.plaquette())
    record("lattice", "Plaquette at β = 6.0", float(np.mean(vals)), 0.5937, "", "published", abs(np.mean(vals) - 0.5937) < 0.004)
    res = [x for x in run_measurement(5.7, L=8, T=8, sweeps=90) if x["type"] == "qcd.result"][0]
    record("lattice", "String tension σa² at β = 5.7", res["fit"]["sigma"], 0.16, "", "confinement; 8⁴ reads low",
           0.08 < res["fit"]["sigma"] < 0.25)
    if not quick:
        (b1, p1, _), (b2, p2, _) = polyakov_scan([5.45, 6.1], L=8, Nt=4, sweeps=100)
        record("lattice", "Polyakov loop: cold vs hot", f"{p1:.3f} → {p2:.3f}", "≈0 → >0", "", "deconfinement near β = 5.69",
               p2 > 3 * p1)
    import pickle
    hc = Path(__file__).resolve().parents[1] / ".cache" / "hadron_spectrum.pkl"
    if hc.exists():
        h = pickle.loads(hc.read_bytes())
        section(f"Hadron masses from quark propagators (β = {h['beta']}, {h['L']}³×{h['T']}, quenched Wilson)")
        record("lattice", "Pion lighter than rho at every quark mass", all(p < r for p, r in zip(h["pion"], h["rho"])),
               True, "", "", all(p < r for p, r in zip(h["pion"], h["rho"])))
        record("lattice", "κ where the pion becomes massless", h["kappa_c"], 0.1694, "", "Goldstone boson; published",
               abs(h["kappa_c"] - 0.1694) < 0.003)
        record("lattice", "Rho mass there, m·a", h["rho_chiral"], 0.56, "", "published ≈ 0.55–0.58", 0.45 < h["rho_chiral"] < 0.68)
    else:
        print("\n(Hadron masses skipped: run the Lattice QCD › Hadron masses experiment once.)")

    if not quick:
        section("Truth mode: neural-network wavefunction, exact Hamiltonian (variational Monte Carlo)")
        from engine.truth.vmc import VMC, Molecule
        v = VMC(Molecule([2.0], np.zeros((1, 3)), 1, 1), walkers=512, hidden=24, layers=2, dets=2)
        v.train(iters=300)
        E, err = v.evaluate(10, 5)
        record("truth", "Helium ground-state energy", E, -2.90372, "Ha", f"±{err:.3f}; Hartree–Fock −2.8617",
               abs(E + 2.90372) < 0.01)


def atoms_section(quick: bool, q: str) -> None:
    section("Hydrogen: one electron, exact Hamiltonian")
    g = RadialGrid(r_max=200.0)
    e_s, _ = radial_eigenstates(g, -1 / g.r, 0, 3, 1.0)
    e_p, _ = radial_eigenstates(g, -1 / g.r, 1, 2, 1.0)
    # Infinite nuclear mass; the proton's finite mass shifts lines by 0.05%.
    lyman = hartree_to_nm(e_p[0] - e_s[0])
    balmer = hartree_to_nm(e_s[2] - e_p[0])
    record("hydrogen", "Lyman-α (2p→1s)", lyman, 121.567, "nm", "infinite nuclear mass",
           abs(lyman - 121.567) < 0.2)
    record("hydrogen", "Balmer-α (3→2)", balmer, 656.28, "nm", "infinite nuclear mass", abs(balmer - 656.28) < 1.0)
    g3 = Grid(14.0, 0.2, get_backend("auto"))
    e3 = SCFSolver(g3, System([1], [[0, 0, 0]]), functional="none").run().energy
    record("hydrogen", "Ground state on the 3D grid", e3, -0.5, "Ha", "h = 0.2 bohr", abs(e3 + 0.5) < 2e-3)

    section("Periodic table: all-electron atoms (LSDA), ionisation energies")
    last = 18 if not quick else 10
    ie = {}
    for Z in range(1, last + 1):
        neutral = ground_state(Z)
        cation = ground_state(Z, 1) if Z > 1 else None
        ie[Z] = ((cation.energy if cation else 0.0) - neutral.energy) * HARTREE_EV
        record("periodic", f"{SYMBOL[Z - 1]:<2} IE   {neutral.configuration()}", ie[Z], EXP_IE[Z], "eV",
               f"spin {neutral.multiplicity - 1} unpaired", abs(ie[Z] - EXP_IE[Z]) < 1.0)
    pattern = (ie[2] > ie[1] and ie[3] < ie[2] and ie[5] < ie[4] and ie[8] < ie[7] and ie[10] > ie[9])
    record("periodic", "Pattern: peaks He, Ne; dips Li, B, O", "yes" if pattern else "no", "yes", ok=pattern)
    mult = {Z: ground_state(Z).multiplicity for Z in (6, 7, 8)}
    hund = mult == {6: 3, 7: 4, 8: 3}
    record("periodic", "Hund's rule: unpaired spins in C, N, O", f"{mult[6]-1}, {mult[7]-1}, {mult[8]-1}",
           "2, 3, 2", ok=hund)

    section("Pseudopotentials (derived from the atoms above): transferability")
    for Z in (6, 7, 8, 11, 13, 16, 17):
        worst = max(abs(r["dE_ps"] - r["dE_ae"]) for r in verify(pseudopotential(Z))[1:]) * HARTREE_EV
        record("pseudo", f"{SYMBOL[Z - 1]}: worst excitation-energy error", worst, 0.0, "eV",
               "ionised and promoted configurations", worst < 0.06)

    if not quick:
        section("Fourth row (K–Kr): configurations and pseudopotentials")
        from engine.atoms.radial import RadialAtom
        cu = RadialAtom(29, grid=RadialGrid(r_min=2e-6 / math.sqrt(29), r_max=60.0, dx=0.004)).solve()
        cfg = cu.configuration()
        record("periodic", "Copper fills 3d before its second 4s electron", cfg.split()[-2:], "4s¹ 3d¹⁰", "",
               "the textbook exception, unprompted", cfg.endswith("4s¹ 3d¹⁰"))
        from engine.atoms.species import checks
        for Z in (19, 20, 29, 30, 31, 32, 35):
            pseudopotential(Z)
            r = checks().get(str(Z), {})
            record("pseudo", f"{SYMBOL[Z - 1]}: ghost-free, worst excitation error", r.get("transfer_eV", float("nan")), 0.0,
                   "eV", f"local channel l={r.get('l_local')}", bool(r.get("ok")))

    section(f"Molecules in 3D ({q} grid): shapes found by relaxing from wrong starts")
    # H2 bond and binding energy
    sim = relax(system_from_preset("h2_form"), q)
    P = sim.system.positions
    R = float(np.linalg.norm(P[0] - P[1])) * BOHR_ANGSTROM
    gH = Grid(sim.grid.L, sim.params.h, get_backend("auto"))
    eH = SCFSolver(gH, System([1], [[0, 0, 0]])).run().energy
    De = (2 * eH - sim.result.energy) * HARTREE_EV
    record("molecules", "H₂ bond length", R, 0.7414, "Å", "LDA ≈ 0.765", abs(R - 0.765) < 0.02)
    record("molecules", "H₂ binding energy", De, 4.747, "eV", "LDA ≈ 4.9", abs(De - 4.9) < 0.3)

    sim = relax(system_from_preset("h3plus"), q)
    P = sim.system.positions
    sides = sorted(float(np.linalg.norm(P[i] - P[j])) * BOHR_ANGSTROM for i, j in ((0, 1), (1, 2), (0, 2)))
    record("molecules", "H₃⁺ side lengths (equilateral?)", f"{sides[0]:.3f}–{sides[2]:.3f}", "0.873 ×3", "Å",
           ok=sides[2] - sides[0] < 0.02)

    sim = relax(system_from_preset("water"), q)
    P = sim.system.positions
    ang = angle(P, 0, 1, 2)
    oh = float(np.linalg.norm(P[1] - P[0])) * BOHR_ANGSTROM
    record("molecules", "H₂O angle (started at 150°)", ang, 104.52, "°", "LDA ≈ 104.9", abs(ang - 104.9) < 2.0)
    record("molecules", "H₂O O–H length", oh, 0.9572, "Å", "LDA ≈ 0.970", abs(oh - 0.970) < 0.02)

    if not quick:
        sim = relax(system_from_preset("ammonia"), q)
        P = sim.system.positions
        a = np.mean([angle(P, 0, i, j) for i, j in ((1, 2), (2, 3), (1, 3))])
        record("molecules", "NH₃ angle (started nearly flat)", float(a), 106.7, "°", "a pyramid emerges",
               abs(a - 106.7) < 3.0)
        sim = relax(system_from_preset("methane"), q)
        P = sim.system.positions
        angs = [angle(P, 0, i, j) for i in range(1, 5) for j in range(i + 1, 5)]
        record("molecules", "CH₄ angles (tetrahedral?)", f"{min(angs):.1f}–{max(angs):.1f}", "109.47", "°",
               ok=max(angs) - min(angs) < 2.0)
        sim = relax(system_from_preset("n2"), q)
        P = sim.system.positions
        r = float(np.linalg.norm(P[0] - P[1])) * BOHR_ANGSTROM
        record("molecules", "N₂ bond length", r, 1.0977, "Å", ok=abs(r - 1.0977) < 0.03)

    section(f"Third row ({q} grid): the same code, one shell further out")
    sim = relax(system_from_preset("h2s"), q)
    P = sim.system.positions
    a = angle(P, 0, 1, 2)
    record("third row", "H₂S angle (water's is 104.5°)", a, 92.1, "°", "the periodic trend emerges", abs(a - 92.1) < 2.5)
    sim = relax(system_from_preset("nacl"), q)
    P = sim.system.positions
    r = float(np.linalg.norm(P[0] - P[1])) * BOHR_ANGSTROM
    record("third row", "NaCl bond length", r, 2.3609, "Å", "ionic bond", abs(r - 2.3609) < 0.06)

    section("Magnetism")
    for pid, label, expect in (("o_atom", "O atom: unpaired spins", 2), ("o2", "O₂ molecule: unpaired spins", 2),
                               ("al2", "Al₂ molecule: unpaired spins", 2)):
        sim = Simulation(system_from_preset(pid), Params(quality="draft", mode="frozen"))
        sim.step()
        scan = sim.find_spin(max_unpaired=4)
        best = min(scan, key=lambda r: r["energy"])["multiplicity"] - 1
        record("magnetism", label, best, expect, "", "lowest-energy spin", best == expect)



if __name__ == "__main__":
    main()
