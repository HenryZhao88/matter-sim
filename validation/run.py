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
SYMBOL = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar".split()

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
    ok = [r["ok"] for r in rows if r["ok"] is not None]
    print(f"\n{sum(ok)}/{len(ok)} checks pass  ({time.time() - t0:.0f} s)")
    OUT.write_text(json.dumps({"quick": quick, "part": part, "rows": rows}, indent=2, default=float))
    print(f"Wrote {OUT}")


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

    section("Real-time QED in one space dimension (exact)")
    _, it = run_scenario("pair_creation", N=14, mass=0.3, t_max=4.0, frames=21, strength=1.0)
    frames = list(it)
    created = max(f[1]["particles"] for f in frames)
    drift = max(f[1]["energy"] for f in frames) - min(f[1]["energy"] for f in frames)
    record("lattice", "Pairs created from empty space by a field", created, "> 0", "particles", ok=created > 1)
    record("lattice", "Energy conservation", drift, 0.0, "", "exact evolution", drift < 1e-9)

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
    g3 = Grid(14.0, 0.2, get_backend("mlx"))
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

    section(f"Molecules in 3D ({q} grid): shapes found by relaxing from wrong starts")
    # H2 bond and binding energy
    sim = relax(system_from_preset("h2_form"), q)
    P = sim.system.positions
    R = float(np.linalg.norm(P[0] - P[1])) * BOHR_ANGSTROM
    gH = Grid(sim.grid.L, sim.params.h, get_backend("mlx"))
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
