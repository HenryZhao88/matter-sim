"""The materials experiments for any fcc or bcc element's potential, stage by stage and resumable.

Temperature scale (the potential's own), thermal curve, melting point by solid-liquid
coexistence (a large box, on the GPU when there is one) and latent heat: the same functions as
experiments.run_all, each stage cached in .cache/materials/<el>_experiments/ so a stopped run
resumes where it was. The lattice constant is the potential's own zero-temperature minimum.

    uv run --extra gpu python scripts/element_experiments.py Cu results/cu_eam_final.npz 63.546 20 20 60
    uv run --extra gpu python scripts/element_experiments.py Fe results/fe_eam_final.npz 55.845 25 25 75 bcc

The mass is the element's standard atomic weight (the natural isotope mix, amu). Writes
results/<el>_results.json, and never over an existing one (set MATTER_SIM_OVERWRITE=1 to replace it).
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from engine.core.accel import have_cuda
from engine.crystal.periodic import cubic
from engine.materials.eam import EAM, energy_forces
from engine.materials.experiments import (BOHR_A, HA_EV, latent_heat, schedule_from_scale, solid_near_melting,
                                          temperature_scale, thermal_curve)
from engine.materials.md import ATOMS_PER_CELL, coexistence

ROOT = Path(__file__).resolve().parents[1]


def cached(path: Path, compute):
    if path.exists():
        return json.loads(path.read_text())
    t = time.time()
    out = compute()
    out["seconds"] = time.time() - t
    path.write_text(json.dumps(out))
    return out


def potential_a0(model: EAM, Z: int, structure: str = "fcc") -> float:
    """The lattice constant (bohr) of ``structure`` that minimises the potential's own energy."""
    def E(a):
        c = cubic(structure, a, Z)
        return energy_forces(model, c.cell, c.positions)[0] / len(c.charges)
    # the same range of volume per atom for either structure (fcc's original 5.5-9.0 bohr)
    aa = np.linspace(5.5, 9.0, 36) * (ATOMS_PER_CELL[structure] / 4) ** (1 / 3)
    a1 = aa[np.argmin([E(a) for a in aa])]
    aa = a1 * np.linspace(0.97, 1.03, 13)
    c = np.polyfit(aa, [E(a) for a in aa], 4)
    r = np.roots(np.polyder(c))
    r = r[np.isreal(r)].real
    return float(r[np.argmin(np.abs(r - a1))])


def main(el: str, pot: str, mass: float, n=(20, 20, 60), structure: str = "fcc") -> None:
    from engine.core.elements import ELEMENTS
    Z = next(z for z, e in ELEMENTS.items() if e.symbol.lower() == el.lower())
    model = EAM.load(ROOT / pot)
    result = ROOT / "results" / f"{el.lower()}_results.json"
    if result.exists() and os.environ.get("MATTER_SIM_OVERWRITE") != "1":
        raise SystemExit(f"{result.name} exists: not overwritten (set MATTER_SIM_OVERWRITE=1 to replace it)")
    atoms = ATOMS_PER_CELL[structure] * n[0] * n[1] * n[2]
    cache = ROOT / ".cache" / "materials" / f"{el.lower()}_experiments"
    cache.mkdir(parents=True, exist_ok=True)
    engine = "torch" if have_cuda() else "numpy"
    a0 = potential_a0(model, Z, structure)
    print(f"{el}: {structure}, potential's a0 = {a0 * BOHR_A:.4f} A, mass {mass} amu, coexistence box {n} ({atoms} atoms, {engine})", flush=True)
    sc = cached(cache / "scale.json", lambda: temperature_scale(model, a0, mass, structure=structure))
    sched = schedule_from_scale(sc["T_order"])
    print(f"T_order = {sc['T_order']} K -> schedule {sched}", flush=True)
    curve = []
    for T in sched["thermal_T"]:
        row = cached(cache / f"thermal_{T}.json", lambda: {"row": thermal_curve(model, a0, [T], mass_amu=mass, structure=structure)[0]})["row"]
        curve.append(row)
        print(f"thermal {T} K: a = {row['a_A']:.4f} A", flush=True)
    t, a = np.array([(r["T"], r["a_A"]) for r in curve]).T
    ca = np.polyfit(t, a, 2)
    lo, hi = sched["melt_bracket"]
    trail = []
    for _ in range(6):
        T = 0.5 * (lo + hi)
        f = cache / f"coexist_{n[0]}x{n[1]}x{n[2]}_T{T:.4f}.json"
        r = cached(f, lambda: {k: v for k, v in coexistence(model, float(np.polyval(ca, T)) / BOHR_A, T, n=n,
                                                            engine=engine, mass_amu=mass, melt_T=sched["melt_T"],
                                                            structure=structure).items()
                               if k in ("T", "slope", "U_start", "U_end")})
        trail.append(r)
        print(f"coexistence {T:.2f} K: slope {r['slope']:+.3e} ({'melts' if r['slope'] > 0 else 'freezes'}), {r['seconds']:.0f} s", flush=True)
        if r["slope"] > 0:
            hi = T
        else:
            lo = T
    Tm = 0.5 * (lo + hi)
    # v3: the reference solid is checked to still be a solid at T_m; where it is not (iron), its H and V at
    # T_m come from crystalline runs just below (solid_near_melting). Earlier files are kept: v1's 'solid'
    # had melted (liquid minus liquid, 18.8 meV/atom); v2 extrapolated the 161-1150 K curve 1050 K (312).
    a_of_T = lambda T: float(np.polyval(ca, T))

    def latent():
        lat = latent_heat(model, Tm, a_of_T(Tm), mass_amu=mass, melt_T=sched["melt_T"], structure=structure)
        if not lat.get("solid_melted"):
            return lat
        near = cached(cache / f"solid_near_{Tm:.4f}.json",
                      lambda: solid_near_melting(model, a_of_T, Tm, mass, structure, engine=engine))
        if near["H_eV"] is None:
            return {**lat, "solid_near_melting": near["points"]}
        return {**latent_heat(model, Tm, a_of_T(Tm), mass_amu=mass, melt_T=sched["melt_T"], structure=structure,
                              solid_H_eV=near["H_eV"], solid_V_bohr3=near["V_bohr3"], solid_method=near["method"]),
                "solid_near_melting": near["points"]}

    lat = cached(cache / f"latent_v3_{Tm:.4f}.json", latent)
    out = {"element": el, "structure": structure, "potential": pot, "mass_amu": mass, "a0_A": a0 * BOHR_A, "schedule": sched,
           "T_order": sc["T_order"], "thermal": curve,
           "melting": {"T_melt": Tm, "bracket": [lo, hi], "trail": trail, "atoms": atoms, "engine": engine},
           "latent": {k: v for k, v in lat.items() if k != "seconds"}}
    result.write_text(json.dumps(out, indent=1))
    L = "withheld (the reference solid melted)" if lat["latent_eV"] is None else f"{lat['latent_eV'] * 1000:.1f} meV/atom"
    dV = "-" if lat["dV_melt_frac"] is None else f"{lat['dV_melt_frac'] * 100:.1f}%"
    print(f"{el}: T_melt = {Tm:.1f} K ({lo:.1f}-{hi:.1f}), latent heat {L}, melting expansion {dV}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]), tuple(int(x) for x in a[3:6]) if len(a) >= 6 else (20, 20, 60),
         a[6] if len(a) >= 7 else "fcc")
