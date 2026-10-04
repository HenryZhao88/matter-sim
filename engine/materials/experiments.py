"""Laboratory measurements, done on the simulated metal.

Each function runs molecular dynamics with a learned potential and returns what an
experimentalist would record. Together they supply every per-atom property the continuum
block (continuum.py) needs.

Any fcc or bcc element: pass its mass (and ``structure="bcc"`` for a bcc one), and let
``temperature_scale`` find its temperatures from its own potential (the temperature at which a
crystal heated in steps loses its order). Aluminium's original settings stay the defaults, so its
recorded results reproduce.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .eam import EAM
from .md import ATOMS_PER_CELL, AL_MASS_AMU, coexistence, crystal_state, make_md, npt_lattice_constant

HA_EV = 27.211386245988
BOHR_A = 0.529177210903
MELT_T = 1800.0     # K: aluminium's melt-down temperature (hot enough to melt quickly, not so hot
                    # that atoms slam into their cores); another element takes 1.2 × its own T_order
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "materials"
RESULTS = CACHE / "al_results.json"

# aluminium's original schedule, kept so its recorded results reproduce exactly
AL_SCHEDULE = {"thermal_T": [100, 300, 500, 700, 900], "melt_bracket": (500.0, 1500.0), "melt_T": MELT_T}


def thermal_curve(model: EAM, a0: float, temps, n=(4, 4, 4), steps: int = 4000, log=None,
                  mass_amu: float = AL_MASS_AMU, engine: str = "numpy", structure: str = "fcc") -> list[dict]:
    """Lattice constant and enthalpy of the crystal at zero pressure, temperature by temperature."""
    rows = []
    for T in temps:
        r = npt_lattice_constant(model, a0, T, n=n, steps=steps, engine=engine, mass_amu=mass_amu, structure=structure)
        rows.append({"T": T, "a_A": r["a"] * BOHR_A, "H_eV": r["H_per_atom"] * HA_EV, "T_measured": r["T_measured"]})
        if log:
            log(rows[-1])
    return rows


def crystal_order(pos, box, n) -> float:
    """|⟨exp(2πi·2x)⟩| over atoms and the three axes, x in cell units: ~1 for an fcc or bcc crystal
    (their (200) reflection; every atom sits at 0 or ½ of a cell), ~N^(−1/2) for a liquid."""
    f = np.asarray(pos) / np.asarray(box) * np.asarray(n)
    return float(np.mean([abs(np.mean(np.exp(2j * np.pi * 2 * f[:, k]))) for k in range(3)]))


def temperature_scale(model: EAM, a0: float, mass_amu: float, n=(5, 5, 5), dT: float = 50.0,
                      steps: int = 1000, T_max: float = 6000.0, engine: str = "numpy", log=None,
                      structure: str = "fcc") -> dict:
    """The temperature at which a crystal heated in steps of dT at zero pressure loses its order.

    It is above the melting point (a perfect crystal with no surface superheats), so it bounds the
    melting search from above and sets the scale of every other temperature. It comes from the
    element's own potential: no measured temperature enters."""
    md = make_md(model, crystal_state(a0, n, mass_amu, structure), seed=11, engine=engine)
    T = dT
    md.thermalise(T)
    trail = []
    while T <= T_max:
        md.run(steps, T=T, P_GPa=0.0, sample_every=steps)
        s = md.s
        q = crystal_order(s.pos, s.box, n)
        trail.append({"T": T, "order": q})
        if log:
            log(trail[-1])
        if q < 0.1:
            return {"T_order": T, "trail": trail}
        T += dT
    raise RuntimeError(f"the crystal kept its order up to {T_max} K: the potential does not melt")


def schedule_from_scale(T_order: float) -> dict:
    """Temperatures for the experiments, as fractions of the element's own T_order.

    A surface-free crystal heated this way superheats a long way: aluminium's potential loses
    order at 1450 K and melts by coexistence at ~900 K, a ratio of 1.6. So the solid's thermal
    curve stops at 0.5 T_order (below the melting point unless that ratio exceeds 2), the
    melting search spans 0.5–1.0 T_order (the crystal cannot hold order above its melting point
    for long, and melts well above half of T_order), and the melt-down runs at 1.2 T_order."""
    return {"thermal_T": [round(f * T_order, 1) for f in (0.07, 0.17, 0.28, 0.39, 0.5)],
            "melt_bracket": (0.5 * T_order, T_order), "melt_T": 1.2 * T_order, "T_order": T_order}


def melting_point(model: EAM, a_of_T, lo: float, hi: float, iters: int = 5, log=None,
                  n=(5, 5, 12), engine: str = "numpy", mass_amu: float = AL_MASS_AMU,
                  melt_T: float = MELT_T, structure: str = "fcc") -> dict:
    """Bisection on the direction a half-solid, half-liquid box moves: the crystal grows below
    the melting point (potential energy falls) and melts above it (potential energy rises).
    ``n`` is the box in cubic cells (fcc 4 atoms each, bcc 2); ``engine="torch"`` runs it on a GPU."""
    trail = []
    for _ in range(iters):
        T = 0.5 * (lo + hi)
        r = coexistence(model, a_of_T(T) / BOHR_A, T, n=n, engine=engine, mass_amu=mass_amu, melt_T=melt_T,
                        structure=structure)
        trail.append({"T": T, "slope": r["slope"]})
        if log:
            log(trail[-1])
        if r["slope"] > 0:
            hi = T
        else:
            lo = T
    return {"T_melt": 0.5 * (lo + hi), "bracket": [lo, hi], "trail": trail}


def solid_near_melting(model: EAM, a_of_T, T_m: float, mass_amu: float, structure: str = "fcc",
                       fractions=(0.80, 0.85, 0.90, 0.95), n=(6, 6, 6), steps: int = 5000,
                       engine: str = "numpy") -> dict:
    """The solid's enthalpy and volume per atom at T_m, for a crystal that will not stay a crystal at T_m
    itself: measured at fractions of T_m, kept only where the run ends still crystalline (order >= 0.5),
    and fitted linearly to T_m. Iron's potential: crystalline to 2050 K, melted by 2150 K, T_m 2201 K;
    its H(T) steepens near the top (~4.2 k_B), which a fit to its low-temperature curve missed by
    ~0.1 eV/atom."""
    pts = []
    for f in fractions:
        T = f * T_m
        s = npt_lattice_constant(model, a_of_T(T) / BOHR_A, T, n=n, steps=steps, engine=engine,
                                 mass_amu=mass_amu, structure=structure)
        pts.append({"T": T, "order": s["order"], "H_eV": s["H_per_atom"] * HA_EV,
                    "V_bohr3": s["a"] ** 3 / ATOMS_PER_CELL[structure]})
    good = [p for p in pts if p["order"] >= 0.5]
    if len(good) < 2:
        return {"points": pts, "H_eV": None, "V_bohr3": None}
    T_, H_, V_ = np.array([(p["T"], p["H_eV"], p["V_bohr3"]) for p in good]).T
    return {"points": pts, "H_eV": float(np.polyval(np.polyfit(T_, H_, 1), T_m)),
            "V_bohr3": float(np.polyval(np.polyfit(T_, V_, 1), T_m)),
            "method": f"linear fit of the crystalline solid at {T_.min():.0f}-{T_.max():.0f} K"}


def latent_heat(model: EAM, T: float, a_T: float, n=(4, 4, 4), steps: int = 5000,
                mass_amu: float = AL_MASS_AMU, melt_T: float = MELT_T, engine: str = "numpy",
                structure: str = "fcc", solid_H_eV: float | None = None, solid_V_bohr3: float | None = None,
                solid_method: str | None = None) -> dict:
    """Enthalpy per atom of liquid minus solid, both held at T and zero pressure.

    The solid must still be a solid at the end of its run. A crystal whose order goes only just above
    its melting point can melt on its own there (iron: T_order 2300 K, T_m 2201 K; its 'solid' ended at
    order 0.07 and the latent heat came out 18.8 meV/atom, liquid minus liquid). Then the solid's
    enthalpy at T is ``solid_H_eV`` (its own H(T) extrapolated, given by the caller) if there is one;
    without it the result is marked invalid rather than reported."""
    solid = npt_lattice_constant(model, a_T / BOHR_A, T, n=n, steps=steps, engine=engine, mass_amu=mass_amu,
                                 structure=structure)
    solid_melted = solid["order"] < 0.5
    state = crystal_state(a_T / BOHR_A, n, mass_amu, structure)
    md = make_md(model, state, seed=3, engine=engine)
    md.thermalise(melt_T)
    md.run(2000, T=melt_T, sample_every=100)                  # melt it at fixed volume
    md.run(1500, T=T, P_GPa=0.0, sample_every=100)            # cool the liquid to T, then let it relax
    rows = md.run(steps, T=T, P_GPa=0.0, sample_every=10)
    tail = rows[len(rows) // 2:]
    N = len(state.pos)
    H_liq = float(np.mean([r["E"] for r in tail])) / N
    V_liq = float(np.mean([r["V"] for r in tail])) / N
    V_sol = (solid["a"] ** 3) / ATOMS_PER_CELL[structure]
    if not solid_melted:
        return {"latent_eV": (H_liq - solid["H_per_atom"]) * HA_EV, "dV_melt_frac": V_liq / V_sol - 1}
    out = {"solid_melted": True, "solid_order": solid["order"],
           "solid_H_method": (solid_method or "supplied by the caller") if solid_H_eV is not None else None}
    if solid_H_eV is None:
        return {**out, "latent_eV": None, "dV_melt_frac": None}
    # the solid's volume likewise: supplied, or else from the caller's a(T) extrapolated to T
    V_sol = solid_V_bohr3 if solid_V_bohr3 is not None else (a_T / BOHR_A) ** 3 / ATOMS_PER_CELL[structure]
    return {**out, "latent_eV": H_liq * HA_EV - solid_H_eV, "dV_melt_frac": V_liq / V_sol - 1}


def run_all(model: EAM, a0_A: float, log=print, element: str = "al", mass_amu: float = AL_MASS_AMU,
            schedule: dict | str | None = None, engine: str = "numpy", coexist_n=(5, 5, 12),
            structure: str = "fcc") -> dict:
    """Thermal curve, melting point and latent heat, written to .cache/materials/<element>_results.json.

    ``schedule``: None for aluminium's original temperatures, "auto" to derive them from this
    potential's own temperature scale (temperature_scale), or a dict like AL_SCHEDULE."""
    t0 = time.perf_counter()
    if schedule is None:
        schedule = AL_SCHEDULE
    elif schedule == "auto":
        sc = temperature_scale(model, a0_A / BOHR_A, mass_amu, engine=engine,
                               log=lambda r: log("order", r), structure=structure)
        schedule = schedule_from_scale(sc["T_order"])
    curve = thermal_curve(model, a0_A / BOHR_A, schedule["thermal_T"], log=lambda r: log("thermal", r),
                          mass_amu=mass_amu, engine=engine, structure=structure)
    t, a = np.array([(r["T"], r["a_A"]) for r in curve]).T
    ca = np.polyfit(t, a, 2)
    a_of_T = lambda T: float(np.polyval(ca, T))
    lo, hi = schedule["melt_bracket"]
    melt = melting_point(model, a_of_T, lo, hi, iters=6, log=lambda r: log("coexistence", r), n=coexist_n,
                         engine=engine, mass_amu=mass_amu, melt_T=schedule["melt_T"], structure=structure)
    lat = latent_heat(model, melt["T_melt"], a_of_T(melt["T_melt"]), mass_amu=mass_amu,
                      melt_T=schedule["melt_T"], engine=engine, structure=structure)
    out = {"element": element, "mass_amu": mass_amu, "schedule": {k: (list(v) if isinstance(v, tuple) else v)
                                                                  for k, v in schedule.items()},
           "thermal": curve, "melting": melt, "latent": lat, "seconds": time.perf_counter() - t0}
    path = RESULTS if element == "al" else CACHE / f"{element}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out
