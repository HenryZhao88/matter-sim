"""Laboratory measurements, done on the simulated metal.

Each function runs molecular dynamics with a learned potential and returns what an
experimentalist would record. Together they supply every per-atom property the continuum
block (continuum.py) needs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .eam import EAM
from .md import MD, al_state, coexistence, npt_lattice_constant

HA_EV = 27.211386245988
BOHR_A = 0.529177210903
MELT_T = 1800.0     # K: hot enough to melt quickly, not so hot that atoms slam into their cores
RESULTS = Path(__file__).resolve().parents[2] / ".cache" / "materials" / "al_results.json"


def thermal_curve(model: EAM, a0: float, temps, n=(4, 4, 4), steps: int = 4000, log=None) -> list[dict]:
    """Lattice constant and enthalpy of the crystal at zero pressure, temperature by temperature."""
    rows = []
    for T in temps:
        r = npt_lattice_constant(model, a0, T, n=n, steps=steps)
        rows.append({"T": T, "a_A": r["a"] * BOHR_A, "H_eV": r["H_per_atom"] * HA_EV, "T_measured": r["T_measured"]})
        if log:
            log(rows[-1])
    return rows


def melting_point(model: EAM, a_of_T, lo: float, hi: float, iters: int = 5, log=None,
                  n=(5, 5, 12), engine: str = "numpy") -> dict:
    """Bisection on the direction a half-solid, half-liquid box moves: the crystal grows below
    the melting point (potential energy falls) and melts above it (potential energy rises).
    ``n`` is the box in fcc cells (4 atoms each); ``engine="torch"`` runs it on a GPU."""
    trail = []
    for _ in range(iters):
        T = 0.5 * (lo + hi)
        r = coexistence(model, a_of_T(T) / BOHR_A, T, n=n, engine=engine)
        trail.append({"T": T, "slope": r["slope"]})
        if log:
            log(trail[-1])
        if r["slope"] > 0:
            hi = T
        else:
            lo = T
    return {"T_melt": 0.5 * (lo + hi), "bracket": [lo, hi], "trail": trail}


def latent_heat(model: EAM, T: float, a_T: float, n=(4, 4, 4), steps: int = 5000) -> dict:
    """Enthalpy per atom of liquid minus solid, both held at T and zero pressure."""
    solid = npt_lattice_constant(model, a_T / BOHR_A, T, n=n, steps=steps)
    state = al_state(model, a_T / BOHR_A, n)
    md = MD(model, state, seed=3)
    md.thermalise(MELT_T)
    md.run(2000, T=MELT_T, sample_every=100)                  # melt it at fixed volume
    md.run(1500, T=T, P_GPa=0.0, sample_every=100)            # cool the liquid to T, then let it relax
    rows = md.run(steps, T=T, P_GPa=0.0, sample_every=10)
    tail = rows[len(rows) // 2:]
    N = len(state.pos)
    H_liq = float(np.mean([r["E"] for r in tail])) / N
    V_liq = float(np.mean([r["V"] for r in tail])) / N
    V_sol = (solid["a"] ** 3) / 4
    return {"latent_eV": (H_liq - solid["H_per_atom"]) * HA_EV, "dV_melt_frac": V_liq / V_sol - 1}


def run_all(model: EAM, a0_A: float, log=print) -> dict:
    t0 = time.perf_counter()
    temps = [100, 300, 500, 700, 900]
    curve = thermal_curve(model, a0_A / BOHR_A, temps, log=lambda r: log("thermal", r))
    t, a = np.array([(r["T"], r["a_A"]) for r in curve]).T
    ca = np.polyfit(t, a, 2)
    a_of_T = lambda T: float(np.polyval(ca, T))
    melt = melting_point(model, a_of_T, 500.0, 1500.0, iters=6, log=lambda r: log("coexistence", r))
    lat = latent_heat(model, melt["T_melt"], a_of_T(melt["T_melt"]))
    out = {"thermal": curve, "melting": melt, "latent": lat, "seconds": time.perf_counter() - t0}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=1))
    return out
