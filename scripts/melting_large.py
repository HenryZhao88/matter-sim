"""Aluminium's melting point by solid-liquid coexistence in a large box, on a GPU.

The same measurement as experiments.run_all (bisection on whether a half-solid, half-liquid box
freezes or melts, with the lattice constant at each temperature from the thermal curve already
in results/al_results.json), in a box of n fcc cells instead of 5x5x12: the interface is a smaller
fraction of the atoms, so the measured slope is less noisy and the finite-size error smaller.
Each temperature's result is cached, so the run can be stopped and restarted.

    uv run --extra gpu python scripts/melting_large.py 20 20 60      # 96 000 atoms

Writes results/al_melting_<N>.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from engine.materials.eam import EAM
from engine.materials.md import coexistence

ROOT = Path(__file__).resolve().parents[1]
BOHR_A = 0.529177210903


def main(n: tuple[int, int, int], lo: float = 500.0, hi: float = 1500.0, iters: int = 6, engine: str = "torch") -> None:
    model = EAM.load(ROOT / "results" / "al_eam_aluminium.npz")
    thermal = json.loads((ROOT / "results" / "al_results.json").read_text())["thermal"]
    t, a = np.array([(r["T"], r["a_A"]) for r in thermal]).T
    ca = np.polyfit(t, a, 2)                                   # as run_all does
    N = 4 * n[0] * n[1] * n[2]
    cache = ROOT / ".cache" / "materials" / f"melt_{n[0]}x{n[1]}x{n[2]}"
    cache.mkdir(parents=True, exist_ok=True)
    trail = []
    t0 = time.time()
    for _ in range(iters):
        T = 0.5 * (lo + hi)
        f = cache / f"T{T:.4f}.json"
        if f.exists():
            row = json.loads(f.read_text())
        else:
            s = time.time()
            r = coexistence(model, float(np.polyval(ca, T)) / BOHR_A, T, n=n, engine=engine)
            row = {"T": T, "slope": r["slope"], "U_start": r["U_start"], "U_end": r["U_end"], "seconds": time.time() - s}
            f.write_text(json.dumps(row))
        trail.append(row)
        print(f"T {T:8.3f} K: slope {row['slope']:+.3e} Ha/atom/fs ({'melts' if row['slope'] > 0 else 'freezes'}), "
              f"{row['seconds']:.0f} s", flush=True)
        if row["slope"] > 0:
            hi = T
        else:
            lo = T
    out = {"atoms": N, "cells": list(n), "engine": engine, "T_melt": 0.5 * (lo + hi), "bracket": [lo, hi],
           "trail": trail, "seconds": time.time() - t0,
           "method": "experiments.melting_point / md.coexistence, bisection on [500, 1500] K, 6 steps"}
    (ROOT / "results" / f"al_melting_{N}.json").write_text(json.dumps(out, indent=1))
    print(f"T_melt = {out['T_melt']:.1f} K (bracket {lo:.1f}-{hi:.1f}), {N} atoms", flush=True)


if __name__ == "__main__":
    main(tuple(int(x) for x in sys.argv[1:4]) if len(sys.argv) >= 4 else (20, 20, 60))
