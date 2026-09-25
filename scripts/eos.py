"""An element's own fcc lattice constant and bulk modulus from our DFT (resumable, float64).

The lattice constant a training set is built around must come from this engine's DFT, not from
experiment. Each volume is one periodic DFT run on the 4-atom cubic cell, cached in
.cache/materials/eos/ by its settings, so the scan can be stopped and restarted at will; the
Birch–Murnaghan fit is redone from whatever is cached. The grid (h, xc_grid) is the element's
converged one from dataset.GRID.

    uv run python scripts/eos.py 29 [k] [workers]

Writes results/<symbol>_eos.json. Experiment appears only in validation/run.py.
"""

import hashlib
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from engine.core.elements import ELEMENTS
from engine.core.units import BOHR_ANGSTROM as BOHR_A
from engine.crystal.eos import fit
from engine.crystal.periodic import PeriodicDFT, cubic
from engine.materials.dataset import CACHE, T_E, grid_settings

ROOT = Path(__file__).resolve().parents[1]
V_RANGE = (0.88, 1.12)       # volume per atom, relative to the first guess
N_POINTS = 9


def _point(args):
    Z, a, k = args
    g = grid_settings([Z])
    key = hashlib.sha1(pickle.dumps((Z, round(a, 8), k, T_E, g["h"], g["xc_grid"]), protocol=4)).hexdigest()[:16]
    path = CACHE / "eos" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = time.time()
    r = PeriodicDFT(cubic("fcc", a, Z), kmesh=k, T_e=T_E, symmetry=True, **g).run()
    out = {"Z": Z, "a_bohr": a, "k": k, "T_e": T_E, **g, "energy_per_atom": r.energy / 4,
           "free_energy_per_atom": r.free_energy / 4, "converged": bool(r.converged), "seconds": time.time() - t}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def main(Z: int, k: int = 10, workers: int = 2, a_guess_bohr: float | None = None) -> None:
    # the first guess only sets where to look (a window of ±12% in volume); the answer is the
    # minimum of our own curve, and the window is re-centred if the minimum lands at its edge
    a_guess = a_guess_bohr or 6.8
    t = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    for _ in range(3):
        vols = np.linspace(*V_RANGE, N_POINTS) * a_guess ** 3
        jobs = [(Z, float(v ** (1 / 3)), k) for v in vols]
        with ctx.Pool(processes=workers, maxtasksperchild=2) as pool:
            for r in pool.imap(_point, jobs):
                print(f"a = {r['a_bohr']:.4f} bohr: E = {r['energy_per_atom']:.6f} Ha/atom "
                      f"(converged {r['converged']}, {r['seconds']:.0f}s)  {time.time() - t:.0f}s", flush=True)
                rows.append(r)
        good = [r for r in rows if r["converged"]]
        i = int(np.argmin([r["energy_per_atom"] for r in good]))
        a_sorted = sorted(r["a_bohr"] for r in good)
        a_min = good[i]["a_bohr"]
        if a_sorted[1] <= a_min <= a_sorted[-2]:
            break
        a_guess = a_min                                  # minimum at an edge: re-centre and extend
    good = sorted({r["a_bohr"]: r for r in rows if r["converged"]}.values(), key=lambda r: r["a_bohr"])
    res = fit([(r["a_bohr"] ** 3 / 4, r["energy_per_atom"], r["a_bohr"], True) for r in good])
    a0 = (4 * res["V0"]) ** (1 / 3)
    out = {"source": "scripts/eos.py: periodic LDA DFT, fcc 4-atom cell, float64", "Z": Z,
           "k": k, "T_e": T_E, **grid_settings([Z]), "points": len(good),
           "refused": len(rows) - len(good), "a0_bohr": a0, "a0_A": a0 * BOHR_A,
           "B_GPa": res["B0_GPa"], "B1": res["B1"],
           "rows": [{"a_bohr": r["a_bohr"], "E_Ha_per_atom": r["energy_per_atom"]} for r in good]}
    sym = ELEMENTS[Z].symbol.lower()
    (ROOT / "results" / f"{sym}_eos.json").write_text(json.dumps(out, indent=1))
    print(f"a0 = {a0 * BOHR_A:.4f} Å, B = {res['B0_GPa']:.1f} GPa, B' = {res['B1']:.2f}  "
          f"({len(good)} points)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), *(int(x) for x in sys.argv[2:4]))
