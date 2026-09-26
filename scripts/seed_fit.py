"""A seed potential for an element from its crystal labels, checked against the element's own DFT.

    uv run python scripts/seed_fit.py 29 6.704 [force_weight]

The seed is what md_snapshots runs to find the configurations the metal visits when hot; those
are labelled with DFT in turn, and the final potential is fitted to everything. It is not a
result, so nothing here compares with experiment: the checks are against this engine's DFT
(the lattice constant a0 given, and each label). The short-range settings come from the data
(eam.short_range_for): the element's nuclear charge, and a splice below the shortest distance
the labels sampled.

Writes results/<el>_eam_seed.npz (the model, pickled like aluminium's) and <el>_eam_seed.json.
"""

import glob
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from engine.core.elements import ELEMENTS
from engine.crystal.periodic import cubic
from engine.materials.dataset import CACHE, K_SPACING, cache_key
from engine.materials.eam import energy_forces, fit, refine, short_range_for

HA_EV, BOHR_A, GPA = 27.211386245988, 0.529177210903, 29421.02648438959
ROOT = Path(__file__).resolve().parents[1]


def main(Z: int, a0: float, w_force: float = 3.0) -> None:
    el = ELEMENTS[Z].symbol
    data = []
    for f in sorted(glob.glob(str(CACHE / "dft/*.pkl"))):
        d = pickle.loads(open(f, "rb").read())
        if set(d["charges"]) == {Z} and d["converged"] and d.get("kspacing", K_SPACING) == K_SPACING:
            data.append(d)
    data.sort(key=cache_key)                           # the split must not depend on directory order
    sr = short_range_for(data, Z)
    print(f"{el}: {len(data)} labels; short range {sr}", flush=True)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(data))
    nt = max(4, len(data) // 7)
    test, train = [data[i] for i in idx[:nt]], [data[i] for i in idx[nt:]]
    t = time.time()
    m = fit(train, iters=12000, e_scale_eV=0.3, short_range=sr,
            log=lambda i, l: print("fit", i, f"{l:.3g}", f"{time.time() - t:.0f}s", flush=True))
    m = refine(m, train, w_force=w_force, e_scale_eV=0.3, log=print)

    def errs(ds):
        de, df = [], []
        for d in ds:
            E, F = energy_forces(m, d["cell"], d["positions"])
            de.append((E - d["energy"]) / len(d["positions"]))
            df.append((F - d["forces"]).ravel())
        return (float(np.sqrt(np.mean(np.square(de))) * HA_EV * 1000),
                float(np.sqrt(np.mean(np.concatenate(df) ** 2)) * HA_EV / BOHR_A))

    etr, ftr = errs(train)
    ete, fte = errs(test)
    print(f"train E {etr:.1f} meV/atom F {ftr:.3f} eV/A | test E {ete:.1f} F {fte:.3f}", flush=True)

    def E_cell(a, kind="fcc"):
        c = cubic(kind, a, Z)
        return energy_forces(m, c.cell, c.positions)[0] / len(c.charges)

    aa = a0 * np.linspace(0.95, 1.05, 11)
    c = np.polyfit(aa, [E_cell(a) for a in aa], 4)
    r = np.roots(np.polyder(c))
    r = r[np.isreal(r)].real
    amin = float(r[np.argmin(np.abs(r - a0))])
    B = 4 * np.polyval(np.polyder(c, 2), amin) / (9 * amin) * GPA
    dE_bcc = (E_cell((amin ** 3 / 2) ** (1 / 3), "bcc") - E_cell(amin)) * HA_EV * 1000
    # the same difference in the DFT labels themselves: bcc and fcc cells nearest each one's minimum
    lab = lambda tag: min((d["energy"] / len(d["charges"]) for d in data if d["tag"] == tag), default=np.nan)
    dE_bcc_dft = (lab("bcc") - lab("fcc-volume")) * HA_EV * 1000
    out = {"element": el, "n_train": len(train), "n_test": len(test), "short_range": sr,
           "E_meV_train": etr, "F_eVA_train": ftr, "E_meV_test": ete, "F_eVA_test": fte,
           "a0_A": amin * BOHR_A, "a0_A_dft": a0 * BOHR_A, "B_GPa": float(B),
           "bcc_minus_fcc_meV": float(dE_bcc), "bcc_minus_fcc_meV_dft_labels": float(dE_bcc_dft),
           "w_force": w_force}
    print(json.dumps(out, indent=1), flush=True)
    m.save(ROOT / "results" / f"{el.lower()}_eam_seed.npz")
    (ROOT / "results" / f"{el.lower()}_eam_seed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 3.0)
