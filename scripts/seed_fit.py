"""A seed potential for an element from its crystal labels, checked against the element's own DFT.

    uv run python scripts/seed_fit.py 29 6.704 [force_weight] [name] [structure] [split]
    uv run python scripts/seed_fit.py 26 5.4655 3.0 seed bcc        # iron

``name`` (default "seed") names the output: the same fit on crystal + MD labels is the final potential
(``... 3.0 final``).

The seed is what md_snapshots runs to find the configurations the metal visits when hot; those
are labelled with DFT in turn, and the final potential is fitted to everything. It is not a
result, so nothing here compares with experiment: the checks are against this engine's DFT
(the lattice constant a0 given, and each label). The short-range settings come from the data
(eam.short_range_for): the element's nuclear charge, and a splice below the shortest distance
the labels sampled.

``split`` picks the held-out seventh. "tag" (the default): about a seventh of each tag's labels, never a
tag's smallest or largest volume per atom, so every kind of configuration and the whole range of
compression is trained on and the test asks only for interpolation. "random": a seventh of all labels
at random, as copper's committed potentials and iron's first final potential were fitted; for iron it
held out all three most compressed cells, so the fit never saw strong compression (test 176 meV/atom).

Writes results/<el>_eam_<name>.npz (the model, pickled like aluminium's) and <el>_eam_<name>.json.
"""

import glob
import json
import os
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


def split_by_tag(data: list, seed: int = 1) -> tuple[list[int], list[int]]:
    """(test, train) indices: round(n/7) of each tag held out, chosen at random from the tag's labels
    other than its smallest and largest volume per atom (ties broken by the stable cache-key order)."""
    rng = np.random.default_rng(seed)
    test = []
    for tag in sorted({d["tag"] for d in data}):
        own = [i for i, d in enumerate(data) if d["tag"] == tag]
        own.sort(key=lambda i: np.prod(data[i]["cell"]) / len(data[i]["charges"]))    # cells are orthorhombic
        inner = own[1:-1]
        k = min(len(inner), round(len(own) / 7))
        test += [inner[j] for j in sorted(rng.choice(len(inner), size=k, replace=False))] if k else []
    test = sorted(test)
    return test, [i for i in range(len(data)) if i not in set(test)]


def main(Z: int, a0: float, w_force: float = 3.0, name: str = "seed", structure: str = "fcc", split: str = "tag") -> None:
    el = ELEMENTS[Z].symbol
    data = []
    for f in sorted(glob.glob(str(CACHE / "dft/*.pkl"))):
        d = pickle.loads(open(f, "rb").read())
        if set(d["charges"]) == {Z} and d["converged"] and d.get("kspacing", K_SPACING) == K_SPACING:
            data.append(d)
    data.sort(key=cache_key)                           # the split must not depend on directory order
    sr = short_range_for(data, Z)
    print(f"{el}: {len(data)} labels; short range {sr}", flush=True)
    if split == "random":
        idx = np.random.default_rng(1).permutation(len(data))
        nt = max(4, len(data) // 7)
        i_test, i_train = list(idx[:nt]), list(idx[nt:])
    else:
        i_test, i_train = split_by_tag(data)
    test, train = [data[i] for i in i_test], [data[i] for i in i_train]
    print(f"split {split}: {len(train)} train, {len(test)} test "
          f"({', '.join(sorted({d['tag'] for d in test}))})", flush=True)
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

    other = {"fcc": "bcc", "bcc": "fcc"}[structure]
    per_atom = {"fcc": 4, "bcc": 2}                    # atoms in the cubic cell: a^3 / n is the volume per atom

    def E_cell(a, kind=structure):
        c = cubic(kind, a, Z)
        return energy_forces(m, c.cell, c.positions)[0] / len(c.charges)

    aa = a0 * np.linspace(0.95, 1.05, 11)
    c = np.polyfit(aa, [E_cell(a) for a in aa], 4)
    r = np.roots(np.polyder(c))
    r = r[np.isreal(r)].real
    amin = float(r[np.argmin(np.abs(r - a0))])
    # B = V d2E/dV2 with V = a^3 / n per atom: (n / 9a) d2E/da2. Was written with fcc's n = 4 for
    # every structure, which doubled bcc iron's (288 GPa reported for ~144)
    B = per_atom[structure] * np.polyval(np.polyder(c, 2), amin) / (9 * amin) * GPA
    # the other cubic structure at the same volume per atom, above the ground one
    v_atom = amin ** 3 / per_atom[structure]
    dE_other = (E_cell((v_atom * per_atom[other]) ** (1 / 3), other) - E_cell(amin)) * HA_EV * 1000
    # the same difference in the DFT labels themselves: each structure's lowest cell
    lab = lambda tag: min((d["energy"] / len(d["charges"]) for d in data if d["tag"] == tag), default=np.nan)
    dE_other_dft = (lab(other) - lab(f"{structure}-volume")) * HA_EV * 1000
    out = {"element": el, "n_train": len(train), "n_test": len(test), "short_range": sr,
           "E_meV_train": etr, "F_eVA_train": ftr, "E_meV_test": ete, "F_eVA_test": fte,
           "a0_A": amin * BOHR_A, "a0_A_dft": a0 * BOHR_A, "B_GPa": float(B),
           "structure": structure, f"{other}_minus_{structure}_meV": float(dE_other),
           f"{other}_minus_{structure}_meV_dft_labels": float(dE_other_dft),
           "w_force": w_force, "split": split,
           "test_set": [{"tag": d["tag"], "key": cache_key(d)} for d in test], "tags": {t: sum(d["tag"] == t for d in data) for t in sorted({d["tag"] for d in data})}}
    print(json.dumps(out, indent=1), flush=True)
    target = ROOT / "results" / f"{el.lower()}_eam_{name}.npz"
    if target.exists() and os.environ.get("MATTER_SIM_OVERWRITE") != "1":
        # a committed potential is a record (copper's melting point was computed with its final one)
        print(f"{target.name} exists: not overwritten (set MATTER_SIM_OVERWRITE=1 to replace it)", flush=True)
        return
    m.save(target)
    (ROOT / "results" / f"{el.lower()}_eam_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 3.0,
         sys.argv[4] if len(sys.argv) > 4 else "seed", sys.argv[5] if len(sys.argv) > 5 else "fcc",
         sys.argv[6] if len(sys.argv) > 6 else "tag")
