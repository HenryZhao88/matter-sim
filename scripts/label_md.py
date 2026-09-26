"""MD snapshots from an element's seed potential, labelled with DFT (resumable): the second phase of a
training set, after scripts/label_crystal.py and scripts/seed_fit.py.

    uv run python scripts/label_md.py 29 6.704 63.546 870,1450,2040 torch

dataset.md_snapshots runs 8-atom cells with the seed potential (results/<el>_eam_seed.npz) and
samples them; the snapshots are written to results/<el>_md_snapshots.json before any labelling,
because the dynamics are chaotic and another machine (or another NumPy) would not regenerate the same
positions. A rerun reads that file instead of running the dynamics again. The temperatures are a
sampling choice, not a result; for copper they are aluminium's set as fractions of the melting point.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from engine.core.elements import ELEMENTS
from engine.materials.dataset import label_all, md_snapshots
from engine.materials.eam import EAM, pairs_with_images

ROOT = Path(__file__).resolve().parents[1]


def main(Z: int, a0: float, mass: float, temps, solver: str = "torch") -> None:
    el = ELEMENTS[Z].symbol.lower()
    out = ROOT / "results" / f"{el}_md_snapshots.json"
    if out.exists():
        confs = [dict(c, cell=np.array(c["cell"]), positions=np.array(c["positions"])) for c in json.loads(out.read_text())]
        print(f"{len(confs)} snapshots from {out.name}", flush=True)
    else:
        model = EAM.load(ROOT / "results" / f"{el}_eam_seed.npz")
        t = time.time()
        confs = md_snapshots(model, a0, temps=temps, Z=Z, mass_amu=mass)
        print(f"{len(confs)} snapshots in {time.time() - t:.0f}s", flush=True)
        out.write_text(json.dumps([dict(c, cell=np.asarray(c["cell"]).tolist(), positions=c["positions"].tolist())
                                   for c in confs]))
    for c in confs:
        r = np.linalg.norm(pairs_with_images(c["cell"], c["positions"], rc=6.0)[2], axis=1).min()
        print(f"  {c['tag']}: shortest distance {r:.2f} bohr", flush=True)
    t = time.time()
    data = label_all(confs, solver=solver, progress=lambda d, n: print(f"{d}/{n}  {time.time() - t:.0f}s", flush=True))
    print(f"done: {len(data)} converged labels, {len(confs) - len(data)} refused", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), [float(x) for x in sys.argv[4].split(",")],
         sys.argv[5] if len(sys.argv) > 5 else "torch")
