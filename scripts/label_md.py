"""MD snapshots from an element's seed potential, labelled with DFT (resumable): the second phase of a
training set, after scripts/label_crystal.py and scripts/seed_fit.py.

    uv run python scripts/label_md.py 29 6.704 63.546 870,1450,2040 torch
    uv run python scripts/label_md.py 26 5.4655 55.845 <T1,T2,T3> torch bcc      # a bcc metal
    uv run python scripts/label_md.py 26 5.4655 55.845 2500,3000,3500 torch bcc final liquid 1.10,1.25 4

Optional, after the structure: the potential that runs the dynamics (default "seed":
results/<el>_eam_<name>.npz), a name for the snapshot set (default none: results/<el>_md_snapshots.json;
"liquid" writes <el>_md_snapshots_liquid.json), the cell volumes as multiples of a0's (default 1; each
other volume's tags get "-v<factor>"), and snapshots per temperature (default 6). A volume is set by
scaling a0, so the dynamics run at fixed volume: the cell, not the potential, decides the density.

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


def main(Z: int, a0: float, mass: float, temps, solver: str = "torch", structure: str = "fcc",
         potential: str = "seed", name: str = "", volumes=(1.0,), per_T: int = 6) -> None:
    el = ELEMENTS[Z].symbol.lower()
    out = ROOT / "results" / f"{el}_md_snapshots{'_' + name if name else ''}.json"
    if out.exists():
        confs = [dict(c, cell=np.array(c["cell"]), positions=np.array(c["positions"])) for c in json.loads(out.read_text())]
        print(f"{len(confs)} snapshots from {out.name}", flush=True)
    else:
        model = EAM.load(ROOT / "results" / f"{el}_eam_{potential}.npz")
        t = time.time()
        confs = []
        for v in volumes:
            got = md_snapshots(model, a0 * v ** (1 / 3), temps=temps, per_T=per_T, Z=Z, mass_amu=mass, structure=structure)
            confs += [c if v == 1.0 else dict(c, tag=f"{c['tag']}-v{v:.2f}") for c in got]
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
         sys.argv[5] if len(sys.argv) > 5 else "torch", sys.argv[6] if len(sys.argv) > 6 else "fcc",
         sys.argv[7] if len(sys.argv) > 7 else "seed", sys.argv[8] if len(sys.argv) > 8 else "",
         [float(x) for x in sys.argv[9].split(",")] if len(sys.argv) > 9 else (1.0,),
         int(sys.argv[10]) if len(sys.argv) > 10 else 6)
