"""Label an element's crystal configurations with DFT (resumable): the first phase of a training set.

The configurations are dataset.initial_configurations(Z, a0) without the simple-cubic and
disordered ones, as for aluminium: the crystal at many volumes, strains and thermal
displacements, 8-atom cells, and bcc. The lattice constant a0 must be the element's own DFT value
(results/<el>_eos.json), never the measured one. The grid comes from dataset.GRID by itself, and
every label records the solver, precision, device and grid that produced it.

    uv run python scripts/label_crystal.py 29 6.704 torch     # copper, fp32 on the GPU, one at a time
    uv run python scripts/label_crystal.py 29 6.704 numpy 3   # float64, three workers

The MD snapshots that complete a set (dataset.md_snapshots) need a seed potential fitted to these
labels, and temperatures chosen for the element; that is a separate step.
"""

import sys
import time

from engine.core.elements import ELEMENTS
from engine.materials.dataset import CACHE, cache_key, initial_configurations, label_all


def main(Z: int, a0: float, solver: str = "torch", workers: int = 1) -> None:
    t = time.time()
    confs = [c for c in initial_configurations(Z, a0) if c["tag"] not in ("sc", "disordered")]
    have = sum((CACHE / "dft" / f"{cache_key(c)}.pkl").exists() for c in confs)
    print(f"{ELEMENTS[Z].symbol}: {len(confs)} configurations at a0 = {a0} bohr, {have} already labelled; "
          f"solver {solver}", flush=True)
    data = label_all(confs, workers=workers, solver=solver,
                     progress=lambda d, n: print(f"{d}/{n}  {time.time() - t:.0f}s", flush=True))
    print(f"done: {len(data)} converged labels, {len(confs) - len(data)} refused (SCF not converged)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "torch",
         int(sys.argv[4]) if len(sys.argv) > 4 else 1)
