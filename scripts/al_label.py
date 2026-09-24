"""Label the aluminium training set with DFT (resumable, memory-bounded).

Every configuration is cached by content, so this can be stopped and restarted at will: it
recomputes only what is missing. The dataset is the crystal in many states (volumes, strains,
thermal disorder, a supercell, bcc) plus snapshots from hot molecular dynamics run with a
potential fitted to the crystal data alone — configurations the metal actually visits, rather
than random packings.

    uv run python scripts/al_label.py [workers]

Workers are bounded by memory, not cores: each holds the projectors for every k-point, about
1.5 GB at this mesh. Three is safe on 16 GB alongside a browser.
"""

import glob
import pickle
import sys
import time

from engine.core.units import angstrom_to_bohr
from engine.materials.dataset import CACHE, initial_configurations, label_all, md_snapshots
from engine.materials.eam import fit, refine

A0_ANGSTROM = 3.955          # the DFT lattice constant this engine computed (engine/crystal/eos.py)


def main(workers: int = 3) -> None:
    t = time.time()
    a0 = angstrom_to_bohr(A0_ANGSTROM)
    cached = [pickle.loads(open(f, "rb").read()) for f in glob.glob(str(CACHE / "dft/*.pkl"))]
    crystal = [d for d in cached if d.get("tag") in ("fcc-volume", "fcc-strain", "fcc-thermal")]
    if len(crystal) < 12:
        raise SystemExit("label the crystal configurations first: they seed the snapshot dynamics")
    seed_model = refine(fit(crystal, iters=6000), crystal, w_force=3)
    snaps = md_snapshots(seed_model, a0)
    confs = [c for c in initial_configurations(13, a0) if c["tag"] not in ("sc", "disordered")] + snaps
    print(f"{len(confs)} configurations, {len(cached)} already cached", flush=True)
    data = label_all(confs, workers=workers,
                     progress=lambda d, n: print(f"{d}/{n}  {time.time() - t:.0f}s", flush=True))
    print(f"done: {len(data)} converged labels, {len(confs) - len(data)} refused (SCF not converged)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
