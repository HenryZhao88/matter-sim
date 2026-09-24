"""Recompute ~5% of an element's DFT labels on the NumPy float64 path and report their consistency.

The labels may come from either path (dataset.label records which); the cross-checks prove the
paths agree across every kind of configuration in the set, which is what a fit needs. Selection
is per tag and deterministic, so any machine picks the same configurations, and each check is
cached in .cache/materials/dft_check/, so this can be stopped and rerun at will.

    uv run python scripts/cross_check.py 29 [workers]

Writes results/<symbol>_crosscheck.json (small: one row per check).
"""

import glob
import json
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

from engine.core.elements import ELEMENTS
from engine.materials.dataset import CACHE, NotConverged, consistency, cross_check, select_for_cross_check

BUDGET_MEV, BUDGET_F = 1.0, 1e-3        # per-label accuracy the fp32 path was accepted on


def _check(lab):
    try:
        cross_check(lab)
        return None
    except NotConverged as e:
        return str(e)


def main(Z: int, workers: int = 1) -> None:
    t = time.time()
    labels = [pickle.loads(open(f, "rb").read()) for f in glob.glob(str(CACHE / "dft/*.pkl"))]
    labels = [d for d in labels if set(d["charges"]) == {Z}]
    todo = select_for_cross_check(labels)
    print(f"{len(labels)} {ELEMENTS[Z].symbol} labels, {len(todo)} to cross-check", flush=True)
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=2) as pool:
        for i, err in enumerate(pool.imap(_check, todo)):
            print(f"{i + 1}/{len(todo)}  {time.time() - t:.0f}s" + (f"  refused: {err}" if err else ""), flush=True)
    rep = consistency(labels)
    rep["element"] = ELEMENTS[Z].symbol
    rep["solvers"] = sorted({d.get("solver", "numpy") for d in labels})
    over = [r for r in rep["rows"] if abs(r["dE_meV_per_atom"]) > BUDGET_MEV or r["max_dF_Ha_per_bohr"] > BUDGET_F]
    rep["over_budget"] = len(over)
    out = Path(__file__).resolve().parents[1] / "results" / f"{ELEMENTS[Z].symbol.lower()}_crosscheck.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"{rep['checked']} checked; worst {rep.get('max_abs_dE_meV_per_atom', float('nan')):.4f} meV/atom, "
          f"{rep.get('max_dF_Ha_per_bohr', float('nan')):.1e} Ha/bohr; {len(over)} over budget -> {out}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 1)
