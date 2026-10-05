"""Aluminium's latent heat of fusion against box size, at the large-box melting point, on a GPU.

The same measurement as experiments.latent_heat (enthalpy per atom of liquid minus solid, both held
at T and zero pressure; the solid checked still to be a solid), at T = the 96 000-atom coexistence
melting point (results/al_melting_96000.json), for boxes of n³ fcc cells. The lattice constant at T
comes from the thermal curve in results/al_results.json, as for the melting runs. Each size is
cached, so the run resumes.

    uv run --extra gpu python scripts/latent_large.py 4 10 20 30

Writes results/al_latent_scale.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from engine.core.accel import have_cuda
from engine.materials.eam import EAM
from engine.materials.experiments import latent_heat

ROOT = Path(__file__).resolve().parents[1]


def main(sizes) -> None:
    model = EAM.load(ROOT / "results" / "al_eam_aluminium.npz")
    T = json.loads((ROOT / "results" / "al_melting_96000.json").read_text())["T_melt"]
    thermal = json.loads((ROOT / "results" / "al_results.json").read_text())["thermal"]
    t, a = np.array([(r["T"], r["a_A"]) for r in thermal]).T
    a_T = float(np.polyval(np.polyfit(t, a, 2), T))
    engine = "torch" if have_cuda() else "numpy"
    cache = ROOT / ".cache" / "materials" / "al_latent_scale"
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in sizes:
        f = cache / f"n{k}_T{T:.4f}.json"
        if f.exists():
            row = json.loads(f.read_text())
        else:
            s = time.time()
            r = latent_heat(model, T, a_T, n=(k, k, k), engine=engine)
            row = {"n": k, "atoms": 4 * k ** 3, "T": T, **r, "seconds": time.time() - s, "engine": engine}
            f.write_text(json.dumps(row))
        rows.append(row)
        L = "melted solid: withheld" if row.get("latent_eV") is None else f"{row['latent_eV'] * 1000:.2f} meV/atom"
        dV = "-" if row.get("dV_melt_frac") is None else f"{row['dV_melt_frac'] * 100:.2f}%"
        print(f"{row['atoms']:7d} atoms: latent heat {L}, melting expansion {dV}, {row['seconds']:.0f} s", flush=True)
    out = {"T": T, "a_T_A": a_T, "engine": engine, "rows": rows,
           "method": "experiments.latent_heat at the 96 000-atom coexistence melting point, boxes of n^3 fcc cells"}
    (ROOT / "results" / "al_latent_scale.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [4, 10, 20, 30])
