"""Iron's DFT grid, chosen like copper's: which (h, xc_grid) holds what the training set needs.

Ferromagnetic bcc iron (spin-polarised, starting moment 3 μB/atom), 2-atom cell, 6³ k. For each
grid: the half-step egg-box at V = 76 bohr³/atom, the energy difference between V = 68 and 84
(what an equation of state and strained cells measure), and FM − NM and the moment at V = 76.
Every run is cached in .cache/fe_grid/, so the scan resumes. Compare each row with the finest.

    uv run python scripts/fe_grid.py 0.30:1 0.30:2 0.24:1 0.24:2 0.20:2 0.16:2
"""

import json
import sys
from pathlib import Path

from engine.crystal.periodic import Crystal, PeriodicDFT, cubic

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "fe_grid"
HA_MEV = 27211.386


def run(h, xc, v, spin=True, shift=0.0):
    key = CACHE / f"h{h}_x{xc}_v{v}_s{int(spin)}_d{shift}.json"
    if key.exists():
        return json.loads(key.read_text())
    base = cubic("bcc", (2 * v) ** (1 / 3), 26)
    kw = dict(h=h, kmesh=6, smearing="mp", T_e=0.01, xc_grid=xc)
    if spin:
        kw.update(spin=True, moments=3.0)
    step = base.cell[0] / PeriodicDFT(base, **kw).N[0] if shift else 0.0
    d = PeriodicDFT(Crystal(base.cell, base.charges, base.positions + [shift * step, 0, 0]), **kw)
    r = d.run(max_iter=100)
    rec = dict(E=r.energy / 2, M=r.moment / 2, converged=r.converged, grid=list(d.N), seconds=r.seconds)
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(rec))
    return rec


def main(specs):
    for spec in specs:
        h, xc = float(spec.split(":")[0]), int(spec.split(":")[1])
        fm = run(h, xc, 76.0)
        shifted = run(h, xc, 76.0, shift=0.5)
        small, large = run(h, xc, 68.0), run(h, xc, 84.0)
        nm = run(h, xc, 76.0, spin=False)
        ok = all(x["converged"] for x in (fm, shifted, small, large, nm))
        print(json.dumps(dict(h=h, xc_grid=xc, grid=fm["grid"], converged=ok,
                              eggbox_meV=(shifted["E"] - fm["E"]) * HA_MEV,
                              dE_68_84_meV=(small["E"] - large["E"]) * HA_MEV,
                              fm_minus_nm_meV=(fm["E"] - nm["E"]) * HA_MEV, M=fm["M"], E_fm=fm["E"],
                              seconds=round(sum(x["seconds"] for x in (fm, shifted, small, large, nm))))), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
