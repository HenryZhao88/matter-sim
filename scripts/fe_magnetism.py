"""Iron: does it choose to be a magnet, and which crystal does it pick? (resumable)

Energy against volume for bcc and fcc iron, each without spin polarisation and with a starting
moment (ferromagnetic for bcc and fcc, layered antiferromagnetic for fcc). Nothing tells the code
that iron is magnetic or that it is bcc: the starting moment is only a push, which a non-magnetic
metal gives back (aluminium does). Each point is cached as it finishes, so this can be stopped and
restarted at will. Writes results/fe_magnetism.json.

    uv run python scripts/fe_magnetism.py [h] [k_bcc] [workers] [phase,phase,...]

Grid h = 0.20 bohr, xc_grid = 2 (scripts/fe_grid.py, v5 potential): within 1.1 meV/atom of h = 0.16 on
volume energy, FM − NM and egg-box, and 1e-4 μB on the moment. k_bcc = 8 is within ~3 meV and 0.02 μB
of 12. Phases can be split across machines; each machine caches its own points.

Memory is small (2- and 4-atom cells); time is not: 2 spins x a transition metal's fine grid.
"""

import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

import numpy as np

from engine.crystal.eos import ATOMS_PER_CELL, fit, lattice_constant
from engine.crystal.periodic import PeriodicDFT, cubic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fe_magnetism.json"
CACHE = ROOT / ".cache" / "fe_magnetism"

Z = 26
V_ATOM = np.linspace(60.0, 84.0, 7)          # bohr³ per atom: a wide scan; the fit must land inside it
# phase: (structure, starting moment per atom in μB, or None for no spin polarisation)
PHASES = {
    "bcc-nonmagnetic": ("bcc", None),
    "bcc-ferromagnetic": ("bcc", [3.0, 3.0]),
    "fcc-nonmagnetic": ("fcc", None),
    "fcc-ferromagnetic": ("fcc", [3.0] * 4),
    "fcc-antiferromagnetic": ("fcc", [3.0, -3.0, -3.0, 3.0]),   # up planes z = 0, down planes z = ½
}


def kmesh(kind: str, k_bcc: int) -> int:
    """The same k-point spacing for both structures at equal volume per atom: the conventional fcc
    cell is 2^(1/3) longer than the bcc one. Fixed per structure, never per volume, so the sampling
    error is the same smooth function along each energy curve."""
    return k_bcc if kind == "bcc" else max(1, round(k_bcc / 2 ** (1 / 3)))


def point(args):
    phase, v, h, k, xc = args
    kind, moments = PHASES[phase]
    k = kmesh(kind, k)
    key = CACHE / f"{phase}_v{v:.3f}_h{h}_x{xc}_k{k}.json"
    if key.exists():
        return json.loads(key.read_text())
    n = ATOMS_PER_CELL[kind]
    a = lattice_constant(kind, v)
    kw = dict(h=h, kmesh=k, smearing="mp", T_e=0.01, xc_grid=xc)
    dft = PeriodicDFT(cubic(kind, a, Z), spin=moments is not None, moments=moments, **kw)
    r = dft.run(max_iter=100)
    per_atom = None
    if r.rho_spin is not None:
        # moment on each atom: ∫ (ρ↑ − ρ↓) inside a sphere of half the nearest-neighbour distance
        m = (r.rho_spin[0] - r.rho_spin[1]) * dft.dV
        idx = np.indices(dft.N).reshape(3, -1).T * (dft.c.cell / np.array(dft.N))
        rad = 0.5 * a * (np.sqrt(3) / 2 if kind == "bcc" else 1 / np.sqrt(2))
        per_atom = []
        for R in dft.c.positions:
            d = (idx - R + dft.c.cell / 2) % dft.c.cell - dft.c.cell / 2
            per_atom.append(float(m.ravel()[np.linalg.norm(d, axis=1) < rad].sum()))
    rec = dict(phase=phase, v_atom=v, a=a, h=h, xc_grid=xc, k=k, E_atom=r.energy / n, converged=r.converged,
               iterations=r.iterations, seconds=r.seconds, grid=list(dft.N), n_k=len(dft.kpts),
               moment_atom=r.moment / n, abs_moment_atom=r.abs_moment / n, sphere_moments=per_atom,
               top_band_occupation=r.notes.get("top_band_occupation"))
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps(rec))
    return rec


def main(h: float = 0.20, k: int = 8, workers: int = 1, phases=None, xc: int = 2) -> None:
    t0 = time.time()
    phases = phases or list(PHASES)
    jobs = [(p, round(float(v), 3), h, k, xc) for p in phases for v in V_ATOM]
    rows = [json.loads(f.read_text()) for f in CACHE.glob(f"*_h{h}_x{xc}_*.json")] if CACHE.exists() else []
    if OUT.exists():                  # phases another machine computed and committed, same grid
        old = json.loads(OUT.read_text())
        if old.get("h") == h and old.get("xc_grid") == xc and old.get("k") == k:
            seen = {(r["phase"], r["v_atom"]) for r in rows}
            rows += [r for r in old["points"] if (r["phase"], r["v_atom"]) not in seen]
    rows = [r for r in rows if r["phase"] not in phases]      # other phases, kept in the file
    with cf.ProcessPoolExecutor(max_workers=workers) as pool:
        for rec in pool.map(point, jobs):
            rows.append(rec)
            print(f"{rec['phase']:22s} V={rec['v_atom']:6.2f}  E={rec['E_atom']:.6f}  "
                  f"M={rec['moment_atom']:+.3f}  conv={rec['converged']}  {rec['seconds']:.0f}s  "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            write(rows, h, k, xc)        # after every point: a stopped run still leaves its finished phases
    print(json.dumps(write(rows, h, k, xc), indent=1))
    print(f"wrote {OUT}  ({time.time() - t0:.0f}s)")


def write(rows, h, k, xc) -> dict:
    fits = {}
    for p in PHASES:
        pr = sorted((r for r in rows if r["phase"] == p and r["converged"]), key=lambda r: r["v_atom"])
        if len(pr) < 5:
            continue
        try:
            f = fit([(r["v_atom"], r["E_atom"]) for r in pr])
        except RuntimeError as e:                 # curve_fit gave up: report, keep the points
            fits[p] = {"error": str(e), "n_points": len(pr)}
            continue
        f["n_points"] = len(pr)
        f["a0"] = lattice_constant(PHASES[p][0], f["V0"])
        f["inside_scan"] = bool(pr[0]["v_atom"] < f["V0"] < pr[-1]["v_atom"])
        f["moment_at_V0"] = float(np.interp(f["V0"], [r["v_atom"] for r in pr], [r["abs_moment_atom"] for r in pr]))
        fits[p] = f
    kmeshes = {kind: kmesh(kind, k) for kind in ("bcc", "fcc")}
    OUT.write_text(json.dumps({"h": h, "xc_grid": xc, "k": k, "kmesh": kmeshes, "smearing": "mp 0.01 Ha", "points": rows,
                               "fits": fits}, indent=1))
    return fits


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.20, int(a[1]) if len(a) > 1 else 8, int(a[2]) if len(a) > 2 else 1,
         a[3].split(",") if len(a) > 3 else None)
