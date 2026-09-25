"""Pseudopotential candidates and whether a solid survives them (a check the free atom cannot make).

    uv run python scripts/pp_check.py candidates Z
        every candidate pseudo.generate's ghost-avoidance loop tries for Z (local channel x radius
        scale), with its ghost and free-atom transferability checks; each is pickled to
        .cache/pp_check/ for the next command. Nothing is written to .cache/pseudo.
    uv run python scripts/pp_check.py eos Z kind h k default|<pickle> V1 V2 ...
        energy per atom at each volume (bohr³/atom) and the successive differences: a sound
        potential gives a convex curve with its minimum inside the scan.

Why it exists: the generator accepts a candidate on free-atom tests alone. For iron it chose an
s-local potential with r_s = 3.45 bohr whose bcc E(V) is concave with no minimum (collapse).
"""

import json
import math
import pickle
import sys
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "pp_check"


def candidates(Z: int) -> None:
    from engine.atoms import pseudo
    from engine.atoms.radial import RadialAtom, RadialGrid
    CACHE.mkdir(parents=True, exist_ok=True)
    grid = RadialGrid(r_min=2e-6 / math.sqrt(Z), r_max=60.0, dx=0.004)
    ref = RadialAtom(Z, spin=0.0, grid=grid).solve()
    if not ref.converged:
        ref = pseudo._lowest_configuration(Z, grid, ref)
    base = pseudo.DEFAULT_RC[Z]
    base = base if isinstance(base, dict) else {0: base, 1: base, 2: base}
    for scale in (1.0, 1.25, 1.5):
        for cand in (1, 0, 2):
            rcs = dict(base)
            rcs[cand] = base[cand] * scale
            rec = dict(Z=Z, scale=scale, l_local=cand, rc={l: round(v, 2) for l, v in rcs.items()})
            try:
                pp = pseudo._build(Z, ref, grid, rcs, cand)
            except Exception as e:
                print(json.dumps(dict(rec, error=repr(e)[:80])), flush=True)
                continue
            ghost = max((abs(a - b) for a, b in pseudo.ghost_check(pp).values()), default=0.0)
            rec.update(ghost_Ha=round(ghost, 4), ghost_free=ghost < pseudo.GHOST_TOL,
                       transfer_meV=round(pseudo.transfer_error(pp) * 27211.386, 1),
                       kb={l: round(float(v), 2) for l, v in pp.kb_energy.items()})
            path = CACHE / f"Z{Z}_s{scale}_l{cand}.pkl"
            path.write_bytes(pickle.dumps(pp))
            print(json.dumps(dict(rec, pickle=str(path))), flush=True)


def eos(Z: int, kind: str, h: float, k: int, which: str, vols: list[float]) -> None:
    from engine.atoms import species
    if which != "default":
        pp = pickle.loads(Path(which).read_bytes())
        orig = species.pseudopotential
        species.pseudopotential = lambda z: pp if z == Z else orig(z)
    from engine.crystal.periodic import PeriodicDFT, cubic
    n = {"sc": 1, "bcc": 2, "fcc": 4}[kind]
    E = []
    for v in vols:
        a = (n * v) ** (1 / 3)
        r = PeriodicDFT(cubic(kind, a, Z), h=h, kmesh=k, smearing="mp", T_e=0.01).run(max_iter=100)
        E.append(r.energy / n)
        print(json.dumps(dict(Z=Z, pp=which, v=v, a_A=round(a * 0.529177, 4), E=r.energy / n,
                              converged=r.converged, seconds=round(r.seconds))), flush=True)
    print("successive dE (mHa/atom):", [round(float(E[i + 1] - E[i]) * 1000, 2) for i in range(len(E) - 1)])


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    if cmd == "candidates":
        candidates(int(a[0]))
    else:
        eos(int(a[0]), a[1], float(a[2]), int(a[3]), a[4], [float(x) for x in a[5:]])
