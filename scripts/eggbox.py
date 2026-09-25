"""Egg-box test: slide a perfect crystal by half a grid step and see whether the energy moves.

No physical energy can change under a rigid translation, so any change is discretisation error.
A half step along x keeps every atom on a mirror plane of the grid, so forces stay zero by
symmetry: the energy is the signal. One JSON line per grid spacing.

    uv run python scripts/eggbox.py --Z 29 --kind fcc --a 6.83 --h 0.30 0.26 0.22
    uv run python scripts/eggbox.py --Z 29 ... --core-factor 1     # prototype softer partial core
    uv run python scripts/eggbox.py --Z 29 ... --no-core           # diagnostic only, not physics

``--core-factor f`` rebuilds the pseudopotential with the partial core kept outside the radius where
the core density falls to f x the valence density (pseudo._partial_core uses 2), and prints the
engine's own ghost and transferability checks for it. Nothing here is written to .cache/pseudo.
"""

import argparse
import json

import numpy as np

from engine.atoms import pseudo, species


def soft_core_pseudopotential(Z: int, factor: float):
    orig = pseudo._partial_core
    pseudo._partial_core = lambda grid, rho_core, rho_val: orig(grid, rho_core, rho_val * factor / 2)
    try:
        pp = pseudo.generate(Z)
    finally:
        pseudo._partial_core = orig
    ghosts = pseudo.ghost_check(pp)
    print(json.dumps(dict(Z=Z, core_factor=factor, core_electrons=float(pp.core_density_q(np.array([0.0]))[0]),
                          ghost_free=all(abs(a - b) < pseudo.GHOST_TOL for a, b in ghosts.values()),
                          transfer_meV=float(pseudo.worst_transfer(pseudo.verify(pp)) * 27211.386))), flush=True)
    return pp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--Z", type=int, required=True)
    ap.add_argument("--kind", default="fcc")
    ap.add_argument("--a", type=float, required=True, help="lattice constant, bohr")
    ap.add_argument("--h", type=float, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--T_e", type=float, default=0.01)
    ap.add_argument("--core-factor", type=float)
    ap.add_argument("--no-core", action="store_true")
    ap.add_argument("--spin-moment", type=float, help="spin-polarised, with this starting moment per atom")
    args = ap.parse_args()

    if args.core_factor is not None:
        pp = soft_core_pseudopotential(args.Z, args.core_factor)
        orig = species.pseudopotential
        species.pseudopotential = lambda Z: pp if Z == args.Z else orig(Z)
    from engine.crystal.periodic import Crystal, PeriodicDFT, cubic

    base = cubic(args.kind, args.a, args.Z)
    n = len(base.charges)
    kw = dict(kmesh=args.k, T_e=args.T_e)
    if args.spin_moment is not None:
        kw.update(spin=True, moments=args.spin_moment)
    for h in args.h:
        runs = []
        for shift in (0.0, 0.5):
            d = PeriodicDFT(base, h=h, **kw)
            step = base.cell[0] / d.N[0]
            d = PeriodicDFT(Crystal(base.cell, base.charges, base.positions + [shift * step, 0, 0]), h=h, **kw)
            if args.no_core:
                d.core_q = {}
                d._set_local()
            runs.append((d, d.run(max_iter=100)))
        (d0, r0), (_, r1) = runs
        print(json.dumps(dict(
            Z=args.Z, kind=args.kind, a=args.a, h=h, grid=list(d0.N), k=args.k, core_factor=args.core_factor,
            no_core=args.no_core, converged=[r0.converged, r1.converged],
            dE_meV_atom=(r1.energy - r0.energy) * 27211.386 / n,
            components_meV_atom={c: (r1.components[c] - r0.components[c]) * 27211.386 / n for c in r0.components},
            seconds=[r0.seconds, r1.seconds])), flush=True)


if __name__ == "__main__":
    main()
