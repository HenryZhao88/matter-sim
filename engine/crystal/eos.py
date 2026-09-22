"""Equations of state: which structure wins, at what spacing, how stiff.

Energy against volume for each candidate structure, fitted to the Birch–Murnaghan form,
gives the equilibrium lattice constant, the bulk modulus, and (compared with a free
atom) the cohesive energy. The lowest curve is the structure the element picks.
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import os

import numpy as np
from scipy.optimize import curve_fit

from .periodic import PeriodicDFT, cubic

HA_PER_BOHR3_GPA = 29421.02648438959
ATOMS_PER_CELL = {"sc": 1, "bcc": 2, "fcc": 4}


def birch_murnaghan(V, E0, V0, B0, B1):
    x = (V0 / V) ** (2 / 3)
    return E0 + 9 * V0 * B0 / 16 * ((x - 1) ** 3 * B1 + (x - 1) ** 2 * (6 - 4 * x))


def _point(args):
    kind, a, Z, h, k = args
    r = PeriodicDFT(cubic(kind, a, Z), h=h, kmesh=k).run()
    n = ATOMS_PER_CELL[kind]
    return kind, a, r.energy / n, a ** 3 / n, r.converged


def scan(Z: int, kinds=("fcc", "bcc", "sc"), v_atom=(95, 128), n_points: int = 7, h=0.3, k=8,
         workers: int | None = None) -> dict:
    """E per atom against volume per atom, for each structure (parallel over cores)."""
    vols = np.linspace(v_atom[0], v_atom[1], n_points)
    jobs = [(kind, (v * ATOMS_PER_CELL[kind]) ** (1 / 3), Z, h, k) for kind in kinds for v in vols]
    out: dict = {kind: [] for kind in kinds}
    with cf.ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for kind, a, e, v, ok in pool.map(_point, jobs):
            out[kind].append((v, e, a, ok))
    return {kind: sorted(rows) for kind, rows in out.items()}


def fit(rows) -> dict:
    V = np.array([r[0] for r in rows])
    E = np.array([r[1] for r in rows])
    i = int(np.argmin(E))
    p0 = (E[i], V[i], 0.003, 4.0)
    (E0, V0, B0, B1), _ = curve_fit(birch_murnaghan, V, E, p0=p0, maxfev=20000)
    return {"E0": float(E0), "V0": float(V0), "B0_GPa": float(B0 * HA_PER_BOHR3_GPA), "B1": float(B1)}


def lattice_constant(kind: str, V0: float) -> float:
    return (V0 * ATOMS_PER_CELL[kind]) ** (1 / 3)


def free_atom_energy(Z: int) -> float:
    """Energy of the isolated pseudo-atom with its lowest-energy spin (same pseudopotential)."""
    from ..atoms.radial import RadialAtom
    from ..atoms.species import pseudopotential
    pp = pseudopotential(Z)
    best = None
    ne = int(round(pp.Z_val))
    for spin in range(ne % 2, ne + 1, 2):
        try:
            e = RadialAtom(Z, spin=spin, grid=pp.grid, v_external=pp.v_ion, n_valence=pp.Z_val).solve().energy
        except ValueError:
            continue
        best = e if best is None else min(best, e)
    return best


# ------------------------------------------------------------------ elastic constants
def _strained(args):
    kind, a, Z, strain, rotated, h, k, T_e = args
    from .periodic import Crystal, PeriodicDFT
    import numpy as np
    s = np.asarray(strain, float)
    if not rotated:
        c = cubic(kind, a, Z, strain=tuple(s))
    else:
        # FCC seen along [110]: an orthorhombic 2-atom cell (a/√2, a/√2, a). A stretch along x'
        # with an equal squeeze along y' is a pure shear of the cube axes.
        L = np.array([a / np.sqrt(2), a / np.sqrt(2), a]) * (1 + s)
        c = Crystal(L, [Z, Z], np.array([[0, 0, 0], [0.5, 0.5, 0.5]]) * L)
    n = len(c.charges)
    r = PeriodicDFT(c, h=h, kmesh=k, T_e=T_e).run()
    return float(r.energy / n), float(c.volume / n)


def elastic_constants(Z: int, a0: float, B_GPa: float, h=0.3, deltas=(-0.05, -0.025, 0.0, 0.025, 0.05),
                      k=14, T_e=0.01, workers: int | None = None) -> dict:
    """C11, C12 (from B and a volume-conserving tetragonal strain) and C44 (from a shear).

    Strains are large enough (±5%) that the strain energy (tens of meV per atom) stands well
    above k-point sampling noise (a few meV in a metal); a quartic fit isolates the quadratic,
    elastic term. The k-mesh is fixed in fractional coordinates, so it deforms with the cell
    and its error varies smoothly with strain.
    """
    kr = (round(k * 1.41), round(k * 1.41), k)
    tet = [("fcc", a0, Z, (d, d, (1 + d) ** -2 - 1), False, h, k, T_e) for d in deltas]
    shear = [("fcc", a0, Z, (d, -d, d * d / (1 - d * d)), True, h, kr, T_e) for d in deltas]
    with cf.ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        tet_E = list(pool.map(_strained, tet))
        sh_E = list(pool.map(_strained, shear))
    d = np.array(deltas)
    v_t = tet_E[len(d) // 2][1]
    v_s = sh_E[len(d) // 2][1]
    ct = np.polyfit(d, [e for e, _ in tet_E], 4)[2] / v_t      # E/V ≈ 3 (C11 − C12) δ² + O(δ³)
    cs = np.polyfit(d, [e for e, _ in sh_E], 4)[2] / v_s       # E/V ≈ 2 C44 δ² + O(δ³)
    c11_minus_c12 = ct / 3 * HA_PER_BOHR3_GPA
    c44 = cs / 2 * HA_PER_BOHR3_GPA
    c11 = B_GPa + 2 / 3 * c11_minus_c12
    c12 = B_GPa - 1 / 3 * c11_minus_c12
    return {"C11": c11, "C12": c12, "C44": c44}
