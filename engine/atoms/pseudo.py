"""Norm-conserving pseudopotentials derived from this engine's own atoms.

Core electrons are tightly bound, chemically inert, and would need a grid
spacing of roughly 0.6/Z bohr to resolve. A pseudopotential replaces the
nucleus plus core with a smooth potential that acts on the valence electrons
*exactly* as the real atom does outside a core radius r_c.

Construction (Troullier & Martins, PRB 43, 1993, 1991):

1. Solve the all-electron atom (``radial.RadialAtom``, same LSDA as 3D).
2. For each valence channel l, replace the orbital inside r_c by
   u(r) = r^{l+1} exp(p(r)),  p = Σ_{k=0..6} c_{2k} r^{2k},  matching the true
   orbital's value and four derivatives at r_c, conserving the charge inside
   r_c (norm conservation), and with zero curvature of the potential at r = 0.
3. Invert the radial equation for the potential that has this orbital as its
   eigenstate at the *same* energy, then remove the valence electrons' own
   Hartree and exchange–correlation potential ("unscreening").
4. Recast in separable Kleinman–Bylander form for use on the 3D grid.

Nothing here is fitted: every number traces back to the all-electron atom.
The only choices are r_c (smoothness vs. transferability) and which channel
serves as the local potential; both are checked by :func:`verify`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

from .radial import AtomResult, RadialAtom, RadialGrid, hartree_potential
from ..electrons.xc import lda_xc

# Core radii (bohr), close to Troullier & Martins' published choices.
DEFAULT_RC = {3: 2.4, 4: 1.9, 5: 1.6, 6: 1.5, 7: 1.5, 8: 1.45, 9: 1.4, 10: 1.4}
R_GAUSS = 0.7  # width of the Gaussian that carries the long-range −Z_v/r in Fourier space


@dataclass
class Pseudopotential:
    Z: int
    Z_val: float
    l_local: int
    rc: dict[int, float]
    grid: RadialGrid = field(repr=False)
    v_ion: dict[int, np.ndarray] = field(repr=False)      # semilocal potentials
    u_ps: dict[int, np.ndarray] = field(repr=False)       # pseudo orbitals
    eps: dict[int, float] = field(repr=False)             # reference energies
    occ: dict[int, float] = field(repr=False)             # reference occupations
    kb_energy: dict[int, float] = field(repr=False)       # 1/⟨φ|δV|φ⟩
    beta: dict[int, np.ndarray] = field(repr=False)       # δV·u/r (radial projector)

    @property
    def v_local(self) -> np.ndarray:
        return self.v_ion[self.l_local]

    @property
    def nonlocal_channels(self) -> list[int]:
        return [l for l in self.beta]

    # ------------------------------------------------------ Fourier tables
    def local_short_range_q(self, q: np.ndarray) -> np.ndarray:
        """FT of V_loc(r) + Z_v erf(r/r_g)/r (short-ranged, smooth)."""
        r = self.grid.r
        dv = self.v_local + self.Z_val * erf(r / R_GAUSS) / r
        return _bessel_transform(self.grid, dv, 0, q)

    def projector_q(self, l: int, q: np.ndarray) -> np.ndarray:
        """∫ β_l(r) j_l(qr) r² dr · 4π."""
        return _bessel_transform(self.grid, self.beta[l], l, q)


def _bessel_transform(grid: RadialGrid, f: np.ndarray, l: int, q: np.ndarray) -> np.ndarray:
    r = grid.r
    out = np.empty_like(q, dtype=float)
    w = f * r * r * r * grid.dx  # f r² dr, dr = r dx
    for i0 in range(0, q.size, 256):
        qq = q[i0:i0 + 256][:, None]
        x = qq * r[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            if l == 0:
                j = np.where(x > 1e-8, np.sin(x) / x, 1.0 - x * x / 6)
            elif l == 1:
                j = np.where(x > 1e-4, np.sin(x) / x ** 2 - np.cos(x) / x, x / 3)
            else:
                raise NotImplementedError("l > 1 projectors")
        out[i0:i0 + 256] = 4 * np.pi * (j * w[None, :]).sum(axis=1)
    return out


def _core_valence_split(levels) -> tuple[list, list]:
    """Split occupied subshells at the largest energy gap (ratio): core below."""
    occ = sorted({(lv.n, lv.l): lv for lv in levels if lv.spin == 0 and lv.occupation > 1e-6}.values(),
                 key=lambda lv: lv.energy)
    if len(occ) < 2:
        return [], occ
    ratios = [occ[i].energy / occ[i + 1].energy for i in range(len(occ) - 1)]
    k = int(np.argmax(ratios))
    if ratios[k] < 4.0:
        return [], occ
    return occ[: k + 1], occ[k + 1:]


def _local_fit(r, y, i, half=12, deg=6):
    """Value and first two derivatives of y at r[i] from a local polynomial fit."""
    sl = slice(max(i - half, 0), i + half + 1)
    x = r[sl] - r[i]
    c = np.polyfit(x, y[sl], deg)
    p = np.poly1d(c)
    return p(0.0), p.deriv(1)(0.0), p.deriv(2)(0.0)


def _tm_channel(grid: RadialGrid, u: np.ndarray, V: np.ndarray, eps: float, l: int, rc: float):
    """Troullier–Martins pseudo-orbital and screened potential for one channel."""
    r = grid.r
    ic = int(np.searchsorted(r, rc))
    rc = r[ic]
    if u[ic] < 0:
        u = -u
    u0, u1, _ = _local_fit(r, u, ic)
    v0, v1, v2 = _local_fit(r, V, ic)
    p0 = math.log(u0 / rc ** (l + 1))
    p1 = u1 / u0 - (l + 1) / rc
    p2 = 2 * (v0 - eps) - p1 ** 2 - 2 * (l + 1) * p1 / rc
    p3 = 2 * v1 - 2 * p1 * p2 - 2 * (l + 1) * (p2 / rc - p1 / rc ** 2)
    p4 = 2 * v2 - 2 * p2 ** 2 - 2 * p1 * p3 - 2 * (l + 1) * (p3 / rc - 2 * p2 / rc ** 2 + 2 * p1 / rc ** 3)
    target = [p0, p1, p2, p3, p4]
    f_ae = (u * u * r)[: ic + 1]
    norm_ae = float(np.sum(f_ae) * grid.dx - 0.5 * grid.dx * (f_ae[0] + f_ae[-1]))

    powers = np.arange(0, 13, 2)  # 0,2,...,12

    def deriv_row(k):
        # d^k/dr^k of r^p at rc for each power p
        row = []
        for p in powers:
            c = 1.0
            for j in range(k):
                c *= (p - j)
            row.append(c * rc ** (p - k) if p - k >= 0 or c == 0 else 0.0)
        return np.array(row)

    D = np.array([deriv_row(k) for k in range(5)])  # 5 x 7

    def coeffs(c2):
        c4 = -c2 * c2 / (2 * l + 5)
        known = D[:, 1] * c2 + D[:, 2] * c4
        A = D[:, [0, 3, 4, 5, 6]]
        sol = np.linalg.solve(A, np.array(target) - known)
        return np.array([sol[0], c2, c4, sol[1], sol[2], sol[3], sol[4]])

    rin = r[: ic + 1]

    def norm_err(c2):
        c = coeffs(c2)
        p = sum(ck * rin ** pk for ck, pk in zip(c, powers))
        f = rin ** (2 * l + 2) * np.exp(2 * p) * rin
        val = float(np.sum(f) * grid.dx - 0.5 * grid.dx * (f[0] + f[-1]))
        return val - norm_ae

    grid_c2 = np.linspace(-15, 15, 3001)
    vals = np.array([norm_err(c) for c in grid_c2])
    roots = []
    for i in range(len(grid_c2) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            a, b = grid_c2[i], grid_c2[i + 1]
            for _ in range(80):
                m = 0.5 * (a + b)
                if norm_err(a) * norm_err(m) <= 0:
                    b = m
                else:
                    a = m
            roots.append(0.5 * (a + b))
    if not roots:
        raise RuntimeError(f"Troullier–Martins: no norm-conserving solution for l={l}, rc={rc:.2f}")
    c2 = min(roots, key=abs)
    c = coeffs(c2)

    p = sum(ck * r ** pk for ck, pk in zip(c, powers))
    dp = sum(ck * pk * r ** (pk - 1) for ck, pk in zip(c, powers) if pk > 0)
    ddp = sum(ck * pk * (pk - 1) * r ** (pk - 2) for ck, pk in zip(c, powers) if pk > 1)
    u_ps = np.where(np.arange(grid.n) <= ic, r ** (l + 1) * np.exp(np.minimum(p, 50)), u)
    v_scr = np.where(np.arange(grid.n) <= ic, eps + 0.5 * (ddp + dp * dp + 2 * (l + 1) * dp / r), V)
    return u_ps, v_scr, rc


def generate(Z: int, rc: dict[int, float] | None = None, l_local: int = 1) -> Pseudopotential:
    grid = RadialGrid(r_min=2e-6 / math.sqrt(Z), r_max=60.0, dx=0.004)
    ref: AtomResult = RadialAtom(Z, spin=0.0, grid=grid).solve()
    core, valence = _core_valence_split(ref.levels)
    core_nl = {(lv.n, lv.l) for lv in core}
    valence_nl = {(lv.n, lv.l) for lv in valence}
    if not core:
        raise ValueError(f"Z={Z} has no core electrons; use the bare nucleus")
    n_core = 2 * sum((2 * lv.l + 1) for lv in core)
    Z_val = Z - n_core
    V = ref.v_up  # spin-unpolarised reference: both spins identical
    rc_all = rc or {0: DEFAULT_RC[Z], 1: DEFAULT_RC[Z]}

    u_ps, v_scr, eps, occ, rcs = {}, {}, {}, {}, {}
    for l in (0, 1):
        lv_list = [lv for lv in ref.levels if lv.spin == 0 and lv.l == l and (lv.n, lv.l) not in core_nl]
        if not lv_list:
            raise RuntimeError(f"no bound l={l} valence level for Z={Z}")
        lv = min(lv_list, key=lambda x: x.energy)
        u_ps[l], v_scr[l], rcs[l] = _tm_channel(grid, lv.u, V, lv.energy, l, rc_all[l])
        eps[l] = lv.energy
        occ[l] = 2 * lv.occupation  # both spins

    r = grid.r
    rho_v = sum(occ[l] * u_ps[l] ** 2 for l in u_ps) / (4 * np.pi * r * r)
    vH = hartree_potential(grid, rho_v)
    _, vxc, _ = lda_xc(0.5 * rho_v, 0.5 * rho_v)
    v_ion = {l: v_scr[l] - vH - vxc for l in v_scr}
    # Beyond the core region every channel equals −Z_v/r (up to exponentially small terms).
    rho_core = ref.rho_up + ref.rho_dn - sum(
        2 * lv.occupation * lv.u ** 2 for lv in ref.levels if lv.spin == 0 and (lv.n, lv.l) in valence_nl
    ) / (4 * np.pi * r * r)
    q_out = 4 * np.pi * r * r * np.abs(rho_core)
    tail = max(max(rcs.values()), float(r[np.nonzero(q_out > 1e-10)[0].max()])) * 1.05
    for l in v_ion:
        v_ion[l] = np.where(r > tail, -Z_val / r, v_ion[l])

    kb, beta = {}, {}
    for l in v_ion:
        if l == l_local:
            continue
        dv = v_ion[l] - v_ion[l_local]
        D = grid.integrate(u_ps[l] ** 2 * dv)
        kb[l] = 1.0 / D
        beta[l] = dv * u_ps[l] / r
    return Pseudopotential(Z, float(Z_val), l_local, rcs, grid, v_ion, u_ps, eps, occ, kb, beta)


# ------------------------------------------------------------------ checks
def verify(pp: Pseudopotential, configs: list[dict] | None = None) -> list[dict]:
    """Compare pseudo-atom and all-electron atom for several configurations.

    Each config is {l: electrons} for the valence channels. Returns rows with
    AE and PS excitation energies (relative to the first config) and
    eigenvalues. Agreement across configurations = transferability.
    """
    configs = configs or [{0: pp.occ[0], 1: pp.occ[1]},
                          {0: pp.occ[0], 1: max(pp.occ[1] - 1, 0)},
                          {0: max(pp.occ[0] - 1, 0), 1: pp.occ[1] + 1}]
    grid = pp.grid
    rows = []
    core_occ_ae = _core_occupations(pp)
    for cfg in configs:
        ae_fixed, ps_fixed = dict(core_occ_ae), {}
        for l, n in cfg.items():
            idx_ae = 1 if l == 0 else 0  # 2s is the 2nd s level; 2p the 1st p level
            ae_fixed[(0, l, idx_ae)] = n / 2
            ae_fixed[(1, l, idx_ae)] = n / 2
            ps_fixed[(0, l, 0)] = n / 2
            ps_fixed[(1, l, 0)] = n / 2
        ne_val = sum(cfg.values())
        ae = RadialAtom(pp.Z, charge=int(round(pp.Z_val - ne_val)), grid=grid,
                        occupations=ae_fixed).solve()
        ps = RadialAtom(pp.Z, charge=int(round(pp.Z_val - ne_val)), grid=grid, v_external=pp.v_ion,
                        n_valence=pp.Z_val, occupations=ps_fixed).solve()
        rows.append({"config": cfg, "E_ae": ae.energy, "E_ps": ps.energy,
                     "eps_ae": {l: _level(ae, l, 1 if l == 0 else 0) for l in cfg},
                     "eps_ps": {l: _level(ps, l, 0) for l in cfg}})
    for row in rows:
        row["dE_ae"] = row["E_ae"] - rows[0]["E_ae"]
        row["dE_ps"] = row["E_ps"] - rows[0]["E_ps"]
    return rows


def _core_occupations(pp: Pseudopotential) -> dict:
    n_core_levels = int(round((pp.Z - pp.Z_val) / 2))  # first-row atoms: 1s² only
    return {(0, 0, 0): 1.0, (1, 0, 0): 1.0} if n_core_levels == 1 else {}


def _level(res: AtomResult, l: int, index: int) -> float:
    """Energy of the index-th (0-based) spin-up level of angular momentum l."""
    for lv in res.levels:
        if lv.spin == 0 and lv.l == l and lv.n == index + l + 1:
            return lv.energy
    return float("nan")
