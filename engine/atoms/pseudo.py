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

from .radial import AtomResult, RadialAtom, RadialGrid, hartree_potential, scattering_state
from ..electrons.xc import lda_xc

# Core radii (bohr), close to Troullier & Martins' published choices.
DEFAULT_RC = {3: 2.4, 4: 1.9, 5: 1.6, 6: 1.5, 7: 1.5, 8: 1.45, 9: 1.4, 10: 1.4,
              11: 2.6, 12: 2.4, 13: 2.3, 14: 2.0, 15: 1.9, 16: 1.8, 17: 1.7, 18: 1.6,
              19: 3.4, 20: 3.0, 31: 2.4, 32: 2.2, 33: 2.1, 34: 2.0, 35: 1.9, 36: 1.8}
# d-block: s and p channels at one radius, the compact 3d orbital at a smaller one
for _Z, (_sp, _d) in zip(range(21, 31), [(2.6, 2.3), (2.5, 2.2), (2.5, 2.1), (2.4, 2.0), (2.4, 2.0),
                                          (2.3, 1.9), (2.3, 1.9), (2.3, 1.8), (2.2, 1.8), (2.2, 1.8)]):
    DEFAULT_RC[_Z] = {0: _sp, 1: _sp, 2: _d}
VALENCE_WINDOW = 0.35   # Ha: an inner shell this close below the outer s level is chemically active
NLCC_FROM = 19          # nonlinear core correction for Z ≥ 19 (large, soft cores)
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
    core: list[tuple[int, int]] = field(default_factory=list)       # (n, l) core subshells
    valence_n: dict[int, int] = field(default_factory=dict)         # l → n of the valence level
    rho_core: np.ndarray | None = field(default=None, repr=False)  # partial core density (NLCC)

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

    def core_density_q(self, q: np.ndarray) -> np.ndarray:
        """FT of the partial core density (zero when there is no core correction)."""
        if self.rho_core is None:
            return np.zeros_like(q, dtype=float)
        return _bessel_transform(self.grid, self.rho_core, 0, q)


def real_harmonics_k(l: int, n: list[np.ndarray]) -> list[np.ndarray]:
    """Real spherical harmonics of the unit vector n = (nx, ny, nz), times (−i)^l — the angular
    factors of a projector β_l(r) Y_lm(r̂) in Fourier space."""
    x, y, z = n
    if l == 0:
        return [np.full(np.shape(x), 1 / math.sqrt(4 * math.pi)) + 0j]
    if l == 1:
        c = math.sqrt(3 / (4 * math.pi))
        return [-1j * c * x, -1j * c * y, -1j * c * z]
    if l == 2:
        c = math.sqrt(15 / (4 * math.pi))
        return [-(c * x * y) + 0j, -(c * y * z) + 0j, -(c * x * z) + 0j,
                -(math.sqrt(5 / (16 * math.pi)) * (3 * z * z - (x * x + y * y + z * z))) + 0j,
                -(0.5 * c * (x * x - y * y)) + 0j]
    raise NotImplementedError(l)


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
            elif l == 2:
                j = np.where(x > 1e-2, (3 / x ** 3 - 1 / x) * np.sin(x) - 3 * np.cos(x) / x ** 2,
                             x * x / 15 - x ** 4 / 210)
            else:
                raise NotImplementedError("l > 2 projectors")
        out[i0:i0 + 256] = 4 * np.pi * (j * w[None, :]).sum(axis=1)
    return out


def _period(n: int, l: int) -> int:
    """Row of the periodic table an orbital belongs to: n for s and p, n + 1 for d (3d fills
    after 4s), n + 2 for f."""
    return n + max(l - 1, 0)


def _core_valence_split(levels) -> tuple[list, list]:
    """Valence = the occupied orbitals of the outermost row: its s and p shells, and a d (or f)
    shell of that row if it lies within VALENCE_WINDOW of the row's s level — or is the top
    occupied shell. The 3d of the transition metals is valence (as shallow as 4s, it bonds);
    the full 3d of gallium onward sits deeper and joins the core. Everything else is core.

    Principal numbers come from counting each orbital's radial nodes.
    """
    occ = sorted({(lv.n, lv.l): lv for lv in levels if lv.spin == 0 and lv.occupation > 1e-6}.values(),
                 key=lambda lv: lv.energy)
    if not occ:
        return [], []
    row = max(_period(lv.n, lv.l) for lv in occ)
    outer = [lv for lv in occ if _period(lv.n, lv.l) == row]
    s_levels = [lv.energy for lv in outer if lv.l == 0]
    e_ref = max(s_levels) if s_levels else max(lv.energy for lv in outer)
    val = [lv for lv in outer if lv.l <= 1 or lv.energy > e_ref - VALENCE_WINDOW]
    return [lv for lv in occ if lv not in val], val


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


def default_local_channel(Z: int) -> int:
    """First row: p (no core p states, so V_p is the gentlest channel).
    Third row and beyond: d, the channel with no core states to be orthogonal to, so the d-like
    scattering electrons meet in solids and molecules is not pushed out by core repulsion.
    Transition metals, whose 3d is valence: p, the unoccupied channel."""
    if 21 <= Z <= 30:
        return 1
    return 2 if Z >= 11 else 1


def _partial_core(grid: RadialGrid, rho_core: np.ndarray, rho_val: np.ndarray) -> np.ndarray:
    """Louie–Froyen–Cohen partial core: the true core density outside r_pc (where it has fallen
    to twice the valence density), a smooth A sin(Br)/r inside matching value and slope."""
    r = grid.r
    over = np.nonzero(rho_core > 2 * rho_val)[0]
    i = int(over.max()) if len(over) else 0
    rpc = r[i]
    val, d1, _ = _local_fit(r, rho_core, i)
    target = d1 / val                                         # log-derivative to match
    lo, hi = 1e-6, math.pi / rpc - 1e-9                       # B cot(B r) − 1/r is decreasing in B
    for _ in range(200):
        B = 0.5 * (lo + hi)
        f = B / math.tan(B * rpc) - 1 / rpc - target
        lo, hi = (B, hi) if f > 0 else (lo, B)
    A = val * rpc / math.sin(B * rpc)
    inner = A * np.sin(B * r) / r
    return np.where(r < rpc, inner, rho_core)


GHOST_TOL = 2e-3   # Ha


def generate(Z: int, rc: dict[int, float] | None = None, l_local: int | None = None) -> Pseudopotential:
    """Pseudopotential for element Z. Unless ``l_local`` is given, the local channel is the first
    of (default, s, p, d) whose separable form has no ghost states (see :func:`ghost_check`);
    for Z ≤ 18 the default is ghost-free and is kept."""
    grid = RadialGrid(r_min=2e-6 / math.sqrt(Z), r_max=60.0, dx=0.004)
    ref: AtomResult = RadialAtom(Z, spin=0.0, grid=grid).solve()
    if not ref.converged:
        ref = _lowest_configuration(Z, grid, ref)
    if l_local is not None:
        return _build(Z, ref, grid, rc, l_local)
    first = default_local_channel(Z)
    if Z <= 18:
        return _build(Z, ref, grid, rc, first)
    base = rc or DEFAULT_RC[Z]
    base = base if isinstance(base, dict) else {0: base, 1: base, 2: base}
    ghostly, clean = [], []
    # a softer local potential (its channel pseudised further out) removes most ghosts
    for scale in (1.0, 1.25, 1.5):
        for cand in [first] + [l for l in (0, 1, 2) if l != first]:
            rcs = dict(base)
            rcs[cand] = base[cand] * scale
            try:
                pp = _build(Z, ref, grid, rcs, cand)
            except Exception:
                continue
            g = max((abs(a - b) for a, b in ghost_check(pp).values()), default=0.0)
            if g >= GHOST_TOL:
                ghostly.append((g, pp))
                continue
            try:
                err = transfer_error(pp)
            except Exception:
                continue
            if err < GOOD_TRANSFER:
                return pp
            clean.append((err, pp))
    if clean:
        return min(clean, key=lambda x: x[0])[1]
    if ghostly:
        return min(ghostly, key=lambda x: x[0])[1]
    raise RuntimeError(f"no pseudopotential could be built for Z={Z}")


def _lowest_configuration(Z: int, grid: RadialGrid, trial: AtomResult) -> AtomResult:
    """When filling levels by energy does not settle (electrons slosh between two nearly
    degenerate shells, as 4s and 3d do in the transition metals), try each way of sharing the
    outer electrons between the outermost row's s and d shells and keep the lowest energy."""
    occ = {(lv.n, lv.l): lv.occupation for lv in trial.levels if lv.spin == 0 and lv.occupation > 1e-6}
    row = max(_period(n, l) for n, l in occ)
    s_nl = (row, 0)
    d_nl = (row - 1, 2)
    if d_nl not in occ:
        return trial
    n_sd = 2 * (occ.get(s_nl, 0.0) + occ[d_nl])
    fixed = {}
    for (n, l), f in occ.items():
        if (n, l) not in (s_nl, d_nl):
            for spin in (0, 1):
                fixed[(spin, l, n - l - 1)] = float(round(2 * f) / 2)
    best = trial
    for s_e in (2, 1, 0):
        d_e = round(n_sd) - s_e
        if not 0 <= d_e <= 10:
            continue
        cfg = dict(fixed)
        for spin in (0, 1):
            cfg[(spin, 0, s_nl[0] - 1)] = s_e / 2
            cfg[(spin, 2, d_nl[0] - 3)] = d_e / 2
        res = RadialAtom(Z, spin=0.0, grid=grid, occupations=cfg).solve()
        if res.converged and (not best.converged or res.energy < best.energy):
            best = res
    return best


GOOD_TRANSFER = 0.05 / 27.211386     # Ha: accept the first ghost-free candidate this transferable


def transfer_error(pp: Pseudopotential) -> float:
    """Largest AE–PS difference in excitation energy over verify()'s configurations (Ha)."""
    return worst_transfer(verify(pp))


def worst_transfer(rows: list[dict]) -> float:
    """Largest AE–PS excitation-energy difference over the configurations whose all-electron and
    pseudo-atom calculations both converged (an unconverged solve says nothing about the
    pseudopotential). Infinite if fewer than two remain."""
    good = [r for r in rows if r.get("converged", True)]
    if len(good) < 2 or not rows[0].get("converged", True):
        return float("inf")
    return max(abs(r["dE_ps"] - r["dE_ae"]) for r in good)


def _build(Z: int, ref: AtomResult, grid: RadialGrid, rc, l_local: int) -> Pseudopotential:
    core, valence = _core_valence_split(ref.levels)
    core_nl = {(lv.n, lv.l) for lv in core}
    valence_nl = {(lv.n, lv.l) for lv in valence}
    if not core:
        raise ValueError(f"Z={Z} has no core electrons; use the bare nucleus")
    n_core = 2 * sum((2 * lv.l + 1) for lv in core)
    Z_val = Z - n_core
    V = ref.v_up  # spin-unpolarised reference: both spins identical
    rc0 = rc or DEFAULT_RC[Z]
    rc_all = rc0 if isinstance(rc0, dict) else {0: rc0, 1: rc0, 2: rc0}
    d_valence = any(l == 2 for _, l in valence_nl)

    u_ps, v_scr, eps, occ, rcs, valence_n = {}, {}, {}, {}, {}, {}
    scatter = []
    for l in (0, 1, 2):
        if l == 2 and not d_valence:
            if l_local == 2:
                scatter.append(2)
            continue
        lv_list = [lv for lv in ref.levels if lv.spin == 0 and lv.l == l and (lv.n, lv.l) not in core_nl]
        if not lv_list:
            scatter.append(l)          # no bound level in this channel (e.g. 4p of the 3d metals)
            continue
        lv = min(lv_list, key=lambda x: x.energy)
        u_ps[l], v_scr[l], rcs[l] = _tm_channel(grid, lv.u, V, lv.energy, l, rc_all[l])
        eps[l] = lv.energy
        occ[l] = 2 * lv.occupation  # both spins
        valence_n[l] = lv.n
    for l in scatter:
        # unbound channel: the scattering state at the highest valence energy
        e_ref = max(eps.values())
        u_sc = scattering_state(grid, V, l, e_ref)
        u_ps[l], v_scr[l], rcs[l] = _tm_channel(grid, u_sc, V, e_ref, l, rc_all[l])
        eps[l] = e_ref
        occ[l] = 0.0

    r = grid.r
    rho_v = sum(occ[l] * u_ps[l] ** 2 for l in u_ps) / (4 * np.pi * r * r)
    rho_core = ref.rho_up + ref.rho_dn - sum(
        2 * lv.occupation * lv.u ** 2 for lv in ref.levels if lv.spin == 0 and (lv.n, lv.l) in valence_nl
    ) / (4 * np.pi * r * r)
    rho_pc = None
    if Z >= NLCC_FROM:
        # exchange–correlation is not linear in the density: where core and valence overlap,
        # keep (a smoothed copy of) the core density inside the functional
        rho_v_ae = sum(2 * lv.occupation * lv.u ** 2 for lv in ref.levels
                       if lv.spin == 0 and (lv.n, lv.l) in valence_nl) / (4 * np.pi * r * r)
        rho_pc = _partial_core(grid, np.maximum(rho_core, 0), rho_v_ae)
    vH = hartree_potential(grid, rho_v)
    rho_xc = rho_v + (rho_pc if rho_pc is not None else 0)
    _, vxc, _ = lda_xc(0.5 * rho_xc, 0.5 * rho_xc)
    v_ion = {l: v_scr[l] - vH - vxc for l in v_scr}
    # Beyond the core region every channel equals −Z_v/r (up to exponentially small terms).
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
    return Pseudopotential(Z, float(Z_val), l_local, rcs, grid, v_ion, u_ps, eps, occ, kb, beta,
                           core=sorted(core_nl), valence_n=valence_n, rho_core=rho_pc)


# ------------------------------------------------------------------ checks
_AE_CACHE: dict = {}        # all-electron reference atoms, shared by every candidate pseudopotential

def ghost_check(pp: Pseudopotential, r_max: float = 20.0, dr: float = 0.01) -> dict:
    """Lowest levels of the separable (Kleinman–Bylander) Hamiltonian against the semilocal one
    it replaces, per channel. A KB level far below the semilocal level is a ghost: a spurious
    bound state created by the separable form. Returns {l: (ε_KB, ε_semilocal)}."""
    from scipy.linalg import eigh_tridiagonal, eigh
    r = np.arange(1, int(r_max / dr) + 1) * dr
    g = pp.grid

    def on(f):
        return np.interp(r, g.r, f)

    # screened potentials: the ionic parts plus the valence Hartree + xc of the reference
    rho_v = sum(pp.occ[l] * pp.u_ps[l] ** 2 for l in pp.u_ps) / (4 * np.pi * g.r ** 2)
    rho_xc = rho_v + (pp.rho_core if pp.rho_core is not None else 0)
    _, vxc, _ = lda_xc(0.5 * rho_xc, 0.5 * rho_xc)
    scr = on(hartree_potential(g, rho_v) + vxc)
    out = {}
    for l in pp.v_ion:
        cent = l * (l + 1) / (2 * r * r)
        off = np.full(len(r) - 1, -0.5 / dr ** 2)
        e_sl = eigh_tridiagonal(1 / dr ** 2 + cent + on(pp.v_ion[l]) + scr, off, select="i", select_range=(0, 0))[0][0]
        if l == pp.l_local:
            continue
        H = np.diag(1 / dr ** 2 + cent + on(pp.v_local) + scr) + np.diag(off, 1) + np.diag(off, -1)
        b = on(pp.beta[l]) * r
        H += pp.kb_energy[l] * np.outer(b, b) * dr
        e_kb = eigh(H, eigvals_only=True, subset_by_index=(0, 0))[0]
        out[l] = (float(e_kb), float(e_sl))
    return out

def verify(pp: Pseudopotential, configs: list[dict] | None = None) -> list[dict]:
    """Compare pseudo-atom and all-electron atom for several configurations.

    Each config is {l: electrons} for the valence channels. Returns rows with
    AE and PS excitation energies (relative to the first config) and
    eigenvalues. Agreement across configurations = transferability.
    """
    if configs is None:
        if 2 in pp.valence_n:
            # transition metal: reference, s → d promotion, d → s, and the cation
            s0, d0 = pp.occ[0], pp.occ[2]
            ds = min(1.0, 2.0 - s0, d0)                          # move up to one electron d → s
            if ds > 0.25:
                third = {0: s0 + ds, 2: d0 - ds}
            else:                                                # s already full: two electrons s → d
                move = min(s0, 10.0 - d0, 2.0)
                third = {0: s0 - move, 2: d0 + move} if move > 0.25 else {0: s0 - 1, 2: d0}
            configs = [{0: s0, 2: d0}, {0: max(s0 - 1, 0), 2: min(d0 + min(1, s0), 10)},
                       third, {0: max(s0 - 1, 0), 2: d0 if s0 >= 1 else d0 - 1}]
        else:
            configs = [{0: pp.occ[0], 1: pp.occ[1]},
                       {0: pp.occ[0], 1: max(pp.occ[1] - 1, 0)},
                       {0: max(pp.occ[0] - 1, 0), 1: pp.occ[1] + 1}]
    grid = pp.grid
    rows = []
    core_occ_ae = _core_occupations(pp)
    for cfg in configs:
        ae_fixed, ps_fixed = dict(core_occ_ae), {}
        for l, n in cfg.items():
            idx_ae = pp.valence_n[l] - l - 1  # e.g. 3s is the third s level
            ae_fixed[(0, l, idx_ae)] = n / 2
            ae_fixed[(1, l, idx_ae)] = n / 2
            ps_fixed[(0, l, 0)] = n / 2
            ps_fixed[(1, l, 0)] = n / 2
        ne_val = sum(cfg.values())
        key = (pp.Z, tuple(sorted(ae_fixed.items())), int(round(pp.Z_val - ne_val)))
        if key not in _AE_CACHE:
            _AE_CACHE[key] = RadialAtom(pp.Z, charge=key[2], grid=grid, occupations=ae_fixed).solve()
        ae = _AE_CACHE[key]
        ps = RadialAtom(pp.Z, charge=int(round(pp.Z_val - ne_val)), grid=grid, v_external=pp.v_ion,
                        n_valence=pp.Z_val, occupations=ps_fixed, rho_core=pp.rho_core).solve()
        rows.append({"config": cfg, "E_ae": ae.energy, "E_ps": ps.energy, "converged": bool(ae.converged and ps.converged),
                     "eps_ae": {l: _level(ae, l, pp.valence_n[l] - l - 1) for l in cfg},
                     "eps_ps": {l: _level(ps, l, 0) for l in cfg}})
    for row in rows:
        row["dE_ae"] = row["E_ae"] - rows[0]["E_ae"]
        row["dE_ps"] = row["E_ps"] - rows[0]["E_ps"]
    return rows


def _core_occupations(pp: Pseudopotential) -> dict:
    """Full core subshells as fixed occupations {(spin, l, index): electrons}."""
    occ = {}
    for n, l in pp.core:
        for spin in (0, 1):
            occ[(spin, l, n - l - 1)] = float(2 * l + 1)
    return occ


def _level(res: AtomResult, l: int, index: int) -> float:
    """Energy of the index-th (0-based) spin-up level of angular momentum l."""
    for lv in res.levels:
        if lv.spin == 0 and lv.l == l and lv.n == index + l + 1:
            return lv.energy
    return float("nan")
