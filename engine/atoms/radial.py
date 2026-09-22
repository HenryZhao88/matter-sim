"""All-electron spherical atoms on a logarithmic radial grid.

Solves the same spin-polarised Kohn–Sham equations as the 3D engine, but for
a single nucleus with a spherically averaged density, where the problem
reduces to one radial equation per angular momentum l:

    −½ u'' + [V_σ(r) + l(l+1)/(2r²)] u = ε u,      u(r) = r R(r).

A logarithmic grid resolves the core near the nucleus for any Z at little
cost, so this solver is essentially exact for the model (to ~1e-6 Ha).

Nothing here assumes shell structure. Levels are filled in order of their
computed energy (with Fermi smearing across each degenerate subshell), and
:func:`ground_state` picks the spin that gives the lowest energy — so the
filling order and Hund's rule are outputs, not inputs.

Method: with x = ln r and u = r^{1/2} φ(x) the equation becomes
φ'' = F(x) φ,  F = 2r²(V − ε) + (l + ½)², integrated by Numerov's method
(4th-order) outward from the nucleus and inward from the far tail. Each level
is found by counting nodes (bisection) and then closing the derivative
mismatch at the classical turning point with a first-order energy correction.
Matrix eigensolvers were tried first; the ~16 orders of magnitude spanned by
r² on this grid defeat them in double precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numba import njit

from ..electrons.mixing import PulayMixer
from ..electrons.xc import lda_xc

L_MAX = 3


@dataclass
class RadialGrid:
    r_min: float = 2e-6
    r_max: float = 60.0
    dx: float = 0.004

    def __post_init__(self) -> None:
        n = int(math.ceil((math.log(self.r_max) - math.log(self.r_min)) / self.dx)) + 1
        self.x = math.log(self.r_min) + self.dx * np.arange(n)
        self.r = np.exp(self.x)
        self.n = n

    def integrate(self, f: np.ndarray) -> float:
        """∫ f(r) dr on the log grid (dr = r dx), trapezoid rule."""
        g = f * self.r
        return float(self.dx * (g.sum() - 0.5 * (g[0] + g[-1])))

    def cumulative(self, f: np.ndarray) -> np.ndarray:
        """∫_0^r f dr for every r."""
        g = f * self.r
        c = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * self.dx)])
        return c + g[0] * self.dx * 0.5  # r_min is tiny: treat [0, r_min] as a sliver


@dataclass
class Level:
    n: int          # principal number, counted from nodes: n = nodes + l + 1
    l: int
    spin: int       # 0 = up, 1 = down
    energy: float
    occupation: float
    u: np.ndarray = field(repr=False)

    @property
    def label(self) -> str:
        return f"{self.n}{'spdf'[self.l]}"


@dataclass
class AtomResult:
    Z: int
    charge: int
    n_up: float
    n_dn: float
    energy: float
    components: dict[str, float]
    levels: list[Level]
    rho_up: np.ndarray
    rho_dn: np.ndarray
    v_up: np.ndarray
    v_dn: np.ndarray
    grid: RadialGrid
    converged: bool
    iterations: int

    @property
    def multiplicity(self) -> int:
        return int(round(self.n_up - self.n_dn)) + 1

    def configuration(self) -> str:
        """e.g. '1s² 2s² 2p⁴' — read off the computed occupations."""
        sup = str.maketrans("0123456789.", "⁰¹²³⁴⁵⁶⁷⁸⁹·")
        tot: dict[tuple[int, int], float] = {}
        for lv in self.levels:
            if lv.occupation > 1e-3:
                tot[(lv.n, lv.l)] = tot.get((lv.n, lv.l), 0.0) + lv.occupation
        parts = []
        for (n, l), occ in sorted(tot.items(), key=lambda kv: (kv[0][0] + kv[0][1], kv[0][0])):
            s = f"{occ:.0f}" if abs(occ - round(occ)) < 1e-3 else f"{occ:.2f}"
            parts.append(f"{n}{'spdf'[l]}{s.translate(sup)}")
        return " ".join(parts)


@njit(cache=True)
def _shoot(r, h, V, l, eps, z0):
    """Numerov outward/inward integration at trial energy ``eps``.

    Returns (nodes, energy correction, φ array).
    """
    n = r.shape[0]
    F = 2.0 * r * r * (V - eps) + (l + 0.5) ** 2
    w = 1.0 - h * h * F / 12.0
    # outermost classically allowed point
    ic = -1
    for i in range(n - 2, 1, -1):
        if F[i] < 0.0:
            ic = i
            break
    if ic < 2:
        ic = n // 2
    # far point: where the tail has decayed by ~e^-25
    kappa = np.sqrt(max(-2.0 * eps, 1e-12))
    iend = n - 1
    for i in range(ic, n):
        if kappa * (r[i] - r[ic]) > 25.0:
            iend = i
            break
    if iend < ic + 3:
        iend = min(n - 1, ic + 3)
    phi = np.zeros(n)
    # outward: φ ≈ r^(l+½) (1 − z0 r/(l+1)) near the nucleus
    phi[0] = r[0] ** (l + 0.5) * (1.0 - z0 * r[0] / (l + 1.0))
    phi[1] = r[1] ** (l + 0.5) * (1.0 - z0 * r[1] / (l + 1.0))
    nodes = 0
    for i in range(1, ic):
        phi[i + 1] = ((12.0 - 10.0 * w[i]) * phi[i] - w[i - 1] * phi[i - 1]) / w[i + 1]
        if phi[i + 1] * phi[i] < 0.0:
            nodes += 1
        if abs(phi[i + 1]) > 1e100:
            for j in range(i + 2):
                phi[j] *= 1e-100
    phic = phi[ic]
    # inward
    phi[iend] = 1e-30
    phi[iend - 1] = 1e-30 * np.exp(kappa * (r[iend] - r[iend - 1]))
    for i in range(iend - 1, ic, -1):
        phi[i - 1] = ((12.0 - 10.0 * w[i]) * phi[i] - w[i + 1] * phi[i + 1]) / w[i - 1]
        if abs(phi[i - 1]) > 1e100:
            for j in range(i - 1, iend + 1):
                phi[j] *= 1e-100
    scale = phic / phi[ic]
    for i in range(ic, iend + 1):
        phi[i] *= scale
    # derivative mismatch at ic → first-order energy correction
    R = w[ic + 1] * phi[ic + 1] + w[ic - 1] * phi[ic - 1] - (12.0 - 10.0 * w[ic]) * phi[ic]
    norm = 0.0
    for i in range(iend + 1):
        norm += r[i] * r[i] * phi[i] * phi[i]
    de = -phi[ic] * R / (2.0 * h * h * norm * h) * h
    return nodes, de, phi


@njit(cache=True)
def _find_level(r, h, V, l, nodes_target, z0, tol):
    elo = 1e300
    for i in range(r.shape[0]):
        v = V[i] + 0.5 * l * (l + 1) / (r[i] * r[i])
        if v < elo:
            elo = v
    ehi = 0.0
    e = 0.5 * (elo + ehi)
    phi = np.zeros(r.shape[0])
    for _ in range(300):
        nodes, de, phi = _shoot(r, h, V, l, e, z0)
        if nodes > nodes_target:
            ehi = e
            e = 0.5 * (elo + ehi)
            continue
        if nodes < nodes_target:
            elo = e
            e = 0.5 * (elo + ehi)
            if ehi - elo < 1e-12:
                return 1.0, phi  # unbound
            continue
        if de > 0.0:
            elo = e
        else:
            ehi = e
        e_new = e + de
        if e_new <= elo or e_new >= ehi:
            e_new = 0.5 * (elo + ehi)
        if abs(e_new - e) < tol:
            return e_new, phi
        e = e_new
        if ehi - elo < 1e-14:
            return e, phi
    return e, phi


@njit(cache=True)
def _outward(r, h, V, l, eps, z0):
    n = r.shape[0]
    F = 2.0 * r * r * (V - eps) + (l + 0.5) ** 2
    w = 1.0 - h * h * F / 12.0
    phi = np.zeros(n)
    phi[0] = r[0] ** (l + 0.5) * (1.0 - z0 * r[0] / (l + 1.0))
    phi[1] = r[1] ** (l + 0.5) * (1.0 - z0 * r[1] / (l + 1.0))
    for i in range(1, n - 1):
        phi[i + 1] = ((12.0 - 10.0 * w[i]) * phi[i] - w[i - 1] * phi[i - 1]) / w[i + 1]
        if abs(phi[i + 1]) > 1e150:
            break
    return phi


def scattering_state(grid: RadialGrid, V: np.ndarray, l: int, eps: float, z0: float = 0.0) -> np.ndarray:
    """Regular solution u(r) at any energy (bound or not), integrated outward from the nucleus.

    Used for channels with no bound state in the atom (e.g. d in aluminium); only its shape
    inside the core radius matters for building a pseudopotential.
    """
    phi = _outward(grid.r, grid.dx, V, l, eps, z0)
    return phi * np.sqrt(grid.r)


def radial_eigenstates(grid: RadialGrid, V: np.ndarray, l: int, count: int, z0: float = 0.0):
    """Lowest ``count`` bound eigenpairs (ε, u normalised) for angular momentum l."""
    r = grid.r
    out_e, out_u = [], []
    for k in range(count):
        e, phi = _find_level(r, grid.dx, V, l, k, z0, 1e-11)
        if e >= 0:
            break
        u = phi * np.sqrt(r)
        u /= math.sqrt(grid.integrate(u * u))
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        out_e.append(e)
        out_u.append(u)
    return np.array(out_e), out_u


def hartree_potential(grid: RadialGrid, rho: np.ndarray) -> np.ndarray:
    r = grid.r
    q_in = grid.cumulative(4 * np.pi * r * r * rho)          # charge inside r
    outer = grid.cumulative(4 * np.pi * r * rho)
    return q_in / r + (outer[-1] - outer)


def _fill(levels: list[tuple[int, int, float]], n_electrons: float, T: float) -> np.ndarray:
    """Occupations for levels (l, index, ε) with degeneracy 2l+1 each (one spin)."""
    if n_electrons <= 0 or not levels:
        return np.zeros(len(levels))
    e = np.array([lv[2] for lv in levels])
    g = np.array([2 * lv[0] + 1 for lv in levels], dtype=float)
    if g.sum() < n_electrons:
        raise ValueError("not enough bound levels for the electrons")

    def occ(mu):
        return g * 0.5 * (1.0 - np.tanh((e - mu) / (2 * T)))

    lo, hi = e.min() - 1.0, e.max() + 1.0
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if occ(mu).sum() > n_electrons:
            hi = mu
        else:
            lo = mu
    f = occ(mu)
    return f * (n_electrons / f.sum())


class RadialAtom:
    def __init__(self, Z: int, charge: int = 0, spin: float = 0.0, grid: RadialGrid | None = None,
                 T_e: float = 1e-4, v_external=None, n_valence: float | None = None,
                 occupations: dict | None = None) -> None:
        """``spin`` = N↑ − N↓.

        ``v_external`` replaces −Z/r: an array, or a dict {l: array} of
        angular-momentum-dependent (semilocal) potentials for pseudo-atoms.
        ``occupations`` optionally fixes {(spin, l, index): electrons} instead
        of filling by energy (used to test excited configurations).
        """
        self.Z = Z
        self.charge = charge
        self.grid = grid or RadialGrid(r_min=2e-6 / max(Z, 1) ** 0.5)
        ne = (Z - charge) if n_valence is None else n_valence - charge
        self.n_up = 0.5 * (ne + spin)
        self.n_dn = 0.5 * (ne - spin)
        self.T_e = T_e
        r = self.grid.r
        if v_external is None:
            v_external = -Z / r
        if isinstance(v_external, dict):
            self.v_l = {l: np.asarray(v) for l, v in v_external.items()}
        else:
            self.v_l = {l: np.asarray(v_external) for l in range(L_MAX + 1)}
        top = max(self.v_l)
        for l in range(L_MAX + 1):
            self.v_l.setdefault(l, self.v_l[top])
        self.v_ext = self.v_l[0]
        self.z0 = {l: float(-v[0] * r[0]) for l, v in self.v_l.items()}  # Coulomb strength at 0
        self.fixed_occ = occupations

    def solve(self, max_iter: int = 400, tol: float = 1e-9) -> AtomResult:
        g, r = self.grid, self.grid.r
        ne = self.n_up + self.n_dn
        # Start from a hydrogen-like cloud; converged result does not depend on it.
        # a compact core-sized cloud for bare nuclei, a diffuse valence-sized one for pseudo-atoms
        zeff = 0.7 if self.z0[0] < 1e-6 else max(self.Z, 1) ** (1 / 3)
        rho0 = ne * zeff ** 3 * np.exp(-2 * zeff * r) / np.pi
        rho = [rho0 * self.n_up / max(ne, 1e-12), rho0 * self.n_dn / max(ne, 1e-12)]
        mixer = PulayMixer(beta=0.3, history=8)
        E_prev = np.inf
        converged = False
        n_per_l = int(math.ceil(max(ne, 1) / 2)) + 3

        for it in range(1, max_iter + 1):
            vH = hartree_potential(g, rho[0] + rho[1])
            _, vxu, vxd = lda_xc(rho[0], rho[1])
            vs = (vH + vxu, vH + vxd)
            new_rho, levels_all = [np.zeros_like(r), np.zeros_like(r)], []
            band = e_ext = pot = 0.0
            for s, nel in ((0, self.n_up), (1, self.n_dn)):
                cands = []
                for l in range(L_MAX + 1):
                    w, U = radial_eigenstates(g, self.v_l[l] + vs[s], l, n_per_l, self.z0[l])
                    for k, (eps, u) in enumerate(zip(w, U)):
                        if eps < 0:
                            cands.append((l, k, eps, u))
                if self.fixed_occ is not None:
                    occ = np.array([self.fixed_occ.get((s, c[0], c[1]), 0.0) for c in cands])
                else:
                    occ = _fill([(c[0], c[1], c[2]) for c in cands], nel, self.T_e)
                for (l, k, eps, u), f in zip(cands, occ):
                    levels_all.append(Level(k + l + 1, l, s, float(eps), float(f), u))
                    if f > 0:
                        new_rho[s] += f * u * u / (4 * np.pi * r * r)
                        band += f * eps
                        e_ext += f * g.integrate(u * u * self.v_l[l])
                        pot += f * g.integrate(u * u * (self.v_l[l] + vs[s]))
            comps = self._energy(new_rho, band - pot, e_ext)
            E = sum(comps.values())
            drho = g.integrate(4 * np.pi * r * r * (np.abs(new_rho[0] - rho[0]) + np.abs(new_rho[1] - rho[1])))
            if abs(E - E_prev) < tol and drho < 1e-6:
                rho = new_rho
                converged = True
                break
            E_prev = E
            x = mixer.mix(np.concatenate(rho), np.concatenate(new_rho))
            x = np.maximum(x, 0.0)
            rho = [x[: g.n], x[g.n:]]
        vH = hartree_potential(g, rho[0] + rho[1])
        _, vxu, vxd = lda_xc(rho[0], rho[1])
        return AtomResult(self.Z, self.charge, self.n_up, self.n_dn, E, comps, levels_all,
                          rho[0], rho[1], self.v_ext + vH + vxu, self.v_ext + vH + vxd,
                          g, converged, it)

    def _energy(self, rho, kinetic, e_ext) -> dict[str, float]:
        g, r = self.grid, self.grid.r
        rt = rho[0] + rho[1]
        w = 4 * np.pi * r * r
        vH = hartree_potential(g, rt)
        e_xc, _, _ = lda_xc(rho[0], rho[1])
        # kinetic = Σ f ε − Σ f ⟨u|V_eff|u⟩ (V_eff that produced the orbitals)
        return {
            "kinetic": kinetic,
            "electron_nuclear": e_ext,
            "hartree": 0.5 * g.integrate(w * vH * rt),
            "exchange_correlation": g.integrate(w * e_xc),
        }


def ground_state(Z: int, charge: int = 0, max_spin: int = 5, **kw) -> AtomResult:
    """Lowest-energy spin state: Hund's rule is found, not assumed."""
    ne = Z - charge
    best = None
    for spin in range(ne % 2, min(ne, max_spin) + 1, 2):
        try:
            res = RadialAtom(Z, charge, spin, **kw).solve()
        except ValueError:  # that many same-spin electrons cannot all be bound
            continue
        if best is None or res.energy < best.energy - 1e-7:
            best = res
    return best
