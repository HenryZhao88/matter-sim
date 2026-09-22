"""Lattice QCD (pure gauge): gluons on a 4D spacetime grid, from the QCD Lagrangian alone.

Each link of an L³×T lattice carries an SU(3) matrix U_μ(x), the gluon field. The Wilson
action S = β Σ_plaquettes (1 − Re Tr U_P / 3) is the lattice form of the QCD Lagrangian
(β = 6/g²). Configurations are drawn with probability ∝ e^{−S} by Cabibbo–Marinari
heatbath (Kennedy–Pendleton for each SU(2) subgroup) plus overrelaxation.

What comes out, with nothing put in:
* Wilson loops W(R, T) → the energy V(R) of a static quark–antiquark pair. A potential
  that keeps rising linearly with distance is confinement: quarks cannot be pulled apart.
* The Polyakov loop on a short time extent (finite temperature) → the deconfinement
  transition, the change from confined hadrons to a quark–gluon plasma.

This is the "quenched" approximation (no dynamical quark loops); dynamical quarks are a
later milestone. Lattice units throughout; physical units need one measured scale.
"""

from __future__ import annotations

import math
import time

import numpy as np

# SU(2) subgroups of SU(3) used by Cabibbo–Marinari
SUBGROUPS = ((0, 1), (1, 2), (0, 2))


class GaugeField:
    def __init__(self, L: int, T: int, beta: float, seed: int = 0, hot: bool = False) -> None:
        self.L, self.T, self.beta = L, T, beta
        self.shape = (T, L, L, L)
        self.rng = np.random.default_rng(seed)
        U = np.zeros((4,) + self.shape + (3, 3), complex)
        U[...] = np.eye(3)
        if hot:
            U = _random_su3(self.rng, U.shape[:-2])
        self.U = U
        t, x, y, z = np.meshgrid(*[np.arange(n) for n in self.shape], indexing="ij")
        self.parity = (t + x + y + z) % 2

    # ------------------------------------------------------------ geometry
    @staticmethod
    def shift(A, mu, n=1):
        """A(x + n μ̂) (periodic)."""
        return np.roll(A, -n, axis=mu)

    def staple(self, mu: int) -> np.ndarray:
        """Σ_ν≠μ of the six-link staples around U_μ(x) (the 'environment' of each link)."""
        U = self.U
        S = np.zeros_like(U[mu])
        for nu in range(4):
            if nu == mu:
                continue
            Un_xmu = self.shift(U[nu], mu)                 # U_ν(x+μ)
            Um_xnu = self.shift(U[mu], nu)                 # U_μ(x+ν)
            up = Un_xmu @ _dag(Um_xnu) @ _dag(U[nu])
            Un_xmu_mnu = self.shift(self.shift(U[nu], mu), nu, -1)   # U_ν(x+μ−ν)
            Um_mnu = self.shift(U[mu], nu, -1)             # U_μ(x−ν)
            Un_mnu = self.shift(U[nu], nu, -1)             # U_ν(x−ν)
            down = _dag(Un_xmu_mnu) @ _dag(Um_mnu) @ Un_mnu
            S += up + down
        return S

    # ------------------------------------------------------------ updates
    def sweep(self, overrelax: int = 3) -> None:
        for mu in range(4):
            for par in (0, 1):
                mask = self.parity == par
                A = self.staple(mu)[mask]
                Uloc = self.U[mu][mask]
                Uloc = self._heatbath(Uloc, A)
                for _ in range(overrelax):
                    Uloc = self._overrelax(Uloc, A)
                self.U[mu][mask] = Uloc
        self.U = _reunitarise(self.U)

    def _subgroup_parts(self, W, i, j):
        a0 = (W[:, i, i] + np.conj(W[:, j, j])).real / 2
        a1 = (W[:, i, j] + W[:, j, i]).imag / 2
        a2 = (W[:, i, j] - W[:, j, i]).real / 2
        a3 = (W[:, i, i] - W[:, j, j]).imag / 2
        a = np.stack([a0, a1, a2, a3], axis=1)
        k = np.linalg.norm(a, axis=1)
        return a / np.maximum(k, 1e-15)[:, None], k

    def _heatbath(self, U, A):
        for (i, j) in SUBGROUPS:
            W = U @ A
            v, k = self._subgroup_parts(W, i, j)
            x = _kennedy_pendleton(self.rng, 2 * self.beta / 3 * k)
            # new SU(2) element R = X V†, embedded, acting from the left
            R = _quat_mul(x, _quat_conj(v))
            U = _embed(R, i, j) @ U
        return U

    def _overrelax(self, U, A):
        for (i, j) in SUBGROUPS:
            W = U @ A
            v, _ = self._subgroup_parts(W, i, j)
            vd = _quat_conj(v)
            R = _quat_mul(vd, vd)                       # reflection: leaves the action unchanged
            U = _embed(R, i, j) @ U
        return U

    # ------------------------------------------------------------ observables
    def plaquette(self) -> float:
        U = self.U
        tot, n = 0.0, 0
        for mu in range(4):
            for nu in range(mu + 1, 4):
                P = U[mu] @ self.shift(U[nu], mu) @ _dag(self.shift(U[mu], nu)) @ _dag(U[nu])
                tot += float(np.trace(P, axis1=-2, axis2=-1).real.mean()) / 3
                n += 1
        return tot / n

    def wilson_loops(self, Rmax: int, Tmax: int) -> np.ndarray:
        """W[R-1, T-1] averaged over position and the three spatial directions (time axis 0)."""
        U = self.U
        W = np.zeros((Rmax, Tmax))
        for sd in (1, 2, 3):
            # spatial line of length R, time line of length T
            space = [np.broadcast_to(np.eye(3), U[sd].shape).copy()]
            for R in range(1, Rmax + 1):
                space.append(space[-1] @ self.shift(U[sd], sd, R - 1))
            time_ = [np.broadcast_to(np.eye(3), U[0].shape).copy()]
            for T in range(1, Tmax + 1):
                time_.append(time_[-1] @ self.shift(U[0], 0, T - 1))
            for R in range(1, Rmax + 1):
                for T in range(1, Tmax + 1):
                    loop = (space[R] @ self.shift(time_[T], sd, R)
                            @ _dag(self.shift(space[R], 0, T)) @ _dag(time_[T]))
                    W[R - 1, T - 1] += float(np.trace(loop, axis1=-2, axis2=-1).real.mean()) / 3
        return W / 3

    def polyakov(self) -> complex:
        """Spatial average of Tr Π_t U_0 / 3: zero when quarks are confined."""
        P = self.U[0][0]
        for t in range(1, self.T):
            P = P @ self.U[0][t]
        return complex(np.trace(P, axis1=-2, axis2=-1).mean() / 3)

    def ape_smear(self, alpha: float = 0.5, steps: int = 10) -> "GaugeField":
        """Spatially smeared copy (improves the signal of Wilson loops; physics unchanged)."""
        g = GaugeField.__new__(GaugeField)
        g.__dict__.update(self.__dict__)
        U = self.U.copy()
        for _ in range(steps):
            new = U.copy()
            for mu in (1, 2, 3):
                S = np.zeros_like(U[mu])
                for nu in (1, 2, 3):
                    if nu == mu:
                        continue
                    S += (self.shift(U[nu], mu) @ _dag(self.shift(U[mu], nu)) @ _dag(U[nu])).conj().swapaxes(-1, -2)
                    S += (_dag(self.shift(self.shift(U[nu], mu), nu, -1)) @ _dag(self.shift(U[mu], nu, -1))
                          @ self.shift(U[nu], nu, -1)).conj().swapaxes(-1, -2)
                new[mu] = _project_su3((1 - alpha) * U[mu] + alpha / 4 * S)
            U = new
        g.U = U
        return g


# ------------------------------------------------------------------ SU(2)/SU(3) helpers
def _dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


def _quat_conj(q):
    return q * np.array([1, -1, -1, -1])


def _quat_mul(p, q):
    a0, a1, a2, a3 = p.T
    b0, b1, b2, b3 = q.T
    return np.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                     a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                     a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], axis=1)


def _embed(q, i, j):
    """SU(2) quaternion a0 + i a·σ embedded in the (i, j) block of 3×3 identities."""
    n = q.shape[0]
    M = np.zeros((n, 3, 3), complex)
    M[:, 0, 0] = M[:, 1, 1] = M[:, 2, 2] = 1
    a0, a1, a2, a3 = q.T
    M[:, i, i] = a0 + 1j * a3
    M[:, i, j] = a2 + 1j * a1
    M[:, j, i] = -a2 + 1j * a1
    M[:, j, j] = a0 - 1j * a3
    return M


def _kennedy_pendleton(rng, alpha):
    """Sample x0 with density ∝ √(1−x0²) e^{α x0}; return random SU(2) quaternions."""
    n = alpha.shape[0]
    x0 = np.empty(n)
    todo = np.arange(n)
    alpha = np.maximum(alpha, 1e-10)
    while todo.size:
        a = alpha[todo]
        r1, r2, r3, r4 = (1 - rng.random((4, todo.size)))
        lam2 = -(np.log(r1) + np.cos(2 * np.pi * r2) ** 2 * np.log(r3)) / (2 * a)
        ok = r4 ** 2 <= 1 - lam2
        x0[todo[ok]] = 1 - 2 * lam2[ok]
        todo = todo[~ok]
    # random direction for the remaining components
    r = np.sqrt(np.clip(1 - x0 * x0, 0, None))
    cos_t = 2 * rng.random(n) - 1
    phi = 2 * np.pi * rng.random(n)
    sin_t = np.sqrt(1 - cos_t * cos_t)
    return np.stack([x0, r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t], axis=1)


def _reunitarise(U):
    """Gram–Schmidt rows back onto SU(3) (removes floating-point drift)."""
    a = U[..., 0, :]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = U[..., 1, :]
    b = b - np.sum(np.conj(a) * b, axis=-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    c = np.conj(np.cross(a, b))
    return np.stack([a, b, c], axis=-2)


def _project_su3(M):
    return _reunitarise(M)


def _random_su3(rng, shape):
    Z = rng.normal(size=shape + (3, 3)) + 1j * rng.normal(size=shape + (3, 3))
    return _reunitarise(Z)


# ------------------------------------------------------------------ experiments
def static_potential(W: np.ndarray, T: int) -> np.ndarray:
    """V(R) = log(W(R,T) / W(R,T+1)) (lattice units)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(W[:, T - 1] / W[:, T])


def fit_cornell(R, V):
    """V = A − α/R + σR by least squares; returns (A, α, σ)."""
    R = np.asarray(R, float)
    X = np.stack([np.ones_like(R), -1 / R, R], axis=1)
    coef, *_ = np.linalg.lstsq(X, V, rcond=None)
    return tuple(float(c) for c in coef)


def run_measurement(beta: float, L: int = 8, T: int = 8, sweeps: int = 80, cancelled=lambda: False,
                    mode: str = "confinement"):
    """Generator of progress/result messages for the viewer."""
    t0 = time.perf_counter()
    g = GaugeField(L, T, beta, seed=1)
    therm = max(20, sweeps // 3)
    Rmax, Tmax = L // 2, min(T // 2, 5)
    Wacc = np.zeros((Rmax, Tmax))
    plaq = []
    n_meas = 0
    for s in range(sweeps):
        if cancelled():
            yield {"type": "qcd.done", "cancelled": True}
            return
        g.sweep()
        p = g.plaquette()
        plaq.append(p)
        if s >= therm and s % 2 == 0:
            Wacc += g.ape_smear().wilson_loops(Rmax, Tmax)
            n_meas += 1
        yield {"type": "qcd.progress", "sweep": s + 1, "sweeps": sweeps, "plaquette": p,
               "thermalised": s >= therm, "measurements": n_meas}
    W = Wacc / max(n_meas, 1)
    Tuse = 2 if Tmax > 2 else 1
    V = static_potential(W, Tuse)
    R = np.arange(1, Rmax + 1)
    good = np.isfinite(V)
    A, alpha, sigma = fit_cornell(R[good], V[good]) if good.sum() >= 3 else (float("nan"),) * 3
    yield {"type": "qcd.result", "beta": beta, "L": L, "T": T, "plaquette": float(np.mean(plaq[therm:])),
           "R": R.tolist(), "V": V.tolist(), "fit": {"A": A, "alpha": alpha, "sigma": sigma},
           "measurements": n_meas, "seconds": time.perf_counter() - t0}


def polyakov_scan(betas, L: int = 8, Nt: int = 4, sweeps: int = 60, seed: int = 3):
    """|⟨P⟩| against β on a lattice with short time extent (finite temperature)."""
    out = []
    for beta in betas:
        g = GaugeField(L, Nt, beta, seed=seed, hot=True)
        vals = []
        for s in range(sweeps):
            g.sweep(overrelax=2)
            if s >= sweeps // 3:
                vals.append(abs(g.polyakov()))
        out.append((float(beta), float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals)))))
    return out
