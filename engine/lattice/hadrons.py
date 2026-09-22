"""Hadron masses from lattice QCD: quarks propagating through the gluon field.

Quarks enter through the Wilson–Dirac operator on the gauge configurations of qcd.py
(quenched: the gluon field does not feel quark loops):

    D ψ(x) = ψ(x) − κ Σ_μ [ (1 − γ_μ) U_μ(x) ψ(x+μ̂) + (1 + γ_μ) U_μ(x−μ̂)† ψ(x−μ̂) ]

with κ = 1/(2m_q a + 8) setting the bare quark mass. The quark propagator S = D⁻¹ from a
point source is found by conjugate gradient. A hadron is a combination of quark fields;
its correlation function in Euclidean time falls as e^{−m t}, so its mass is read off
the decay. Nothing about hadrons is put in: pions, rho mesons and their masses come out
of how quarks move through the gluon field.

The pion is special: as the quark mass goes to zero its mass does too (m_π² ∝ m_q), the
signature of a Goldstone boson of spontaneously broken chiral symmetry.
"""

from __future__ import annotations

import math

import numpy as np

from .qcd import GaugeField, _dag

# Euclidean gamma matrices (Hermitian, {γ_μ, γ_ν} = 2δ_μν); index 0 is time.
_s = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
_Z2, _I2 = np.zeros((2, 2), complex), np.eye(2, dtype=complex)
GAMMA = [np.block([[_Z2, _I2], [_I2, _Z2]])] + [np.block([[_Z2, -1j * s], [1j * s, _Z2]]) for s in _s]
GAMMA5 = GAMMA[0] @ GAMMA[1] @ GAMMA[2] @ GAMMA[3]
ONE4 = np.eye(4, dtype=complex)


class WilsonDirac:
    def __init__(self, g: GaugeField, kappa: float) -> None:
        self.U = g.U                                      # (4, T, L, L, L, 3, 3)
        self.kappa = kappa
        self.shape = g.U.shape[1:5]
        T = self.shape[0]
        # antiperiodic boundary in time for fermions: flip the sign of the links crossing t = T−1 → 0
        self.Ut = self.U.copy()
        self.Ut[0, T - 1] *= -1
        self.P = [(ONE4 - GAMMA[m], ONE4 + GAMMA[m]) for m in range(4)]

    def apply(self, psi):
        """psi: (T, L, L, L, 4, 3)."""
        out = psi.copy()
        for mu in range(4):
            U = self.Ut[mu]
            fwd = np.roll(psi, -1, axis=mu)                              # ψ(x+μ)
            hop_f = np.einsum("...ab,...sb->...sa", U, fwd)              # U_μ(x) ψ(x+μ)
            bwd_links = np.roll(U, 1, axis=mu)                          # U_μ(x−μ)
            bwd = np.roll(psi, 1, axis=mu)                               # ψ(x−μ)
            hop_b = np.einsum("...ba,...sb->...sa", np.conj(bwd_links), bwd)   # U_μ(x−μ)† ψ(x−μ)
            Pm, Pp = self.P[mu]
            out -= self.kappa * (np.einsum("st,...tc->...sc", Pm, hop_f) + np.einsum("st,...tc->...sc", Pp, hop_b))
        return out

    def apply_dag(self, psi):
        """D† = γ5 D γ5."""
        g5 = lambda x: np.einsum("st,...tc->...sc", GAMMA5, x)
        return g5(self.apply(g5(psi)))

    def solve(self, b, tol=1e-8, maxiter=3000):
        """x = D⁻¹ b by conjugate gradient on the normal equations D†D x = D† b."""
        rhs = self.apply_dag(b)
        x = np.zeros_like(b)
        r = rhs.copy()
        p = r.copy()
        rr = np.vdot(r, r).real
        norm = math.sqrt(np.vdot(rhs, rhs).real)
        for it in range(maxiter):
            Ap = self.apply_dag(self.apply(p))
            alpha = rr / np.vdot(p, Ap).real
            x += alpha * p
            r -= alpha * Ap
            rr_new = np.vdot(r, r).real
            if math.sqrt(rr_new) < tol * norm:
                return x, it + 1
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x, maxiter


class WilsonDiracGPU:
    """The same operator on the Metal GPU (MLX, complex64), acting on all 12 point-source
    columns at once. Fields are laid out as X[t, x, y, z, colour, spin, column]."""

    def __init__(self, g: GaugeField, kappa: float) -> None:
        import mlx.core as mx
        self.mx = mx
        T = g.U.shape[1]
        Ut = g.U.copy()
        Ut[0, T - 1] *= -1                                  # antiperiodic in time
        self.fwd = [mx.array(Ut[mu].astype(np.complex64)) for mu in range(4)]
        self.bwd = [mx.array(np.roll(np.conj(np.swapaxes(Ut[mu], -1, -2)), 1, axis=mu).astype(np.complex64))
                    for mu in range(4)]
        self.Pm = [mx.array((ONE4 - GAMMA[m]).astype(np.complex64)) for m in range(4)]
        self.Pp = [mx.array((ONE4 + GAMMA[m]).astype(np.complex64)) for m in range(4)]
        self.g5 = mx.array(GAMMA5.astype(np.complex64))
        self.kappa = kappa
        self.shape = g.U.shape[1:5]

    def _colour(self, U, X):
        s = X.shape
        return (U @ X.reshape(s[:4] + (3, -1))).reshape(s)

    def apply(self, X):
        mx = self.mx
        out = X
        for mu in range(4):
            hf = self._colour(self.fwd[mu], mx.roll(X, -1, axis=mu))
            hb = self._colour(self.bwd[mu], mx.roll(X, 1, axis=mu))
            out = out - self.kappa * (self.Pm[mu] @ hf + self.Pp[mu] @ hb)
        return out

    def apply_dag(self, X):
        return self.g5 @ self.apply(self.g5 @ X)

    def smear(self, X, kappa_w: float = 0.25, steps: int = 40):
        """Wuppertal smearing: a gauge-covariant diffusion of the source over space, which makes
        it overlap mostly with the lowest hadron state instead of every excitation."""
        mx = self.mx
        for _ in range(steps):
            hop = 0
            for mu in (1, 2, 3):
                hop = hop + self._colour(self.fwd[mu], mx.roll(X, -1, axis=mu)) \
                          + self._colour(self.bwd[mu], mx.roll(X, 1, axis=mu))
            X = (X + kappa_w * hop) / (1 + 6 * kappa_w)
        return X

    def solve(self, B, tol=1e-6, maxiter=5000):
        """Conjugate gradient on D†D, one independent CG per column."""
        mx = self.mx
        axes = tuple(range(B.ndim - 1))
        dot = lambda a, b: mx.real(mx.sum(mx.conj(a) * b, axis=axes))
        rhs = self.apply_dag(B)
        x = mx.zeros_like(rhs)
        r, p = rhs, rhs
        rr = dot(r, r)
        stop = (tol ** 2) * rr
        for it in range(maxiter):
            Ap = self.apply_dag(self.apply(p))
            alpha = rr / dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = dot(r, r)
            p = r + (rr_new / rr) * p
            rr = rr_new
            if it % 10 == 0:
                mx.eval(x, r, p, rr)
                if bool(mx.all(rr < stop)):
                    return x, it + 1
        return x, maxiter


def point_propagator_gpu(D: WilsonDiracGPU, smeared: bool = False):
    """S[t, x, y, z, s, c, s0, c0] from a source at the origin (point, or Wuppertal-smeared),
    all 12 columns in one solve."""
    B = np.zeros(D.shape + (3, 4, 12), np.complex64)
    for s0 in range(4):
        for c0 in range(3):
            B[0, 0, 0, 0, c0, s0, s0 * 3 + c0] = 1.0
    B = D.mx.array(B)
    if smeared:
        B = D.smear(B)
    X, iters = D.solve(B)
    X = np.array(X).astype(complex).reshape(D.shape + (3, 4, 4, 3))       # (…, c, s, s0, c0)
    return X.transpose(0, 1, 2, 3, 5, 4, 6, 7), iters


def point_propagator(D: WilsonDirac):
    """S[t, x, y, z, s, c, s0, c0] from a point source at the origin."""
    S = np.zeros(D.shape + (4, 3, 4, 3), complex)
    iters = 0
    for s0 in range(4):
        for c0 in range(3):
            b = np.zeros(D.shape + (4, 3), complex)
            b[0, 0, 0, 0, s0, c0] = 1.0
            x, n = D.solve(b)
            S[..., s0, c0] = x
            iters += n
    return S, iters


def correlators(S) -> dict:
    """Zero-momentum pion and rho correlators C(t)."""
    # pion: Σ_x Tr[S S†] (γ5-hermiticity)
    pion = np.einsum("txyzscSC,txyzscSC->t", S, np.conj(S)).real
    # rho: Σ_x Σ_i Tr[γ_i S γ_i γ5 S† γ5]
    rho = np.zeros(S.shape[0])
    g5 = GAMMA5
    for i in (1, 2, 3):
        gi = GAMMA[i]
        A = np.einsum("st,...tcSC->...scSC", gi @ g5, S)            # (γ_i γ5) S
        A = np.einsum("...scSC,SR->...scRC", A, g5 @ gi)             # … (γ5 γ_i)
        rho += np.einsum("txyzscSC,txyzscSC->t", A, np.conj(S)).real
    return {"pion": pion, "rho": np.abs(rho) / 3}


def effective_mass(C: np.ndarray) -> np.ndarray:
    """m(t) from the periodic (cosh) form of a correlator."""
    T = len(C)
    m = np.full(T // 2 - 1, np.nan)
    for t in range(1, T // 2):
        ratio = (C[t - 1] + C[t + 1]) / (2 * C[t])
        if ratio >= 1:
            m[t - 1] = math.acosh(ratio)
    return m


def plateau(m: np.ndarray, width: int = 3) -> float:
    """Mass from the last ``width`` time slices before T/2, where excited states have died away."""
    vals = m[np.isfinite(m)][-width:]
    return float(np.mean(vals)) if len(vals) else float("nan")


def spectrum(beta: float = 5.7, L: int = 6, T: int = 12, kappas=(0.150, 0.155, 0.160), n_configs: int = 6,
             therm: int = 60, spacing: int = 20, seed: int = 11, progress=None) -> dict:
    """Pion and rho masses (lattice units) at several quark masses, averaged over gluon configurations."""
    g = GaugeField(L, T, beta, seed=seed)
    for _ in range(therm):
        g.sweep()
    acc = {k: {"pion": np.zeros(T), "rho": np.zeros(T)} for k in kappas}
    for n in range(n_configs):
        for _ in range(spacing):
            g.sweep()
        for k in kappas:
            S, _ = point_propagator_gpu(WilsonDiracGPU(g, k), smeared=True)
            C = correlators(S)
            acc[k]["pion"] += C["pion"] / n_configs
            acc[k]["rho"] += C["rho"] / n_configs
        if progress:
            progress(n + 1, n_configs)
    out = {"beta": beta, "L": L, "T": T, "kappas": list(kappas), "pion": [], "rho": [], "correlators": acc}
    for k in kappas:
        out["pion"].append(plateau(effective_mass(acc[k]["pion"])))
        out["rho"].append(plateau(effective_mass(acc[k]["rho"])))
    # chiral limit: m_π² is linear in 1/κ; it vanishes at κ_c
    inv = 1 / np.array(kappas)
    mp2 = np.array(out["pion"]) ** 2
    slope, icpt = np.polyfit(inv, mp2, 1)
    out["kappa_c"] = float(1 / (-icpt / slope))
    out["rho_chiral"] = float(np.polyval(np.polyfit(inv, out["rho"], 1), 1 / out["kappa_c"]))
    return out
