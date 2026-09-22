"""Electron–nucleus and nucleus–nucleus interactions.

Nuclei are exact point charges. Their potential is defined by its Fourier
coefficients  −4πZ/k² · e^{−ik·R}  (with the same open-boundary truncated
kernel as the Hartree term), keeping the components the grid can represent.
A standard spectral filter (Hou & Li 2007: exp(−36 (k/k_max)^36)) rolls off
the last few percent below the grid's Nyquist frequency to suppress Gibbs
ringing, which otherwise makes the energy depend on where a nucleus sits
relative to grid points (the "egg-box" effect). The grid's resolution is
the *only* approximation: as h → 0 this becomes the exact Coulomb potential.

A point nucleus needs grid spacing h ≲ 0.6 / Z for its innermost electrons,
so this is used for H and He. Heavier atoms use pseudopotentials derived from
this engine's own all-electron atom solver (see engine/atoms/pseudo.py): a
smooth local part, handled here in the same Fourier framework, plus separable
non-local projectors (:class:`NonlocalProjectors`).

Ion–ion repulsion uses exact point charges (valence charges for pseudo-ions).
"""

from __future__ import annotations

import numpy as np

from ..atoms import species
from ..atoms.pseudo import R_GAUSS, real_harmonics_k
from ..core.grid import Grid


class NuclearField:
    """Ion potential (point nuclei or pseudo-ions) and its Hellmann–Feynman forces."""

    def __init__(self, grid: Grid) -> None:
        self.grid = grid
        N, h = grid.N, grid.h
        M = 2 * N
        self.M = M
        k = 2 * np.pi * np.fft.fftfreq(M, d=h)
        kz = 2 * np.pi * np.fft.rfftfreq(M, d=h)
        self.kx = k[:, None, None]
        self.ky = k[None, :, None]
        self.kz = kz[None, None, :]
        kk = np.sqrt(self.kx ** 2 + self.ky ** 2 + self.kz ** 2)
        Rc = grid.L
        with np.errstate(divide="ignore", invalid="ignore"):
            kern = 4 * np.pi * (1 - np.cos(kk * Rc)) / kk ** 2
        kern[0, 0, 0] = 2 * np.pi * Rc ** 2
        # Drop Nyquist planes: their phase is ambiguous for off-grid nuclei, which
        # would make the potential and its force inconsistent.
        kern[M // 2, :, :] = 0.0
        kern[:, M // 2, :] = 0.0
        kern[:, :, -1] = 0.0
        kmax = np.pi / h
        self.filter = np.exp(-36.0 * np.minimum(kk / kmax, 1.5) ** 36)
        self.kk = kk
        self.kernel = kern * self.filter
        self._form: dict[int, np.ndarray] = {}
        self._core_form: dict = {}
        # Padded-grid index 0 sits at x0 = -N/2 * h (same as the real grid).
        self.x0 = -(N // 2) * h
        # rfft half-spectrum weights for Parseval sums.
        w = np.full(kz.shape[0], 2.0)
        w[0] = 1.0
        if M % 2 == 0:
            w[-1] = 1.0
        self.weights = w[None, None, :]

    def _phase(self, R):
        d = np.asarray(R, dtype=float) - self.x0
        return np.exp(-1j * (self.kx * d[0] + self.ky * d[1] + self.kz * d[2]))

    def form_factor(self, Z: int) -> np.ndarray:
        """Fourier transform of one ion's potential (continuum convention)."""
        if Z not in self._form:
            if not species.is_pseudized(Z):
                self._form[Z] = -Z * self.kernel
            else:
                pp = species.pseudopotential(Z)
                q = np.linspace(0.0, float(self.kk.max()) * 1.001, 2048)
                short = np.interp(self.kk, q, pp.local_short_range_q(q))
                gauss = np.exp(-self.kk ** 2 * R_GAUSS ** 2 / 4)
                self._form[Z] = -pp.Z_val * self.kernel * gauss + short * self.filter
        return self._form[Z]

    def _vhat(self, Z, R):
        # Fourier coefficients of one ion's potential on the padded grid (DFT convention).
        return self.form_factor(Z) * self._phase(R) / self.grid.dV

    def potential(self, charges, positions) -> np.ndarray:
        N = self.grid.N
        acc = np.zeros(self.kernel.shape, dtype=complex)
        for Z, R in zip(charges, positions):
            acc += self._vhat(Z, R)
        v = np.fft.irfftn(acc, s=(self.M,) * 3, axes=(0, 1, 2))
        return v[:N, :N, :N]

    def forces(self, charges, positions, rho: np.ndarray, hat=None) -> np.ndarray:
        """F_I = −∂/∂R_I ∫ ρ V_I  for fixed electron density ρ (or, with ``hat``, of ∫ ρ f_I for
        any other per-ion field f, such as the partial core density)."""
        hat = hat or self._vhat
        N, M = self.grid.N, self.M
        padded = np.zeros((M,) * 3)
        padded[:N, :N, :N] = rho
        rhat = np.fft.rfftn(padded)
        out = np.zeros((len(charges), 3))
        for i, (Z, R) in enumerate(zip(charges, positions)):
            # ∂V̂/∂R = −ik V̂;  E = dV/M³ Σ_k w Re[conj(ρ̂) V̂]
            prod = self.weights * np.conj(rhat) * hat(Z, R)
            for a, ka in enumerate((self.kx, self.ky, self.kz)):
                dE = np.real(np.sum(prod * (-1j * ka))) * self.grid.dV / M ** 3
                out[i, a] = -dE
        return out


    # ------------------------------------------------------------ core correction
    def _core_hat(self, Z, R):
        if Z not in self._core_form:
            pp = species.pseudopotential(Z) if species.is_pseudized(Z) else None
            if pp is None or pp.rho_core is None:
                self._core_form[Z] = None
            else:
                q = np.linspace(0.0, float(self.kk.max()) * 1.001, 2048)
                self._core_form[Z] = np.interp(self.kk, q, pp.core_density_q(q)) * self.filter
        f = self._core_form[Z]
        return 0 * self.kk if f is None else f * self._phase(R) / self.grid.dV

    def core_density(self, charges, positions) -> np.ndarray | None:
        """Sum of the ions' partial core densities on the grid (None if no ion has one)."""
        acc, any_core = np.zeros(self.kernel.shape, dtype=complex), False
        for Z, R in zip(charges, positions):
            h = self._core_hat(Z, R)
            if self._core_form.get(Z) is not None:
                acc += h
                any_core = True
        if not any_core:
            return None
        N = self.grid.N
        return np.fft.irfftn(acc, s=(self.M,) * 3, axes=(0, 1, 2))[:N, :N, :N]

    def core_forces(self, charges, positions, v_xc_mean: np.ndarray) -> np.ndarray:
        """F_I = −∫ v̄_xc ∂ρ_core,I/∂R_I : the core correction's share of the force."""
        return self.forces(charges, positions, v_xc_mean, hat=self._core_hat)


class NonlocalProjectors:
    """Kleinman–Bylander projectors  V_NL = Σ_p |β_p⟩ E_p ⟨β_p|  on the grid.

    Each projector is built from its radial Fourier transform with the ion's
    phase, so it moves smoothly with the nucleus (no grid pinning), and is
    band-limited by the same spectral filter as the local potential.
    """

    def __init__(self, grid: Grid, charges, positions) -> None:
        self.grid = grid
        b = grid.backend
        N, h = grid.N, grid.h
        k = 2 * np.pi * np.fft.fftfreq(N, d=h)
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        KK = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
        filt = np.exp(-36.0 * np.minimum(KK / (np.pi / h), 1.5) ** 36)
        x0 = -(N // 2) * h
        self._k = (KX, KY, KZ)
        rows, self.E, self.atom = [], [], []
        self._fq = []  # Fourier coefficients (for forces)
        tables: dict[tuple[int, int], np.ndarray] = {}
        for i, (Z, R) in enumerate(zip(charges, positions)):
            if not species.is_pseudized(Z):
                continue
            pp = species.pseudopotential(Z)
            d = np.asarray(R, dtype=float) - x0
            phase = np.exp(-1j * (KX * d[0] + KY * d[1] + KZ * d[2]))
            for l in pp.nonlocal_channels:
                if (Z, l) not in tables:
                    q = np.linspace(0.0, float(KK.max()) * 1.001, 2048)
                    tables[(Z, l)] = np.interp(KK, q, pp.projector_q(l, q)) * filt
                radial = tables[(Z, l)]
                with np.errstate(invalid="ignore", divide="ignore"):
                    kn = np.where(KK > 0, 1.0 / KK, 0.0)
                angular = real_harmonics_k(l, [KX * kn, KY * kn, KZ * kn])   # Y_lm(k̂) · (−i)^l
                for ang in angular:
                    F = radial * ang * phase / grid.dV
                    rows.append(np.real(np.fft.ifftn(F)))
                    self._fq.append(F)
                    self.E.append(pp.kb_energy[l])
                    self.atom.append(i)
        self.count = len(rows)
        self.E = np.array(self.E)
        if self.count:
            self.B_np = np.array(rows).reshape(self.count, -1)
            self.B = b.asarray(self.B_np)
            self.E_b = b.asarray(self.E)

    def apply(self, X):
        """V_NL X for a block X of shape (nb, N, N, N) (orbitals in 2-norm units)."""
        b = self.grid.backend
        nb = X.shape[0]
        Xf = X.reshape(nb, -1)
        c = (Xf @ self.B.T) * self.grid.dV                   # ⟨β|ψ⟩·√dV (2-norm units)
        return ((c * self.E_b) @ self.B).reshape(X.shape)

    def energy(self, X, occ) -> float:
        # Orbitals are stored as ψ√dV, so ⟨β|ψ⟩ = Σ β X · √dV.
        b = self.grid.backend
        c = b.to_numpy((X.reshape(X.shape[0], -1) @ self.B.T)) * np.sqrt(self.grid.dV)
        return float(np.sum(np.asarray(occ)[:, None] * self.E[None, :] * c * c))

    def forces(self, X, occ, n_atoms: int) -> np.ndarray:
        """F_I = −Σ_i f_i ∂/∂R_I ⟨ψ_i|V_NL|ψ_i⟩."""
        b = self.grid.backend
        Xn = b.to_numpy(X).reshape(X.shape[0], -1)
        sq = np.sqrt(self.grid.dV)
        c = (Xn @ self.B_np.T) * sq                          # (nb, P) = ⟨β|ψ⟩
        F = np.zeros((n_atoms, 3))
        for p in range(self.count):
            for a, Ka in enumerate(self._k):
                # ∂β(r − R)/∂R = −∇β ;  ∇β ↔ i k F(k)
                grad = np.real(np.fft.ifftn(1j * Ka * self._fq[p])).ravel()
                dc = -(Xn @ grad) * sq                       # ⟨ψ|∂β/∂R⟩
                F[self.atom[p], a] -= float(np.sum(np.asarray(occ) * 2 * self.E[p] * c[:, p] * dc))
        return F


def ion_ion(charges, positions) -> tuple[float, np.ndarray]:
    """Point-charge ion–ion energy and forces (valence charges for pseudo-ions)."""
    P = np.asarray(positions, dtype=float)
    Zs = np.asarray([species.valence_charge(Z) for Z in charges], dtype=float)
    E = 0.0
    F = np.zeros_like(P)
    for i in range(len(Zs)):
        for j in range(i + 1, len(Zs)):
            d = P[i] - P[j]
            r = float(np.linalg.norm(d))
            E += Zs[i] * Zs[j] / r
            f = Zs[i] * Zs[j] * d / r ** 3
            F[i] += f
            F[j] -= f
    return E, F
