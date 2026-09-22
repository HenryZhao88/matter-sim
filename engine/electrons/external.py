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
so this is used for H and He; heavier atoms use pseudopotentials derived
from this engine's own all-electron atom solver.

Nucleus–nucleus repulsion uses exact point charges.
"""

from __future__ import annotations

import numpy as np

from ..core.grid import Grid


class NuclearField:
    """Point-nucleus potential and Hellmann–Feynman forces on a grid."""

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
        kern *= np.exp(-36.0 * np.minimum(kk / kmax, 1.5) ** 36)
        self.kernel = kern
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

    def _vhat(self, Z, R):
        # Fourier coefficients of one nucleus's potential on the padded grid (DFT convention).
        return -Z * self.kernel * self._phase(R) / self.grid.dV

    def potential(self, charges, positions) -> np.ndarray:
        N = self.grid.N
        acc = np.zeros(self.kernel.shape, dtype=complex)
        for Z, R in zip(charges, positions):
            acc += self._vhat(Z, R)
        v = np.fft.irfftn(acc, s=(self.M,) * 3, axes=(0, 1, 2))
        return v[:N, :N, :N]

    def forces(self, charges, positions, rho: np.ndarray) -> np.ndarray:
        """F_I = −∂/∂R_I ∫ ρ V_I  for fixed electron density ρ."""
        N, M = self.grid.N, self.M
        padded = np.zeros((M,) * 3)
        padded[:N, :N, :N] = rho
        rhat = np.fft.rfftn(padded)
        out = np.zeros((len(charges), 3))
        for i, (Z, R) in enumerate(zip(charges, positions)):
            # ∂V̂/∂R = −ik V̂;  E = dV/M³ Σ_k w Re[conj(ρ̂) V̂]
            prod = self.weights * np.conj(rhat) * self._vhat(Z, R)
            for a, ka in enumerate((self.kx, self.ky, self.kz)):
                dE = np.real(np.sum(prod * (-1j * ka))) * self.grid.dV / M ** 3
                out[i, a] = -dE
        return out


def ion_ion(charges, positions) -> tuple[float, np.ndarray]:
    """Point-charge nucleus–nucleus energy and forces."""
    P = np.asarray(positions, dtype=float)
    Zs = np.asarray(charges, dtype=float)
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
