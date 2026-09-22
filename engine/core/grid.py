"""Uniform real-space grid with spectral (FFT) operators.

The simulation box is a cube of side ``L`` centred on the origin, sampled on
``N`` points per axis (spacing ``h = L / N``). Point ``i`` sits at
``(i - N/2) * h`` so the origin is a grid point.

Two operators live here because they define the geometry of space:

- the kinetic energy  T = -1/2 ∇²,  applied exactly in Fourier space, and
- the Coulomb (Hartree) potential of a charge density with *open* boundaries,
  i.e. no interaction with periodic images. It uses a zero-padded FFT and the
  spherically truncated Coulomb kernel  4π/k² (1 - cos(k R_c)),  R_c = L,
  which is exact while all charge fits inside a sphere of diameter L.
"""

from __future__ import annotations

import math

import numpy as np

from .backend import Backend


def fft_friendly(n: int) -> int:
    """Smallest even integer >= n whose prime factors are only 2, 3, 5."""
    n = max(8, n + (n % 2))
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 2


class Grid:
    def __init__(self, L: float, h: float, backend: Backend) -> None:
        self.backend = backend
        self.N = fft_friendly(math.ceil(L / h))
        self.L = float(L)
        self.h = self.L / self.N
        self.dV = self.h ** 3
        self.shape = (self.N,) * 3

        x = (np.arange(self.N) - self.N // 2) * self.h
        self.axis = x
        self.coords = np.meshgrid(x, x, x, indexing="ij")  # float64, CPU

        k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.h)
        kz = 2 * np.pi * np.fft.rfftfreq(self.N, d=self.h)
        k2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2
        self.k2_np = k2
        self.k2 = backend.asarray(k2)
        self.half_k2 = backend.asarray(0.5 * k2)
        self.ekin_max = 0.5 * float(k2.max())

        # Open-boundary Coulomb kernel on the doubled (zero-padded) grid.
        M = 2 * self.N
        kp = 2 * np.pi * np.fft.fftfreq(M, d=self.h)
        kpz = 2 * np.pi * np.fft.rfftfreq(M, d=self.h)
        kk = np.sqrt(kp[:, None, None] ** 2 + kp[None, :, None] ** 2 + kpz[None, None, :] ** 2)
        Rc = self.L
        with np.errstate(divide="ignore", invalid="ignore"):
            kern = 4 * np.pi * (1 - np.cos(kk * Rc)) / kk ** 2
        kern[0, 0, 0] = 2 * np.pi * Rc ** 2
        self._coulomb_kernel = backend.asarray(kern)

    # ---------------------------------------------------------------- basics
    def integrate(self, f) -> float:
        """∫ f dV, accumulated in float64 on the CPU."""
        return float(np.sum(self.backend.to_numpy(f))) * self.dV

    def r_from(self, center) -> np.ndarray:
        X, Y, Z = self.coords
        return np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)

    # ------------------------------------------------------------- operators
    def kinetic(self, psi):
        """T ψ = -½∇²ψ for a batch of fields shaped (..., N, N, N)."""
        b = self.backend
        return b.irfftn(self.half_k2 * b.rfftn(psi), psi.shape[-3:])

    def hartree(self, rho):
        """Electrostatic potential of charge density ``rho`` (open boundaries).

        ``rho`` may carry leading batch dimensions: (..., N, N, N).
        """
        b, N = self.backend, self.N
        lead = [(0, 0)] * (rho.ndim - 3)
        padded = b.xp.pad(rho, lead + [(0, N), (0, N), (0, N)])
        v = b.irfftn(self._coulomb_kernel * b.rfftn(padded), (2 * N,) * 3)
        return v[..., :N, :N, :N]
