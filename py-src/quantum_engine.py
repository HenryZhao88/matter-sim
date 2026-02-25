"""Mode C quantum engine — TDSE split-operator solver for hydrogen-like atoms.

Solves the time-dependent Schrödinger equation:

    iℏ ∂ψ/∂t = Ĥψ = (-ℏ²/2m ∇² + V(r)) ψ

using the split-operator Fourier method (second-order Trotter decomposition):

    ψ(t+dt) = e^{-iVdt/2ℏ} F⁻¹[ e^{-iTdt/ℏ} F[ e^{-iVdt/2ℏ} ψ(t) ] ]

where:
    T = ℏ²k²/2m   (kinetic energy, diagonal in momentum space)
    V(r) = -Z/√(r²+ε²) + E·r   (softened Coulomb + external field)

All quantities in atomic units:
    ℏ = m_e = e = 4πε₀ = a₀ = 1
    Energy: 1 Hartree = 27.211 eV
    Length: 1 Bohr = 0.529 Å
    Time:   ℏ/Hartree = 24.188 attoseconds

The method is:
    - Unconditionally stable (unitary propagator, ‖ψ‖² preserved exactly)
    - Symplectic (time-reversal symmetric)
    - O(dt²) global error from Trotter splitting
    - O(N log N) per step via FFT
"""

from __future__ import annotations

import math
import numpy as np


class QuantumWorld:
    """3D quantum simulation of a hydrogen-like atom via TDSE split-operator."""

    def __init__(
        self,
        grid_n: int = 64,
        extent: float = 20.0,
        nuclear_Z: int = 1,
        dt: float = 0.02,
    ) -> None:
        self.grid_n = grid_n
        self.extent = extent          # half-width of box in Bohr radii
        self.nuclear_Z = nuclear_Z    # nuclear charge
        self.dt = dt                  # time step in atomic time units
        self.time = 0.0

        # Current eigenstate labels (informational)
        self.current_n = 1
        self.current_l = 0
        self.current_m = 0

        # Simulation controls
        self.steps_per_tick = 3
        self.slice_axis = "y"         # default: xz plane (shows orbital lobes)
        self.view_mode = "project"    # "slice" or "project"

        # External perturbation
        self.electric_field = np.zeros(3, dtype=np.float64)

        # Observable cache
        self.last_energy: float = 0.0
        self.last_norm: float = 1.0
        self.last_Lz: float = 0.0
        self.observable_interval = 5
        self._obs_counter = 0

        # Build grids and propagators
        self._setup_grids()
        self._build_propagators()
        self.init_eigenstate(1, 0, 0)

    # ------------------------------------------------------------------
    # Grid setup
    # ------------------------------------------------------------------

    def _setup_grids(self) -> None:
        """Create 3D real-space and momentum-space grids."""
        N = self.grid_n
        L = 2.0 * self.extent
        self.dx = L / N

        # Softening: Coulomb singularity removal at grid scale
        self.softening = max(0.2, self.dx * 0.5)

        # Position-space coordinates (cell-centered)
        x = np.linspace(-self.extent + self.dx / 2, self.extent - self.dx / 2, N)
        self.x = x
        self.X, self.Y, self.Zgrid = np.meshgrid(x, x, x, indexing="ij")
        self.R = np.sqrt(self.X ** 2 + self.Y ** 2 + self.Zgrid ** 2)
        self.dV = self.dx ** 3

        # Momentum-space coordinates (via FFT frequencies)
        k = np.fft.fftfreq(N, d=self.dx) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        self.K2 = KX ** 2 + KY ** 2 + KZ ** 2

    # ------------------------------------------------------------------
    # Potential and propagators
    # ------------------------------------------------------------------

    def _build_propagators(self) -> None:
        """(Re)compute the potential and split-operator phase arrays."""
        # Softened Coulomb: V(r) = -Z / sqrt(r^2 + eps^2)
        self.V_coulomb = -float(self.nuclear_Z) / np.sqrt(
            self.R ** 2 + self.softening ** 2
        )

        # Total potential = Coulomb + Stark (electric field coupling)
        self.V = self.V_coulomb.copy()
        if np.any(self.electric_field != 0.0):
            self.V += (
                self.electric_field[0] * self.X
                + self.electric_field[1] * self.Y
                + self.electric_field[2] * self.Zgrid
            )

        # Half-step potential propagator: exp(-i V dt/2)
        self.exp_V_half = np.exp(-1j * self.V * self.dt * 0.5)

        # Full-step kinetic propagator: exp(-i (k^2/2) dt)
        self.exp_T = np.exp(-1j * self.K2 * self.dt * 0.5)

    # ------------------------------------------------------------------
    # Eigenstate initialization
    # ------------------------------------------------------------------

    def init_eigenstate(self, n: int, l: int, m: int) -> None:
        """Initialize ψ as an exact hydrogen eigenstate ψ_nlm."""
        n, l, m = int(n), int(l), int(m)
        # Enforce quantum number constraints
        l = max(0, min(l, n - 1))
        m = max(-l, min(m, l))

        self.current_n = n
        self.current_l = l
        self.current_m = m
        self.time = 0.0
        self._obs_counter = 0

        self.psi = self._hydrogen_eigenstate(n, l, m)

        # Normalize
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dV)
        if norm > 1e-30:
            self.psi /= norm

        self._update_observables()

    def _hydrogen_eigenstate(self, n: int, l: int, m: int) -> np.ndarray:
        """Evaluate ψ_nlm(r,θ,φ) on the Cartesian grid."""
        R_safe = np.maximum(self.R, 1e-10)
        theta = np.arccos(np.clip(self.Zgrid / R_safe, -1.0, 1.0))
        phi = np.arctan2(self.Y, self.X)

        radial = self._radial_part(n, l, R_safe)
        angular = self._spherical_harmonic(l, m, theta, phi)

        return (radial * angular).astype(np.complex128)

    def _radial_part(self, n: int, l: int, r: np.ndarray) -> np.ndarray:
        """Hydrogen radial wavefunction R_nl(r)."""
        rho = 2.0 * r / n
        k = n - l - 1
        alpha = 2 * l + 1

        norm_sq = (2.0 / n) ** 3 * math.factorial(n - l - 1) / (
            2.0 * n * math.gamma(n + l + 1)
        )
        norm = math.sqrt(norm_sq)

        lag = self._assoc_laguerre_arr(k, alpha, rho)
        return norm * np.exp(-rho / 2.0) * np.power(rho, float(l)) * lag

    @staticmethod
    def _assoc_laguerre_arr(k: int, alpha: int, x: np.ndarray) -> np.ndarray:
        """Associated Laguerre polynomial L_k^alpha(x) via stable recurrence."""
        if k == 0:
            return np.ones_like(x)
        lm2 = np.ones_like(x)
        lm1 = 1.0 + alpha - x
        if k == 1:
            return lm1
        for j in range(2, k + 1):
            l_new = ((2 * j - 1 + alpha - x) * lm1 - (j - 1 + alpha) * lm2) / j
            lm2, lm1 = lm1, l_new
        return lm1

    def _spherical_harmonic(
        self, l: int, m: int, theta: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        """Complex spherical harmonic Y_l^m(θ,φ) with Condon-Shortley phase."""
        m_abs = abs(m)
        x = np.cos(theta)
        plm = self._assoc_legendre_arr(l, m_abs, x)

        norm = math.sqrt(
            (2 * l + 1)
            / (4.0 * math.pi)
            * math.factorial(l - m_abs)
            / math.factorial(l + m_abs)
        )

        # Y_l^m = (-1)^m * norm * P_l^|m| * e^{imφ}  for m >= 0
        # Y_l^{-|m|} = (-1)^|m| * (Y_l^|m|)* but using the general formula:
        if m >= 0:
            return norm * plm * np.exp(1j * m * phi)
        else:
            sign = (-1.0) ** m_abs
            return sign * norm * plm * np.exp(1j * m * phi)

    @staticmethod
    def _assoc_legendre_arr(l: int, m: int, x: np.ndarray) -> np.ndarray:
        """Associated Legendre polynomial P_l^m(x) with Condon-Shortley phase."""
        m = abs(m)

        # P_m^m
        pmm = np.ones_like(x)
        if m > 0:
            somx2 = np.sqrt(np.clip((1.0 - x) * (1.0 + x), 0.0, None))
            fact = 1.0
            for _i in range(1, m + 1):
                pmm = pmm * (-fact) * somx2
                fact += 2.0

        if l == m:
            return pmm

        # P_{m+1}^m
        pm1m = x * (2 * m + 1) * pmm
        if l == m + 1:
            return pm1m

        # General recurrence
        for ll in range(m + 2, l + 1):
            pll = ((2 * ll - 1) * x * pm1m - (ll + m - 1) * pmm) / (ll - m)
            pmm = pm1m
            pm1m = pll
        return pm1m

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Advance ψ by one time step using split-operator method.

        ψ(t+dt) = e^{-iVdt/2} F⁻¹[ e^{-iTdt} F[ e^{-iVdt/2} ψ(t) ] ]

        This is a second-order symmetric Trotter decomposition.
        The propagator is exactly unitary — ‖ψ‖² is preserved to machine precision.
        """
        # Half-step potential (position space)
        self.psi *= self.exp_V_half

        # Full-step kinetic (momentum space)
        psi_k = np.fft.fftn(self.psi)
        psi_k *= self.exp_T
        self.psi = np.fft.ifftn(psi_k)

        # Half-step potential (position space)
        self.psi *= self.exp_V_half

        self.time += self.dt

        # Update observables periodically (expensive, not every step)
        self._obs_counter += 1
        if self._obs_counter >= self.observable_interval:
            self._obs_counter = 0
            self._update_observables()

    def step_multi(self, count: int | None = None) -> None:
        """Advance by multiple steps (default: self.steps_per_tick)."""
        if count is None:
            count = self.steps_per_tick
        for _ in range(max(1, count)):
            self.step()

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------

    def _update_observables(self) -> None:
        """Compute ⟨H⟩, ⟨Lz⟩, and ‖ψ‖²."""
        rho = np.abs(self.psi) ** 2
        self.last_norm = float(np.sum(rho) * self.dV)
        norm = max(1e-30, self.last_norm)

        # Potential energy: ⟨V⟩ = ∫ V|ψ|² d³r
        E_potential = float(np.sum(self.V * rho).real * self.dV)

        # Kinetic energy via momentum space:
        # ⟨T⟩ = (1/2) Σ_k |k|² |ψ̃_k|² · (dx³/N³)
        # where ψ̃_k is the numpy FFT of ψ (unnormalized DFT).
        psi_k = np.fft.fftn(self.psi)
        N3 = self.grid_n ** 3
        E_kinetic = float(
            0.5 * np.sum(self.K2 * np.abs(psi_k) ** 2).real * self.dV / N3
        )

        self.last_energy = (E_potential + E_kinetic) / norm

        # Angular momentum Lz = -i(x ∂/∂y - y ∂/∂x) via central differences
        dpsi_dx = np.zeros_like(self.psi)
        dpsi_dy = np.zeros_like(self.psi)
        dpsi_dx[1:-1, :, :] = (self.psi[2:, :, :] - self.psi[:-2, :, :]) / (
            2.0 * self.dx
        )
        dpsi_dy[:, 1:-1, :] = (self.psi[:, 2:, :] - self.psi[:, :-2, :]) / (
            2.0 * self.dx
        )
        Lz_integrand = np.conj(self.psi) * (-1j) * (
            self.X * dpsi_dy - self.Y * dpsi_dx
        )
        self.last_Lz = float(np.sum(Lz_integrand).real * self.dV / norm)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def get_density_2d(self) -> np.ndarray:
        """Return 2D array of |ψ|² (slice or projection) for rendering."""
        rho = np.abs(self.psi) ** 2
        ax = self.slice_axis

        if self.view_mode == "project":
            # Integrate along the chosen axis → column density
            if ax == "z":
                return np.sum(rho, axis=2) * self.dx
            elif ax == "y":
                return np.sum(rho, axis=1) * self.dx
            else:
                return np.sum(rho, axis=0) * self.dx
        else:
            # Slice through the grid center
            mid = self.grid_n // 2
            if ax == "z":
                return rho[:, :, mid]
            elif ax == "y":
                return rho[:, mid, :]
            else:
                return rho[mid, :, :]

    # ------------------------------------------------------------------
    # State manipulation
    # ------------------------------------------------------------------

    def set_electric_field(self, ex: float, ey: float, ez: float) -> None:
        """Set external electric field for Stark effect (atomic units)."""
        self.electric_field = np.array([ex, ey, ez], dtype=np.float64)
        self._build_propagators()

    def set_nuclear_Z(self, Z: int) -> None:
        """Set nuclear charge (Z=1 hydrogen, Z=2 He+, etc.)."""
        self.nuclear_Z = int(Z)
        self._build_propagators()

    def set_dt(self, dt: float) -> None:
        """Set time step (atomic units). Smaller = more accurate, slower."""
        self.dt = float(dt)
        self._build_propagators()

    def resize_grid(self, grid_n: int) -> None:
        """Change grid resolution and reinitialize eigenstate."""
        saved = (self.current_n, self.current_l, self.current_m)
        self.grid_n = int(grid_n)
        self._setup_grids()
        self._build_propagators()
        self.init_eigenstate(*saved)

    def superpose_state(self, n2: int, l2: int, m2: int, weight: float = 0.5) -> None:
        """Create superposition: √(1-w²)|current⟩ + w|n2,l2,m2⟩."""
        psi2 = self._hydrogen_eigenstate(n2, l2, m2)
        norm2 = np.sqrt(np.sum(np.abs(psi2) ** 2) * self.dV)
        if norm2 > 1e-30:
            psi2 /= norm2

        w = float(np.clip(weight, 0.0, 1.0))
        self.psi = math.sqrt(1.0 - w * w) * self.psi + w * psi2

        # Renormalize
        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dV)
        if norm > 1e-30:
            self.psi /= norm

        self._update_observables()

    def measure_position(self) -> tuple[float, float, float]:
        """Perform a position measurement (Born rule).

        Collapses the wavefunction to a narrow Gaussian centered at the
        randomly-sampled measurement outcome. This is textbook quantum
        measurement: probability of outcome x is |ψ(x)|².
        """
        rho = np.abs(self.psi) ** 2
        rho_flat = rho.ravel()
        total = rho_flat.sum()
        if total < 1e-30:
            return 0.0, 0.0, 0.0
        rho_flat = rho_flat / total

        idx = np.random.choice(len(rho_flat), p=rho_flat)
        ix, iy, iz = np.unravel_index(idx, rho.shape)

        x0 = float(self.x[ix])
        y0 = float(self.x[iy])
        z0 = float(self.x[iz])

        # Collapse to narrow Gaussian at measurement location
        sigma = self.dx * 2.0
        self.psi = np.exp(
            -(
                (self.X - x0) ** 2
                + (self.Y - y0) ** 2
                + (self.Zgrid - z0) ** 2
            )
            / (2.0 * sigma ** 2)
        ).astype(np.complex128)

        norm = np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dV)
        if norm > 1e-30:
            self.psi /= norm

        self._update_observables()
        return x0, y0, z0

    # ------------------------------------------------------------------
    # Info / status
    # ------------------------------------------------------------------

    def get_info_text(self) -> str:
        """Formatted status string for UI overlay."""
        exact_E = -float(self.nuclear_Z) ** 2 / (2.0 * self.current_n ** 2)
        t_as = self.time * 24.188  # atomic units → attoseconds
        return (
            f"Mode C: TDSE Split-Operator | "
            f"n={self.current_n} l={self.current_l} m={self.current_m} | "
            f"Z={self.nuclear_Z} | Grid: {self.grid_n}\u00b3 | "
            f"\u27e8E\u27e9={self.last_energy:.6f} Ha "
            f"(exact: {exact_E:.6f} Ha) | "
            f"\u27e8Lz\u27e9={self.last_Lz:.4f} \u210f "
            f"(expect: {self.current_m}) | "
            f"\u2016\u03c8\u2016\u00b2={self.last_norm:.8f} | "
            f"t={self.time:.3f} a.u. ({t_as:.1f} as) | "
            f"dt={self.dt} | steps/frame={self.steps_per_tick}"
        )

    def get_info_dict(self) -> dict:
        """Return observables as a dict for programmatic access."""
        exact_E = -float(self.nuclear_Z) ** 2 / (2.0 * self.current_n ** 2)
        return {
            "n": self.current_n,
            "l": self.current_l,
            "m": self.current_m,
            "Z": self.nuclear_Z,
            "grid_n": self.grid_n,
            "energy": self.last_energy,
            "exact_energy": exact_E,
            "Lz": self.last_Lz,
            "norm": self.last_norm,
            "time_au": self.time,
            "time_as": self.time * 24.188,
            "dt": self.dt,
            "steps_per_tick": self.steps_per_tick,
            "electric_field": self.electric_field.tolist(),
        }
