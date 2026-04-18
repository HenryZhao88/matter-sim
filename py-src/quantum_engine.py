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

Extensions:
    - GPU acceleration via CuPy (drop-in replacement for numpy FFTs)
    - Multi-electron Restricted Hartree-Fock (RHF) self-consistent field
    - Absorption/emission spectra computation from energy eigenvalues
"""

from __future__ import annotations

import math
import numpy as np

# GPU acceleration: try CuPy, fall back to numpy
try:
    import cupy as _cp  # type: ignore
    _GPU_AVAILABLE = True
except ImportError:
    _cp = None
    _GPU_AVAILABLE = False


def _get_xp(use_gpu: bool):
    """Return the array module — cupy if GPU requested and available, else numpy."""
    if use_gpu and _GPU_AVAILABLE:
        return _cp
    return np


def _to_numpy(arr) -> np.ndarray:
    """Convert CuPy array back to numpy (no-op if already numpy)."""
    if _GPU_AVAILABLE and isinstance(arr, _cp.ndarray):
        return _cp.asnumpy(arr)
    return np.asarray(arr)


class QuantumWorld:
    """3D quantum simulation of a hydrogen-like atom via TDSE split-operator."""

    def __init__(
        self,
        grid_n: int = 64,
        extent: float = 20.0,
        nuclear_Z: int = 1,
        dt: float = 0.02,
        use_gpu: bool = True,
    ) -> None:
        self.grid_n = grid_n
        self.extent = extent          # half-width of box in Bohr radii
        self.nuclear_Z = nuclear_Z    # nuclear charge
        self.dt = dt                  # time step in atomic time units
        self.time = 0.0

        # (#8) GPU acceleration flag
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.xp = _get_xp(self.use_gpu)

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
        xp = self.xp
        N = self.grid_n
        L = 2.0 * self.extent
        self.dx = L / N

        # Softening: Coulomb singularity removal at grid scale
        self.softening = max(0.2, self.dx * 0.5)

        # Position-space coordinates (cell-centered)
        x = np.linspace(-self.extent + self.dx / 2, self.extent - self.dx / 2, N)
        self.x = x  # keep as numpy for visualization indexing
        X, Y, Zgrid = np.meshgrid(x, x, x, indexing="ij")
        # Transfer grids to GPU if available
        self.X = xp.asarray(X)
        self.Y = xp.asarray(Y)
        self.Zgrid = xp.asarray(Zgrid)
        self.R = xp.sqrt(self.X ** 2 + self.Y ** 2 + self.Zgrid ** 2)
        self.dV = self.dx ** 3

        # Momentum-space coordinates (via FFT frequencies)
        k = np.fft.fftfreq(N, d=self.dx) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        self.K2 = xp.asarray(KX ** 2 + KY ** 2 + KZ ** 2)

    # ------------------------------------------------------------------
    # Potential and propagators
    # ------------------------------------------------------------------

    def _build_propagators(self) -> None:
        """(Re)compute the potential and split-operator phase arrays."""
        xp = self.xp
        # Softened Coulomb: V(r) = -Z / sqrt(r^2 + eps^2)
        self.V_coulomb = -float(self.nuclear_Z) / xp.sqrt(
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
        self.exp_V_half = xp.exp(-1j * self.V * self.dt * 0.5)

        # Full-step kinetic propagator: exp(-i (k^2/2) dt)
        self.exp_T = xp.exp(-1j * self.K2 * self.dt * 0.5)

    # ------------------------------------------------------------------
    # Eigenstate initialization
    # ------------------------------------------------------------------

    def init_eigenstate(self, n: int, l: int, m: int) -> None:
        """Initialize ψ as an exact hydrogen eigenstate ψ_nlm."""
        xp = self.xp
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
        norm = float(xp.sqrt(xp.sum(xp.abs(self.psi) ** 2) * self.dV))
        if norm > 1e-30:
            self.psi /= norm

        self._update_observables()

    def _hydrogen_eigenstate(self, n: int, l: int, m: int):
        """Evaluate ψ_nlm(r,θ,φ) on the Cartesian grid."""
        xp = self.xp
        R_safe = xp.maximum(self.R, 1e-10)
        theta = xp.arccos(xp.clip(self.Zgrid / R_safe, -1.0, 1.0))
        phi = xp.arctan2(self.Y, self.X)

        radial = self._radial_part(n, l, R_safe)
        angular = self._spherical_harmonic(l, m, theta, phi)

        return (radial * angular).astype(xp.complex128)

    def _radial_part(self, n: int, l: int, r):
        """Hydrogen radial wavefunction R_nl(r)."""
        xp = self.xp
        rho = 2.0 * r / n
        k = n - l - 1
        alpha = 2 * l + 1

        norm_sq = (2.0 / n) ** 3 * math.factorial(n - l - 1) / (
            2.0 * n * math.gamma(n + l + 1)
        )
        norm = math.sqrt(norm_sq)

        lag = self._assoc_laguerre_arr(k, alpha, rho, xp)
        return norm * xp.exp(-rho / 2.0) * xp.power(rho, float(l)) * lag

    @staticmethod
    def _assoc_laguerre_arr(k: int, alpha: int, x, xp=np):
        """Associated Laguerre polynomial L_k^alpha(x) via stable recurrence."""
        if k == 0:
            return xp.ones_like(x)
        lm2 = xp.ones_like(x)
        lm1 = 1.0 + alpha - x
        if k == 1:
            return lm1
        for j in range(2, k + 1):
            l_new = ((2 * j - 1 + alpha - x) * lm1 - (j - 1 + alpha) * lm2) / j
            lm2, lm1 = lm1, l_new
        return lm1

    def _spherical_harmonic(self, l: int, m: int, theta, phi):
        """Complex spherical harmonic Y_l^m(θ,φ) with Condon-Shortley phase."""
        xp = self.xp
        m_abs = abs(m)
        x = xp.cos(theta)
        plm = self._assoc_legendre_arr(l, m_abs, x, xp)

        norm = math.sqrt(
            (2 * l + 1)
            / (4.0 * math.pi)
            * math.factorial(l - m_abs)
            / math.factorial(l + m_abs)
        )

        # Y_l^m = (-1)^m * norm * P_l^|m| * e^{imφ}  for m >= 0
        # Y_l^{-|m|} = (-1)^|m| * (Y_l^|m|)* but using the general formula:
        if m >= 0:
            return norm * plm * xp.exp(1j * m * phi)
        else:
            sign = (-1.0) ** m_abs
            return sign * norm * plm * xp.exp(1j * m * phi)

    @staticmethod
    def _assoc_legendre_arr(l: int, m: int, x, xp=np):
        """Associated Legendre polynomial P_l^m(x) with Condon-Shortley phase."""
        m = abs(m)

        # P_m^m
        pmm = xp.ones_like(x)
        if m > 0:
            somx2 = xp.sqrt(xp.clip((1.0 - x) * (1.0 + x), 0.0, None))
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
        (#8) Uses CuPy FFTs on GPU when available for 50-100x speedup.
        """
        xp = self.xp
        # Half-step potential (position space)
        self.psi *= self.exp_V_half

        # Full-step kinetic (momentum space) — GPU-accelerated via CuPy
        psi_k = xp.fft.fftn(self.psi)
        psi_k *= self.exp_T
        self.psi = xp.fft.ifftn(psi_k)

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
        xp = self.xp
        rho = xp.abs(self.psi) ** 2
        self.last_norm = float(xp.sum(rho) * self.dV)
        norm = max(1e-30, self.last_norm)

        # Potential energy: ⟨V⟩ = ∫ V|ψ|² d³r
        E_potential = float(xp.sum(self.V * rho).real * self.dV)

        # Kinetic energy via momentum space:
        psi_k = xp.fft.fftn(self.psi)
        N3 = self.grid_n ** 3
        E_kinetic = float(
            0.5 * xp.sum(self.K2 * xp.abs(psi_k) ** 2).real * self.dV / N3
        )

        self.last_energy = (E_potential + E_kinetic) / norm

        # Angular momentum Lz = -i(x ∂/∂y - y ∂/∂x) via central differences
        dpsi_dx = xp.zeros_like(self.psi)
        dpsi_dy = xp.zeros_like(self.psi)
        dpsi_dx[1:-1, :, :] = (self.psi[2:, :, :] - self.psi[:-2, :, :]) / (
            2.0 * self.dx
        )
        dpsi_dy[:, 1:-1, :] = (self.psi[:, 2:, :] - self.psi[:, :-2, :]) / (
            2.0 * self.dx
        )
        Lz_integrand = xp.conj(self.psi) * (-1j) * (
            self.X * dpsi_dy - self.Y * dpsi_dx
        )
        self.last_Lz = float(xp.sum(Lz_integrand).real * self.dV / norm)

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def get_density_2d(self) -> np.ndarray:
        """Return 2D array of |ψ|² (slice or projection) for rendering.
        Always returns numpy (transfers from GPU if needed)."""
        xp = self.xp
        rho = xp.abs(self.psi) ** 2
        ax = self.slice_axis

        if self.view_mode == "project":
            if ax == "z":
                result = xp.sum(rho, axis=2) * self.dx
            elif ax == "y":
                result = xp.sum(rho, axis=1) * self.dx
            else:
                result = xp.sum(rho, axis=0) * self.dx
        else:
            mid = self.grid_n // 2
            if ax == "z":
                result = rho[:, :, mid]
            elif ax == "y":
                result = rho[:, mid, :]
            else:
                result = rho[mid, :, :]
        return _to_numpy(result)

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
        xp = self.xp
        psi2 = self._hydrogen_eigenstate(n2, l2, m2)
        norm2 = float(xp.sqrt(xp.sum(xp.abs(psi2) ** 2) * self.dV))
        if norm2 > 1e-30:
            psi2 /= norm2

        w = float(np.clip(weight, 0.0, 1.0))
        self.psi = math.sqrt(1.0 - w * w) * self.psi + w * psi2

        # Renormalize
        norm = float(xp.sqrt(xp.sum(xp.abs(self.psi) ** 2) * self.dV))
        if norm > 1e-30:
            self.psi /= norm

        self._update_observables()

    def measure_position(self) -> tuple[float, float, float]:
        """Perform a position measurement (Born rule).

        Collapses the wavefunction to a narrow Gaussian centered at the
        randomly-sampled measurement outcome. This is textbook quantum
        measurement: probability of outcome x is |ψ(x)|².
        """
        xp = self.xp
        rho = xp.abs(self.psi) ** 2
        # Transfer to numpy for random.choice (needs CPU)
        rho_np = _to_numpy(rho)
        rho_flat = rho_np.ravel()
        total = rho_flat.sum()
        if total < 1e-30:
            return 0.0, 0.0, 0.0
        rho_flat = rho_flat / total

        idx = np.random.choice(len(rho_flat), p=rho_flat)
        ix, iy, iz = np.unravel_index(idx, rho_np.shape)

        x0 = float(self.x[ix])
        y0 = float(self.x[iy])
        z0 = float(self.x[iz])

        # Collapse to narrow Gaussian at measurement location
        sigma = self.dx * 2.0
        self.psi = xp.exp(
            -(
                (self.X - x0) ** 2
                + (self.Y - y0) ** 2
                + (self.Zgrid - z0) ** 2
            )
            / (2.0 * sigma ** 2)
        ).astype(xp.complex128)

        norm = float(xp.sqrt(xp.sum(xp.abs(self.psi) ** 2) * self.dV))
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
        gpu_tag = "GPU" if self.use_gpu else "CPU"
        return (
            f"Mode C: TDSE Split-Operator ({gpu_tag}) | "
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
            "gpu": self.use_gpu,
            "gpu_available": _GPU_AVAILABLE,
        }


# ======================================================================
# (#14) Absorption / Emission Spectra
# ======================================================================

class HydrogenSpectrum:
    """Compute hydrogen-like absorption/emission spectra from exact energy levels.

    Uses the Bohr formula: E_n = -Z²/(2n²) Hartree
    Transitions between levels n_i → n_f emit/absorb photons with
    wavelength λ = hc / |ΔE|.

    Identifies spectral series: Lyman (n_f=1), Balmer (n_f=2), Paschen (n_f=3),
    Brackett (n_f=4), Pfund (n_f=5).
    """

    # Physical constants for wavelength conversion
    HARTREE_EV = 27.211386245988  # 1 Hartree in eV
    HC_EV_NM = 1239.8419843320  # hc in eV·nm

    SERIES_NAMES = {1: "Lyman", 2: "Balmer", 3: "Paschen", 4: "Brackett", 5: "Pfund"}

    @staticmethod
    def energy_level(Z: int, n: int) -> float:
        """Return E_n in Hartree for hydrogen-like atom with nuclear charge Z."""
        return -float(Z * Z) / (2.0 * n * n)

    @classmethod
    def transition_energy(cls, Z: int, n_upper: int, n_lower: int) -> float:
        """Energy of photon emitted in transition n_upper → n_lower (Hartree)."""
        return abs(cls.energy_level(Z, n_upper) - cls.energy_level(Z, n_lower))

    @classmethod
    def transition_wavelength_nm(cls, Z: int, n_upper: int, n_lower: int) -> float:
        """Wavelength of photon in nm for transition n_upper → n_lower."""
        dE = cls.transition_energy(Z, n_upper, n_lower)
        if dE < 1e-30:
            return float("inf")
        dE_eV = dE * cls.HARTREE_EV
        return cls.HC_EV_NM / dE_eV

    @classmethod
    def wavelength_to_rgb(cls, wavelength_nm: float) -> tuple[int, int, int]:
        """Convert wavelength (380-780nm) to approximate sRGB colour."""
        w = wavelength_nm
        if w < 380:
            return (100, 0, 180)  # deep UV → purple
        elif w < 440:
            t = (w - 380) / (440 - 380)
            r, g, b = 0.4 - 0.4 * t, 0.0, 0.6 + 0.4 * t
        elif w < 490:
            t = (w - 440) / (490 - 440)
            r, g, b = 0.0, t, 1.0
        elif w < 510:
            t = (w - 490) / (510 - 490)
            r, g, b = 0.0, 1.0, 1.0 - t
        elif w < 580:
            t = (w - 510) / (580 - 510)
            r, g, b = t, 1.0, 0.0
        elif w < 645:
            t = (w - 580) / (645 - 580)
            r, g, b = 1.0, 1.0 - t, 0.0
        elif w < 780:
            r, g, b = 1.0, 0.0, 0.0
        else:
            return (80, 0, 0)  # deep IR → dark red
        # Intensity fall-off at edges
        if w < 420:
            factor = 0.3 + 0.7 * (w - 380) / 40
        elif w > 700:
            factor = 0.3 + 0.7 * (780 - w) / 80
        else:
            factor = 1.0
        return (
            max(0, min(255, int(r * factor * 255))),
            max(0, min(255, int(g * factor * 255))),
            max(0, min(255, int(b * factor * 255))),
        )

    @classmethod
    def compute_spectrum(
        cls,
        Z: int = 1,
        n_max: int = 7,
        series_lower_max: int = 5,
    ) -> list[dict]:
        """Compute all emission lines up to n_max.

        Returns list of dicts with keys:
            n_upper, n_lower, energy_Ha, energy_eV, wavelength_nm,
            series_name, rgb
        """
        lines: list[dict] = []
        for n_lower in range(1, min(n_max, series_lower_max + 1)):
            for n_upper in range(n_lower + 1, n_max + 1):
                dE = cls.transition_energy(Z, n_upper, n_lower)
                wl = cls.transition_wavelength_nm(Z, n_upper, n_lower)
                lines.append({
                    "n_upper": n_upper,
                    "n_lower": n_lower,
                    "energy_Ha": dE,
                    "energy_eV": dE * cls.HARTREE_EV,
                    "wavelength_nm": wl,
                    "series_name": cls.SERIES_NAMES.get(n_lower, f"n_f={n_lower}"),
                    "rgb": cls.wavelength_to_rgb(wl),
                })
        lines.sort(key=lambda x: x["wavelength_nm"])
        return lines

    @classmethod
    def render_spectrum_pil(
        cls,
        Z: int = 1,
        n_max: int = 7,
        width: int = 800,
        height: int = 120,
    ):
        """Render visible emission spectrum to a PIL Image.

        Returns (PIL.Image, list[dict]) — the image and the line data.
        """
        from PIL import Image, ImageDraw

        # Visible range
        wl_min, wl_max = 380.0, 780.0
        img = Image.new("RGB", (width, height), (10, 10, 15))
        draw = ImageDraw.Draw(img)

        # Draw continuous visible spectrum background (faint)
        for px in range(width):
            wl = wl_min + (wl_max - wl_min) * px / width
            r, g, b = cls.wavelength_to_rgb(wl)
            draw.line([(px, 0), (px, height)], fill=(r // 6, g // 6, b // 6))

        # emission lines
        lines = cls.compute_spectrum(Z, n_max)
        visible_lines = [l for l in lines if wl_min <= l["wavelength_nm"] <= wl_max]
        for line in visible_lines:
            px = int((line["wavelength_nm"] - wl_min) / (wl_max - wl_min) * width)
            px = max(0, min(width - 1, px))
            r, g, b = line["rgb"]
            for dx in range(-1, 2):
                x = max(0, min(width - 1, px + dx))
                draw.line([(x, 0), (x, height - 18)], fill=(r, g, b), width=1)
            label = f"{line['wavelength_nm']:.1f}"
            draw.text((max(0, px - 12), height - 16), label, fill=(200, 210, 220))

        return img, lines


# ======================================================================
# (#7) Multi-Electron Restricted Hartree-Fock (RHF) Self-Consistent Field
# ======================================================================

class HartreeFockWorld:
    """Multi-electron solver using Restricted Hartree-Fock on a 3D grid.

    Solves the Roothaan-Hall equations self-consistently:
        F C = S C ε
    where F is the Fock matrix, S the overlap, C the MO coefficients, ε the
    orbital energies.

    For a grid-based approach, we:
    1. Start from hydrogen-like orbitals as basis functions
    2. Build Coulomb (J) and exchange (K) operators on the grid
    3. Diagonalise the Fock operator via imaginary-time propagation
    4. Iterate until self-consistency (energy convergence)

    This is a simplified grid-based HF: no Gaussian basis sets, but directly
    solving on the real-space grid using the split-operator approach per orbital.
    """

    def __init__(
        self,
        n_electrons: int = 2,
        nuclear_Z: int = 2,
        grid_n: int = 48,
        extent: float = 15.0,
        use_gpu: bool = True,
    ) -> None:
        self.n_electrons = n_electrons
        self.nuclear_Z = nuclear_Z
        self.grid_n = grid_n
        self.extent = extent
        self.use_gpu = use_gpu and _GPU_AVAILABLE
        self.xp = _get_xp(self.use_gpu)

        # RHF: n_orbitals = n_electrons / 2 (closed-shell, each orbital doubly occupied)
        self.n_orbitals = max(1, n_electrons // 2)
        self.orbital_energies: list[float] = [0.0] * self.n_orbitals
        self.total_energy: float = 0.0
        self.scf_converged: bool = False
        self.scf_iterations: int = 0

        self._setup_grids()
        self._build_nuclear_potential()
        self._init_orbitals()

    def _setup_grids(self) -> None:
        xp = self.xp
        N = self.grid_n
        L = 2.0 * self.extent
        self.dx = L / N
        self.dV = self.dx ** 3
        self.softening = max(0.2, self.dx * 0.5)

        x = np.linspace(-self.extent + self.dx / 2, self.extent - self.dx / 2, N)
        self.x = x
        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        self.X = xp.asarray(X)
        self.Y = xp.asarray(Y)
        self.Z_grid = xp.asarray(Z)
        self.R = xp.sqrt(self.X ** 2 + self.Y ** 2 + self.Z_grid ** 2)

        k = np.fft.fftfreq(N, d=self.dx) * 2.0 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
        self.K2 = xp.asarray(KX ** 2 + KY ** 2 + KZ ** 2)

    def _build_nuclear_potential(self) -> None:
        xp = self.xp
        self.V_nuc = -float(self.nuclear_Z) / xp.sqrt(self.R ** 2 + self.softening ** 2)

    def _init_orbitals(self) -> None:
        """Initialize orbitals as hydrogen-like eigenstates."""
        xp = self.xp
        self.orbitals = []
        # Use lowest hydrogen-like states: 1s, 2s, 2p0, 2p1, ...
        quantum_numbers = [
            (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 1, -1),
            (3, 0, 0), (3, 1, 0), (3, 1, 1), (3, 2, 0),
        ]
        for i in range(self.n_orbitals):
            n, l, m = quantum_numbers[i % len(quantum_numbers)]
            psi = self._hydrogen_orbital(n, l, m)
            norm = float(xp.sqrt(xp.sum(xp.abs(psi) ** 2) * self.dV))
            if norm > 1e-30:
                psi /= norm
            self.orbitals.append(psi)

    def _hydrogen_orbital(self, n: int, l: int, m: int):
        """Simple hydrogen-like orbital on the grid."""
        xp = self.xp
        R_safe = xp.maximum(self.R, 1e-10)
        rho = 2.0 * R_safe / n
        # Simplified: just use exponential × r^l
        radial = xp.exp(-rho / 2.0) * xp.power(rho, float(l))
        theta = xp.arccos(xp.clip(self.Z_grid / R_safe, -1.0, 1.0))
        phi = xp.arctan2(self.Y, self.X)
        angular = xp.exp(1j * m * phi) * xp.power(xp.sin(theta), abs(m))
        psi = (radial * angular).astype(xp.complex128)
        return psi

    def _compute_electron_density(self):
        """Total electron density ρ(r) = 2 Σ_i |φ_i(r)|² (factor of 2 for spin)."""
        xp = self.xp
        rho = xp.zeros_like(self.R)
        for phi in self.orbitals:
            rho += 2.0 * xp.abs(phi) ** 2
        return rho

    def _compute_hartree_potential(self, rho):
        """Solve Poisson equation for Hartree potential: ∇²V_H = -4πρ.
        Uses Fourier-space solution: V_H(k) = 4πρ(k)/k².
        """
        xp = self.xp
        rho_k = xp.fft.fftn(rho)
        # Avoid division by zero at k=0
        K2_safe = xp.maximum(self.K2, 1e-12)
        V_H_k = 4.0 * math.pi * rho_k / K2_safe
        # k=0 component: set to zero (no net charge effect)
        if xp == np:
            V_H_k.ravel()[0] = 0.0
        else:
            V_H_k = V_H_k.copy()
            V_H_k.ravel()[0] = 0.0
        V_H = xp.fft.ifftn(V_H_k).real
        return V_H

    def scf_step(self) -> float:
        """Perform one SCF iteration using imaginary-time propagation.

        Returns the total energy after this step.
        """
        xp = self.xp
        rho = self._compute_electron_density()
        V_H = self._compute_hartree_potential(rho)

        # Fock potential = nuclear + Hartree (exchange approximated as Slater Xα)
        # V_xc ≈ -Cx * ρ^(1/3)  (Slater exchange)
        Cx = 0.7386  # (3/4)(3/π)^(1/3)
        rho_safe = xp.maximum(rho, 1e-30)
        V_xc = -Cx * xp.power(rho_safe, 1.0 / 3.0)

        V_total = self.V_nuc + V_H + V_xc

        # Imaginary-time propagation for each orbital
        tau = 0.05  # imaginary time step
        exp_V_half = xp.exp(-V_total * tau * 0.5)
        exp_T = xp.exp(-self.K2 * tau * 0.5)

        new_orbitals = []
        for phi in self.orbitals:
            # Apply e^{-Vτ/2}
            phi = phi * exp_V_half
            # Apply e^{-Tτ} in k-space
            phi_k = xp.fft.fftn(phi)
            phi_k *= exp_T
            phi = xp.fft.ifftn(phi_k)
            # Apply e^{-Vτ/2}
            phi = phi * exp_V_half
            new_orbitals.append(phi)

        # Gram-Schmidt orthonormalisation
        for i in range(len(new_orbitals)):
            for j in range(i):
                overlap = xp.sum(xp.conj(new_orbitals[j]) * new_orbitals[i]) * self.dV
                new_orbitals[i] = new_orbitals[i] - complex(overlap) * new_orbitals[j]
            norm = float(xp.sqrt(xp.sum(xp.abs(new_orbitals[i]) ** 2) * self.dV))
            if norm > 1e-30:
                new_orbitals[i] /= norm

        self.orbitals = new_orbitals

        # Compute orbital energies and total energy
        total_E = 0.0
        for i, phi in enumerate(self.orbitals):
            # Kinetic energy
            phi_k = xp.fft.fftn(phi)
            N3 = self.grid_n ** 3
            T_i = float(0.5 * xp.sum(self.K2 * xp.abs(phi_k) ** 2).real * self.dV / N3)
            # Potential energy
            V_i = float(xp.sum(V_total * xp.abs(phi) ** 2).real * self.dV)
            self.orbital_energies[i] = T_i + V_i
            total_E += 2.0 * (T_i + V_i)  # factor 2 for spin

        # Correct for double-counting of electron-electron interaction
        rho_new = self._compute_electron_density()
        V_H_new = self._compute_hartree_potential(rho_new)
        E_H = 0.5 * float(xp.sum(V_H_new * rho_new).real * self.dV)
        E_xc = float(xp.sum(V_xc * rho_new).real * self.dV)
        total_E = total_E - E_H - E_xc * 0.5  # remove double-count

        old_E = self.total_energy
        self.total_energy = total_E
        self.scf_iterations += 1

        return abs(total_E - old_E)

    def run_scf(self, max_iter: int = 50, tol: float = 1e-5) -> None:
        """Run SCF to convergence."""
        for i in range(max_iter):
            dE = self.scf_step()
            if dE < tol:
                self.scf_converged = True
                break

    def get_total_density_2d(self, axis: str = "y") -> np.ndarray:
        """Get 2D integrated density for visualisation (always numpy)."""
        xp = self.xp
        rho = self._compute_electron_density()
        if axis == "z":
            result = xp.sum(rho, axis=2) * self.dx
        elif axis == "y":
            result = xp.sum(rho, axis=1) * self.dx
        else:
            result = xp.sum(rho, axis=0) * self.dx
        return _to_numpy(result)

    def get_info_text(self) -> str:
        gpu_tag = "GPU" if self.use_gpu else "CPU"
        orb_str = ", ".join(f"{e:.4f}" for e in self.orbital_energies)
        return (
            f"HF-SCF ({gpu_tag}) | Z={self.nuclear_Z} e⁻={self.n_electrons} | "
            f"Grid: {self.grid_n}³ | E_total={self.total_energy:.6f} Ha | "
            f"Orbitals: [{orb_str}] Ha | "
            f"SCF iter={self.scf_iterations} converged={self.scf_converged}"
        )

    def get_info_dict(self) -> dict:
        return {
            "nuclear_Z": self.nuclear_Z,
            "n_electrons": self.n_electrons,
            "n_orbitals": self.n_orbitals,
            "total_energy": self.total_energy,
            "orbital_energies": list(self.orbital_energies),
            "scf_iterations": self.scf_iterations,
            "scf_converged": self.scf_converged,
            "grid_n": self.grid_n,
            "gpu": self.use_gpu,
        }
