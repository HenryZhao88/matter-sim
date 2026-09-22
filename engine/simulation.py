"""The simulation loop: electrons (SCF) + nuclei (relax / dynamics).

``Simulation`` is the single object the server drives. Each ``step()``
converges the electrons for the current nuclei, computes quantum forces, and
(depending on mode) moves the nuclei. ``snapshot()`` packages everything a
viewer needs.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from .core.backend import get_backend
from .core.grid import Grid
from .core.units import AU_TIME_FS, KELVIN_HARTREE
from .electrons.scf import SCFResult, SCFSolver
from .nuclei.integrators import FIRE, VelocityVerlet
from .system import System

MODES = ("frozen", "relax", "dynamics")
QUALITY_H = {"draft": 0.30, "standard": 0.20, "fine": 0.13}


def supported_elements() -> set[int]:
    """Elements the live grids resolve accurately.

    Point nuclei need h ≲ 0.6/Z for the innermost electrons, so without
    pseudopotentials only H and He are accurate at interactive resolutions.
    """
    return {1, 2}


@dataclass
class Params:
    functional: str = "lda"
    quality: str = "draft"
    mode: str = "relax"
    backend: str = "mlx"
    T_e: float = 1e-3            # electronic temperature, Hartree
    temperature_K: float = 0.0   # heat-bath temperature for dynamics
    dt: float = 10.0             # MD time step, a.u. (~0.24 fs)
    margin: float = 6.5          # vacuum around nuclei, bohr
    fmax_relaxed: float = 2e-3   # Ha/bohr — relax stops below this

    @property
    def h(self) -> float:
        return QUALITY_H[self.quality]


class Simulation:
    def __init__(self, system: System, params: Params | None = None) -> None:
        self.params = params or Params()
        self.system = system
        self.step_count = 0
        self.time_au = 0.0
        self.relaxed = False
        self.forces = np.zeros_like(system.positions)
        self.result: SCFResult | None = None
        self.last_step_seconds = 0.0
        self.energy_trace: list[float] = []
        self.on_scf_progress = None  # optional callback(row) per SCF iteration
        self._build()

    # ----------------------------------------------------------- building
    def box_size(self) -> float:
        P = self.system.positions
        extent = float(np.max(np.abs(P))) if len(P) else 0.0
        return max(12.0, 2 * (extent + self.params.margin))

    def _build(self) -> None:
        p = self.params
        self.grid = Grid(self.box_size(), p.h, get_backend(p.backend))
        self.solver = SCFSolver(self.grid, self.system, functional=p.functional, T_e=p.T_e)
        self.fire = FIRE()
        self.md = VelocityVerlet(self.system.masses, dt=p.dt,
                                 temperature=p.temperature_K * KELVIN_HARTREE,
                                 friction=1e-3 if p.temperature_K > 0 else 0.0)
        self.relaxed = False
        self.result = None
        self.energy_trace = []

    def set_system(self, system: System) -> None:
        self.system = system
        self.step_count = 0
        self.time_au = 0.0
        self.forces = np.zeros_like(system.positions)
        self._build()

    def set_params(self, **changes) -> None:
        rebuild_keys = {"functional", "quality", "backend", "T_e", "margin"}
        for k, v in changes.items():
            if not hasattr(self.params, k):
                raise ValueError(f"unknown parameter {k}")
            setattr(self.params, k, type(getattr(self.params, k))(v))
        if self.params.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if rebuild_keys & changes.keys():
            self._build()
        else:
            self.md.dt = self.params.dt
            self.md.temperature = self.params.temperature_K * KELVIN_HARTREE
            self.md.friction = 1e-3 if self.params.temperature_K > 0 else 0.0
            self.relaxed = False

    def move_nuclei(self, positions: np.ndarray) -> None:
        """Place nuclei (e.g. user drag). Rebuilds the box if they leave it."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.system.positions = positions
        if self.box_size() > self.grid.L + 1e-9 or self.box_size() < 0.75 * self.grid.L:
            self._build()
        else:
            self.solver.set_positions(positions)
        self.relaxed = False
        self.fire = FIRE()

    # ---------------------------------------------------------- stepping
    def _solve(self) -> None:
        self.result = self.solver.run(callback=self.on_scf_progress)
        self.forces = self.solver.forces()

    def step(self) -> None:
        t0 = time.perf_counter()
        mode = self.params.mode
        sysm = self.system
        if self.result is None:
            self._solve()
        elif mode == "relax" and not self.relaxed:
            new = self.fire.step(sysm.positions, self.forces)
            self.solver.set_positions(new)
            self._solve()
            if float(np.max(np.linalg.norm(self.forces, axis=1))) < self.params.fmax_relaxed:
                self.relaxed = True
        elif mode == "dynamics":
            v = self.md.half_kick(sysm.velocities, self.forces)
            self.solver.set_positions(self.md.drift(sysm.positions, v))
            self._solve()
            v = self.md.half_kick(v, self.forces)
            sysm.velocities = self.md.thermostat(v)
            self.time_au += self.md.dt
        # Keep the molecule inside its vacuum box.
        if self.box_size() > self.grid.L + 1e-9:
            self.move_nuclei(sysm.positions - sysm.positions.mean(axis=0))
            self._solve()
        self.step_count += 1
        self.energy_trace.append(self.total_energy)
        self.energy_trace = self.energy_trace[-400:]
        self.last_step_seconds = time.perf_counter() - t0

    @property
    def is_idle(self) -> bool:
        """Nothing left to compute until the user changes something."""
        m = self.params.mode
        return self.result is not None and (m == "frozen" or (m == "relax" and self.relaxed))

    @property
    def total_energy(self) -> float:
        if self.result is None:
            return float("nan")
        e = self.result.free_energy
        if self.params.mode == "dynamics":
            e += self.md.kinetic_energy(self.system.velocities)
        return e

    # ---------------------------------------------------------- snapshot
    def snapshot(self, max_side: int = 96) -> dict:
        r = self.result
        sysm = self.system
        rho = r.rho if r is not None else np.zeros(self.grid.shape)
        spin = (r.rho_up - r.rho_dn) if r is not None else np.zeros(self.grid.shape)
        stride = max(1, math.ceil(self.grid.N / max_side))
        meta = {
            "step": self.step_count,
            "time_fs": self.time_au * AU_TIME_FS,
            "mode": self.params.mode,
            "relaxed": self.relaxed,
            "params": asdict(self.params) | {"h": self.grid.h},
            "backend": self.grid.backend.name,
            "step_seconds": self.last_step_seconds,
            "grid": {"N": self.grid.N, "L": self.grid.L, "h": self.grid.h, "stride": stride},
            "system": {"charge": sysm.charge, "multiplicity": sysm.multiplicity,
                       "n_electrons": sysm.n_electrons},
            "atoms": [
                {"Z": int(Z), "pos": sysm.positions[i].tolist(), "force": self.forces[i].tolist(),
                 "vel": sysm.velocities[i].tolist()}
                for i, Z in enumerate(sysm.charges)
            ],
            "energy_trace": self.energy_trace[-200:],
        }
        if r is not None:
            meta |= {
                "energy": r.energy, "free_energy": r.free_energy, "total_energy": self.total_energy,
                "components": r.components, "converged": r.converged, "scf_iterations": r.iterations,
                "scf_history": [{"drho": h["drho"], "energy": h["energy"]} for h in r.history],
                "orbitals": {
                    "up": {"energy": r.evals[0].tolist(), "occ": r.occ[0].tolist()},
                    "down": {"energy": r.evals[1].tolist(), "occ": r.occ[1].tolist()},
                },
                "boundary_leak": r.boundary_leak,
            }
        return {"meta": meta, "rho": _downsample(rho, stride), "spin": _downsample(spin, stride)}


def _downsample(a: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return a
    n = (a.shape[0] // s) * s
    a = a[:n, :n, :n]
    return a.reshape(n // s, s, n // s, s, n // s, s).mean(axis=(1, 3, 5))
