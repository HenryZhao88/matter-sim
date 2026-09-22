"""Moving the nuclei under quantum-mechanical forces.

- :class:`FIRE`   — geometry relaxation to the nearest energy minimum
  (Bitzek et al., PRL 97, 170201, 2006). Not physical time; just descent.
- :class:`VelocityVerlet` — Born–Oppenheimer molecular dynamics with real
  nuclear masses, optionally coupled to a Langevin heat bath at temperature T.
"""

from __future__ import annotations

import numpy as np


class FIRE:
    def __init__(self, dt: float = 0.25, dt_max: float = 1.0, max_step: float = 0.15) -> None:
        self.dt = dt
        self.dt_max = dt_max
        self.max_step = max_step
        self.alpha = 0.1
        self.n_positive = 0
        self.v: np.ndarray | None = None

    def step(self, positions: np.ndarray, forces: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(positions)
        P = float(np.sum(forces * self.v))
        fnorm = np.linalg.norm(forces)
        vnorm = np.linalg.norm(self.v)
        if P > 0:
            if fnorm > 0:
                self.v = (1 - self.alpha) * self.v + self.alpha * vnorm * forces / fnorm
            self.n_positive += 1
            if self.n_positive > 5:
                self.dt = min(self.dt * 1.1, self.dt_max)
                self.alpha *= 0.99
        else:
            self.v[:] = 0.0
            self.dt *= 0.5
            self.alpha = 0.1
            self.n_positive = 0
        self.v = self.v + self.dt * forces
        dx = self.dt * self.v
        longest = np.max(np.linalg.norm(dx, axis=1)) if len(dx) else 0.0
        if longest > self.max_step:
            dx *= self.max_step / longest
        return positions + dx


class VelocityVerlet:
    """Velocity Verlet with optional Langevin thermostat (friction γ, temperature T)."""

    def __init__(self, masses: np.ndarray, dt: float = 10.0, temperature: float = 0.0,
                 friction: float = 0.0, seed: int = 0) -> None:
        self.m = np.asarray(masses, dtype=float)[:, None]
        self.dt = dt
        self.temperature = temperature   # Hartree (k_B T)
        self.friction = friction         # 1 / a.u. time
        self._rng = np.random.default_rng(seed)

    def half_kick(self, velocities: np.ndarray, forces: np.ndarray) -> np.ndarray:
        return velocities + 0.5 * self.dt * forces / self.m

    def drift(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        return positions + self.dt * velocities

    def thermostat(self, velocities: np.ndarray) -> np.ndarray:
        if self.friction <= 0:
            return velocities
        c1 = np.exp(-self.friction * self.dt)
        sigma = np.sqrt((1 - c1 * c1) * self.temperature / self.m)
        return c1 * velocities + sigma * self._rng.standard_normal(velocities.shape)

    def kinetic_energy(self, velocities: np.ndarray) -> float:
        return float(0.5 * np.sum(self.m * velocities ** 2))
