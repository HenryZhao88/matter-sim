"""A physical system: nuclei plus a count of electrons. Nothing else.

This is deliberately all a scene may specify. There is no notion of bonds,
orbitals, molecules or shapes here; those emerge from the solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core.elements import ELEMENTS


@dataclass
class System:
    charges: list[int]                      # nuclear charges Z
    positions: np.ndarray                   # (n, 3) bohr
    charge: int = 0                         # net charge (electrons removed)
    multiplicity: int | None = None         # 2S+1; None = lowest allowed
    velocities: np.ndarray | None = field(default=None, repr=False)  # bohr / a.u. time

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 3)
        if len(self.charges) != len(self.positions):
            raise ValueError("one position per nucleus")
        for Z in self.charges:
            if Z not in ELEMENTS:
                raise ValueError(f"element Z={Z} not supported yet")
        if self.multiplicity is None:
            self.multiplicity = 1 if self.n_electrons % 2 == 0 else 2
        if self.n_electrons < 0:
            raise ValueError("more electrons removed than exist")
        if (self.n_electrons + self.multiplicity - 1) % 2 != 0 or self.multiplicity - 1 > self.n_electrons:
            raise ValueError(f"multiplicity {self.multiplicity} impossible for {self.n_electrons} electrons")
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)

    @property
    def n_electrons(self) -> int:
        return int(sum(self.charges)) - self.charge

    @property
    def n_up(self) -> int:
        return (self.n_electrons + self.multiplicity - 1) // 2

    @property
    def n_dn(self) -> int:
        return self.n_electrons - self.n_up

    @property
    def masses(self) -> np.ndarray:
        from .core.units import AMU_ME
        return np.array([ELEMENTS[Z].mass_amu * AMU_ME for Z in self.charges])

    def copy(self) -> "System":
        return System(list(self.charges), self.positions.copy(), self.charge, self.multiplicity,
                      self.velocities.copy())
