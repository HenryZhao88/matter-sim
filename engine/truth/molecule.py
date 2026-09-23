"""The system truth mode solves: bare nuclei and a count of electrons of each spin."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Molecule:
    charges: list[float]
    positions: np.ndarray        # (n_nuclei, 3) bohr
    n_up: int
    n_dn: int

    @property
    def n_elec(self) -> int:
        return self.n_up + self.n_dn

    def nuclear_repulsion(self) -> float:
        E = 0.0
        for i in range(len(self.charges)):
            for j in range(i + 1, len(self.charges)):
                E += self.charges[i] * self.charges[j] / float(np.linalg.norm(self.positions[i] - self.positions[j]))
        return E
