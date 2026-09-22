"""Pulay (DIIS) density mixing for the self-consistent field loop."""

from __future__ import annotations

import numpy as np


class PulayMixer:
    def __init__(self, beta: float = 0.35, history: int = 7) -> None:
        self.beta = beta
        self.history = history
        self._x: list[np.ndarray] = []
        self._f: list[np.ndarray] = []

    def reset(self) -> None:
        self._x.clear()
        self._f.clear()

    def mix(self, x_in: np.ndarray, x_out: np.ndarray) -> np.ndarray:
        f = x_out - x_in
        self._x.append(x_in.copy())
        self._f.append(f)
        if len(self._x) > self.history:
            self._x.pop(0)
            self._f.pop(0)
        m = len(self._f)
        F = np.array(self._f)
        A = F @ F.T
        B = np.zeros((m + 1, m + 1))
        B[:m, :m] = A / max(np.abs(A).max(), 1e-300)
        B[m, :m] = B[:m, m] = 1.0
        rhs = np.zeros(m + 1)
        rhs[m] = 1.0
        try:
            c = np.linalg.solve(B, rhs)[:m]
        except np.linalg.LinAlgError:
            c = np.zeros(m)
            c[-1] = 1.0
        X = np.array(self._x)
        return c @ X + self.beta * (c @ F)
