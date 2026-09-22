"""Fermi–Dirac occupations at electronic temperature T_e.

Each spin channel holds a fixed number of electrons (set by the total spin),
with its own chemical potential. Smearing lets degenerate open shells fill
fractionally and evenly — nothing chooses a configuration by hand.
"""

from __future__ import annotations

import numpy as np


def fermi(evals: np.ndarray, n_electrons: float, T_e: float) -> tuple[np.ndarray, float, float]:
    """Return (occupations, chemical potential, entropy S) for one spin channel."""
    evals = np.asarray(evals, dtype=float)
    if n_electrons <= 0 or evals.size == 0:
        return np.zeros_like(evals), float(evals.min()) if evals.size else 0.0, 0.0
    if n_electrons > evals.size:
        raise ValueError("more electrons than bands")
    T = max(T_e, 1e-8)

    def occ(mu):
        return 0.5 * (1.0 - np.tanh((evals - mu) / (2 * T)))  # overflow-safe Fermi function

    lo, hi = evals.min() - 50 * T - 1.0, evals.max() + 50 * T + 1.0
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if occ(mu).sum() > n_electrons:
            hi = mu
        else:
            lo = mu
    f = occ(mu)
    f *= n_electrons / f.sum()
    fc = np.clip(f, 1e-300, 1 - 1e-16)
    S = float(-np.sum(fc * np.log(fc) + (1 - fc) * np.log1p(-fc)))
    return f, float(mu), S
