"""A real-time universe in one space dimension: the Schwinger model (QED in 1+1D).

Lattice Hamiltonian (Kogut–Susskind staggered fermions, Jordan–Wigner, the gauge field
eliminated with Gauss's law, open boundaries):

    H = w Σ_n (σ⁺_n σ⁻_{n+1} + h.c.) + (m/2) Σ_n (−1)ⁿ σᶻ_n + J Σ_n L_n²,
    L_n = ε₀ + Σ_{k≤n} q_k + Σ_{k≤n} Q_k,     q_k = (σᶻ_k + (−1)ᵏ)/2,

with w = 1/(2a), J = g²a/2. Even sites hold particles, odd sites antiparticles; L_n is the electric field on link n in units of the charge; ε₀ is
a background field and Q_k are optional static external charges.

Evolution is exact (Krylov exponentiation) in the sector of zero total charge. Nothing
decides when particles appear: pair creation, screening, string breaking and particle
production in collisions all follow from H.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply


@dataclass
class SchwingerModel:
    N: int = 18                       # staggered sites (N/2 physical sites)
    mass: float = 0.25                # m / g
    g: float = 1.0
    a: float = 1.0                    # lattice spacing (1/g units)
    background: float = 0.0           # ε₀ (external field, in units of the charge)
    static_charges: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.N % 2:
            raise ValueError("N must be even")
        self.w = 1 / (2 * self.a)
        self.J = self.g * self.g * self.a / 2
        self._basis()
        self._hamiltonian()

    # ------------------------------------------------------------ basis (zero total charge)
    def _basis(self) -> None:
        N = self.N
        states = []
        for ups in combinations(range(N), N // 2):
            s = 0
            for u in ups:
                s |= 1 << u
            states.append(s)
        self.states = np.array(states, dtype=np.int64)
        self.index = {int(s): i for i, s in enumerate(self.states)}
        bits = (self.states[:, None] >> np.arange(N)[None, :]) & 1        # 1 = spin up
        self.sz = (2 * bits - 1).astype(np.int8)                           # (D, N)
        stag = np.array([(-1) ** n for n in range(N)])
        self.q = (self.sz + stag[None, :]) / 2                             # site charge
        Q = np.zeros(N)
        for k, v in self.static_charges.items():
            Q[k] += v
        self.L = self.background + np.cumsum(self.q + Q[None, :], axis=1)[:, :-1]   # (D, N−1)

    def _hamiltonian(self) -> None:
        N, D = self.N, len(self.states)
        stag = np.array([(-1) ** n for n in range(N)])
        diag = 0.5 * self.mass * (self.sz * stag[None, :]).sum(axis=1) + self.J * (self.L ** 2).sum(axis=1)
        rows, cols = [np.arange(D)], [np.arange(D)]
        vals = [diag]
        for n in range(N - 1):
            b = (self.states >> n) & 1
            c = (self.states >> (n + 1)) & 1
            flip = np.nonzero(b != c)[0]
            new = self.states[flip] ^ ((1 << n) | (1 << (n + 1)))
            rows.append(flip)
            cols.append(np.array([self.index[int(s)] for s in new]))
            vals.append(np.full(len(flip), self.w))
        self.H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(D, D))

    @property
    def dim(self) -> int:
        return len(self.states)

    # ------------------------------------------------------------ states
    def bare_vacuum(self) -> np.ndarray:
        """No particles, no antiparticles: even sites down, odd sites up."""
        s = 0
        for n in range(1, self.N, 2):
            s |= 1 << n
        psi = np.zeros(self.dim, complex)
        psi[self.index[s]] = 1
        return psi

    def ground_state(self) -> tuple[float, np.ndarray]:
        e, v = eigsh(self.H, k=1, which="SA")
        return float(e[0]), v[:, 0].astype(complex)

    def _pair_operator(self, n: int) -> sp.csr_matrix:
        """Gauge-invariant creation of a particle–antiparticle pair on link (n, n+1)."""
        D = self.dim
        b = (self.states >> n) & 1
        c = (self.states >> (n + 1)) & 1
        if n % 2 == 0:   # particle on n (↓→↑), antiparticle on n+1 (↑→↓)
            ok = np.nonzero((b == 0) & (c == 1))[0]
        else:            # antiparticle on n, particle on n+1
            ok = np.nonzero((b == 1) & (c == 0))[0]
        new = self.states[ok] ^ ((1 << n) | (1 << (n + 1)))
        cols = ok
        rows = np.array([self.index[int(s)] for s in new])
        return sp.csr_matrix((np.ones(len(ok)), (rows, cols)), shape=(D, D))

    def meson_packet(self, psi: np.ndarray, centre: float, k: float, width: float) -> np.ndarray:
        """Apply a moving wave packet of meson (bound pair) creation to ``psi``."""
        out = np.zeros_like(psi)
        for n in range(self.N - 1):
            amp = math.exp(-((n - centre) ** 2) / (2 * width ** 2)) * np.exp(1j * k * n)
            if abs(amp) > 1e-4:
                out += amp * (self._pair_operator(n) @ psi)
        return out / np.linalg.norm(out)

    def _flip(self, psi: np.ndarray, sites: tuple[int, ...], want: tuple[int, ...]) -> np.ndarray:
        """Flip the given sites where they currently hold the bit values ``want`` (else zero)."""
        ok = np.ones(self.dim, bool)
        mask = 0
        for n, w in zip(sites, want):
            ok &= ((self.states >> n) & 1) == w
            mask |= 1 << n
        idx = np.nonzero(ok & (np.abs(psi) > 0))[0]
        out = np.zeros_like(psi)
        new = self.states[idx] ^ mask
        out[[self.index[int(x)] for x in new]] = psi[idx]
        return out

    def jet_pair(self, psi: np.ndarray, k: float, spread: float = 1.5) -> np.ndarray:
        """A particle and an antiparticle created at the centre, flying apart with momenta ∓k,
        joined by the electric field line that Gauss's law requires between them."""
        c = (self.N // 2 - 1) & ~1                      # an even site near the centre
        out = np.zeros_like(psi)
        for u in range(0, self.N):
            i, j = c - 2 * u, c + 1 + 2 * u             # particle (even site), antiparticle (odd site)
            if i < 0 or j >= self.N:
                break
            amp = math.exp(-(u * u) / (2 * spread * spread)) * np.exp(-1j * k * (j - i))
            out += amp * self._flip(psi, (i, j), (0, 1))
        return out / np.linalg.norm(out)

    # ------------------------------------------------------------ observables
    def measure(self, psi: np.ndarray) -> dict:
        p = np.abs(psi) ** 2
        charge = p @ self.q
        field_ = p @ self.L
        stag = np.array([(-1) ** n for n in range(self.N)])
        # particle number per site: 1 where a particle (even) or antiparticle (odd) sits
        occ = (1 + stag[None, :] * self.sz) / 2
        number = p @ occ
        energy = float(np.real(np.vdot(psi, self.H @ psi)))
        return {"charge": charge, "field": field_, "number": number,
                "particles": float(number.sum()), "energy": energy}

    def evolve(self, psi: np.ndarray, t_max: float, frames: int):
        """Yield (t, observables) at evenly spaced times, exactly."""
        dt = t_max / (frames - 1)
        A = -1j * self.H
        yield 0.0, self.measure(psi)
        for f in range(1, frames):
            psi = expm_multiply(A * dt, psi)
            yield f * dt, self.measure(psi)


# ------------------------------------------------------------------ scenarios
SCENARIOS = {
    "pair_creation": {
        "name": "Pair creation from a strong field",
        "blurb": "Empty space with a strong electric field switched on. The field tears particle–antiparticle pairs out of the vacuum, and they move to cancel it.",
    },
    "string_breaking": {
        "name": "String breaking",
        "blurb": "Two fixed charges joined by a string of electric field. Past a certain length the string snaps and new particles appear to end it.",
    },
    "jet": {
        "name": "Hadronisation",
        "blurb": "A particle and antiparticle fly apart from one point. The field string between them stores energy as it stretches, then snaps into new particle–antiparticle pairs: the outgoing charges end up dressed as mesons.",
    },
    "collision": {
        "name": "Particle collision",
        "blurb": "Two mesons (bound particle–antiparticle pairs) fired at each other. Watch for new particles in the wreckage.",
    },
}


def run_scenario(kind: str, N: int = 18, mass: float = 0.25, t_max: float = 12.0, frames: int = 121,
                 strength: float = 1.0, a: float = 0.5):
    """Returns (model, iterator of (t, observables)).

    ``strength``: background field (pair creation), static charge (string breaking), or
    meson momentum as a fraction of the largest lattice momentum (collision).
    """
    if kind == "pair_creation":
        model = SchwingerModel(N=N, mass=mass, a=a, background=strength)
        psi = model.bare_vacuum()
    elif kind == "string_breaking":
        i, j = N // 4, N - 1 - N // 4
        model = SchwingerModel(N=N, mass=mass, a=a, static_charges={i: +strength, j: -strength})
        # start from the vacuum without the string, then add the charges (a sudden quench)
        _, psi = SchwingerModel(N=N, mass=mass, a=a).ground_state()
    elif kind == "jet":
        model = SchwingerModel(N=N, mass=mass, a=a)
        _, vac = model.ground_state()
        psi = model.jet_pair(vac, k=strength * 2.0)
    elif kind == "collision":
        model = SchwingerModel(N=N, mass=mass, a=a)
        _, vac = model.ground_state()
        k = strength * 2.0
        # On this lattice a packet with phase e^{−ikn} moves right (the hopping sign sets it).
        psi = model.meson_packet(vac, centre=N * 0.22, k=-k, width=1.3)
        psi = model.meson_packet(psi, centre=N * 0.78 - 1, k=+k, width=1.3)
    else:
        raise ValueError(kind)
    return model, model.evolve(psi, t_max, frames)


def vacuum_profile(N: int, mass: float, a: float = 0.5) -> dict:
    """Observables of the interacting vacuum, to show what a scenario adds to it."""
    m = SchwingerModel(N=N, mass=mass, a=a)
    _, vac = m.ground_state()
    return m.measure(vac)
