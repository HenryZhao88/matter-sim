"""Self-consistent field solver for the electrons.

Given fixed nuclei, finds the electronic ground state of

    H = Σ_i [ -½∇²_i + V_nuc(r_i) ] + Σ_{i<j} 1/|r_i - r_j|

in one of three approximations to the electron–electron term:

- ``"lda"``  Kohn–Sham DFT with local spin-density exchange–correlation.
- ``"hf"``   unrestricted Hartree–Fock (exact exchange, zero fitted constants).
- ``"none"`` no electron–electron term. Exact for a single electron, where
             that term is identically zero; wrong for anything else.

Orbitals are stored as grid arrays normalised to Σ|ψ|² = 1 (so the physical
wavefunction is ψ / √dV).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.grid import Grid
from ..system import System
from .eigensolver import lobpcg, teter_preconditioner
from .external import NuclearField, ion_ion
from .mixing import PulayMixer
from .occupations import fermi
from .xc import lda_xc

FUNCTIONALS = ("lda", "hf", "none")


@dataclass
class SCFResult:
    energy: float                     # total energy E (Hartree)
    free_energy: float                # E - T_e S (what forces are derivatives of)
    components: dict[str, float]
    evals: tuple[np.ndarray, np.ndarray]
    occ: tuple[np.ndarray, np.ndarray]
    rho_up: np.ndarray
    rho_dn: np.ndarray
    converged: bool
    iterations: int
    history: list[dict] = field(default_factory=list)
    boundary_leak: float = 0.0        # fraction of electrons near box faces
    seconds: float = 0.0

    @property
    def rho(self) -> np.ndarray:
        return self.rho_up + self.rho_dn


class SCFSolver:
    def __init__(self, grid: Grid, system: System, functional: str = "lda", T_e: float = 1e-3,
                 n_extra: int = 4, seed: int = 0) -> None:
        if functional not in FUNCTIONALS:
            raise ValueError(f"functional must be one of {FUNCTIONALS}")
        self.grid = grid
        self.b = grid.backend
        self.system = system
        self.functional = functional
        self.T_e = T_e
        self.n_extra = n_extra
        self._rng = np.random.default_rng(seed)
        self._precond = teter_preconditioner(grid)
        self.mixer = PulayMixer()
        self.nuclear = NuclearField(grid)
        self.set_positions(system.positions)
        self.orbitals = [self._initial_orbitals(n) for n in (system.n_up, system.n_dn)]
        self.rho_up, self.rho_dn = self._initial_density()
        self.last: SCFResult | None = None

    # --------------------------------------------------------------- setup
    def set_positions(self, positions) -> None:
        self.system.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.v_nuc = self.nuclear.potential(self.system.charges, self.system.positions)
        self.mixer.reset()

    def _envelope(self) -> np.ndarray:
        env = np.zeros(self.grid.shape)
        for R in self.system.positions:
            env += np.exp(-0.5 * self.grid.r_from(R))
        return env

    def _initial_orbitals(self, n_electrons: int):
        nb = int(math.ceil(n_electrons)) + self.n_extra if n_electrons > 0 else 0
        if nb == 0:
            return None
        X = self._rng.standard_normal((nb,) + self.grid.shape) * self._envelope()
        return self.b.asarray(X)

    def _initial_density(self):
        """Superposition of simple exponentials: a numerical starting point only."""
        rho = np.zeros(self.grid.shape)
        for Z, R in zip(self.system.charges, self.system.positions):
            rho += Z * np.exp(-2.0 * self.grid.r_from(R)) / np.pi
        ne = self.system.n_electrons
        total = rho.sum() * self.grid.dV
        rho *= ne / total if total > 0 else 0.0
        if ne == 0:
            return rho, rho.copy()
        return rho * self.system.n_up / ne, rho * self.system.n_dn / ne

    # ---------------------------------------------------------- operators
    def _exchange_op(self, occ_orbitals, occ):
        """Fock exchange K acting on a block, built from fixed occupied orbitals."""
        g, b = self.grid, self.b
        pairs = [(phi, f) for phi, f in zip(occ_orbitals, occ) if f > 1e-8]

        def K(Y):
            out = None
            for phi, f in pairs:
                term = phi * g.hartree(phi * Y) * (-f / g.dV)
                out = term if out is None else out + term
                b.eval(out)
            return out if out is not None else 0.0 * Y

        return K

    def _hamiltonian(self, v_local, K=None):
        g = self.grid
        V = self.b.asarray(v_local)

        def H(X):
            out = g.kinetic(X) + V * X
            if K is not None:
                out = out + K(X)
            return out

        return H

    def _density(self, X, f):
        if X is None:
            return np.zeros(self.grid.shape)
        b = self.b
        occ = b.asarray(f).reshape((-1, 1, 1, 1))
        return b.to_numpy(b.xp.sum(occ * X * X, axis=0)) / self.grid.dV

    def _band_expect(self, X, op):
        b = self.b
        OX = op(X)
        return b.to_numpy(b.xp.sum((X * OX).reshape(X.shape[0], -1), axis=1))

    # --------------------------------------------------------------- SCF
    def run(self, max_iter: int = 80, tol_rho: float = 2e-4, tol_e: float = 2e-6,
            callback=None) -> SCFResult:
        g, sysm = self.grid, self.system
        t0 = time.perf_counter()
        E_nn, _ = ion_ion(sysm.charges, sysm.positions)
        n_elec = (sysm.n_up, sysm.n_dn)
        history: list[dict] = []
        converged = False
        F_prev = None
        # Previous orbitals/occupations for the exchange operator (HF).
        prev_occ = [None, None]

        first = self.last is None
        for it in range(1, max_iter + 1):
            rho_in = (self.rho_up, self.rho_dn)
            rho_tot = rho_in[0] + rho_in[1]
            if self.functional == "none":
                v_h = np.zeros(g.shape)
                v_xc = (np.zeros(g.shape), np.zeros(g.shape))
            else:
                v_h = self.b.to_numpy(g.hartree(self.b.asarray(rho_tot)))
                if self.functional == "lda":
                    _, vu, vd = lda_xc(*rho_in)
                    v_xc = (vu, vd)
                else:
                    v_xc = (np.zeros(g.shape), np.zeros(g.shape))

            evals, occs, S_tot = [], [], 0.0
            for s in range(2):
                X = self.orbitals[s]
                if X is None:
                    evals.append(np.zeros(0)); occs.append(np.zeros(0))
                    continue
                K = None
                if self.functional == "hf" and prev_occ[s] is not None:
                    K = self._exchange_op(self.orbitals[s], prev_occ[s])
                H = self._hamiltonian(self.v_nuc + v_h + v_xc[s], K)
                iters = 60 if (first and it == 1) else 6
                lam, X, _ = lobpcg(self.b, H, X, self._precond, tol=2e-4 if self.b.is_gpu else 1e-5,
                                   maxiter=iters, n_check=int(math.ceil(n_elec[s])))
                self.orbitals[s] = X
                f, _, S = fermi(lam, n_elec[s], self.T_e)
                evals.append(lam); occs.append(f)
                S_tot += S

            if self.functional == "hf":
                prev_occ = [occs[0] if len(occs[0]) else None, occs[1] if len(occs[1]) else None]

            rho_out = tuple(self._density(self.orbitals[s], occs[s]) for s in range(2))
            comps = self._energy(rho_out, occs, E_nn)
            E = sum(comps.values())
            F = E - self.T_e * S_tot
            drho = float(np.sum(np.abs(rho_out[0] - rho_in[0]) + np.abs(rho_out[1] - rho_in[1])) * g.dV)
            dF = abs(F - F_prev) if F_prev is not None else np.inf
            F_prev = F
            history.append({"iter": it, "energy": E, "free_energy": F, "drho": drho, "dE": dF})
            if callback is not None:
                callback(history[-1])

            if drho < tol_rho and dF < tol_e:
                self.rho_up, self.rho_dn = rho_out
                converged = True
                break

            if self.functional == "none":
                self.rho_up, self.rho_dn = rho_out
                continue
            x_in = np.concatenate([rho_in[0].ravel(), rho_in[1].ravel()])
            x_out = np.concatenate([rho_out[0].ravel(), rho_out[1].ravel()])
            x = np.maximum(self.mixer.mix(x_in, x_out), 0.0)
            M = g.N ** 3
            ru, rd = x[:M].reshape(g.shape), x[M:].reshape(g.shape)
            for arr, n in ((ru, n_elec[0]), (rd, n_elec[1])):
                tot = arr.sum() * g.dV
                if tot > 0:
                    arr *= n / tot
            self.rho_up, self.rho_dn = ru, rd

        rho = self.rho_up + self.rho_dn
        result = SCFResult(
            energy=E, free_energy=F, components=comps,
            evals=(evals[0], evals[1]), occ=(occs[0], occs[1]),
            rho_up=self.rho_up, rho_dn=self.rho_dn,
            converged=converged, iterations=it, history=history,
            boundary_leak=self._boundary_leak(rho),
            seconds=time.perf_counter() - t0,
        )
        self.last = result
        return result

    def _energy(self, rho_out, occs, E_nn) -> dict[str, float]:
        g = self.grid
        rho = rho_out[0] + rho_out[1]
        kinetic = 0.0
        exchange_hf = 0.0
        for s in range(2):
            X = self.orbitals[s]
            if X is None:
                continue
            kinetic += float(np.dot(occs[s], self._band_expect(X, g.kinetic)))
            if self.functional == "hf":
                K = self._exchange_op(X, occs[s])
                exchange_hf += 0.5 * float(np.dot(occs[s], self._band_expect(X, K)))
        comps = {
            "kinetic": kinetic,
            "electron_nuclear": float(np.sum(self.v_nuc * rho) * g.dV),
            "hartree": 0.0,
            "exchange_correlation": 0.0,
            "nuclear_nuclear": E_nn,
        }
        if self.functional != "none":
            v_h = self.b.to_numpy(g.hartree(self.b.asarray(rho)))
            comps["hartree"] = 0.5 * float(np.sum(v_h * rho) * g.dV)
        if self.functional == "lda":
            e_xc, _, _ = lda_xc(*rho_out)
            comps["exchange_correlation"] = float(np.sum(e_xc) * g.dV)
        elif self.functional == "hf":
            comps["exchange_correlation"] = exchange_hf
        return comps

    def _boundary_leak(self, rho: np.ndarray) -> float:
        """Fraction of electrons within L/8 of any box face (should be ~0)."""
        n = self.grid.N
        m = max(1, n // 8)
        inner = rho[m:-m, m:-m, m:-m].sum()
        total = rho.sum()
        return float(1 - inner / total) if total > 0 else 0.0

    # -------------------------------------------------------------- forces
    def forces(self) -> np.ndarray:
        """Hellmann–Feynman forces on the nuclei for the last converged state."""
        if self.last is None:
            raise RuntimeError("run() first")
        sysm = self.system
        F_e = self.nuclear.forces(sysm.charges, sysm.positions, self.last.rho)
        _, F_nn = ion_ion(sysm.charges, sysm.positions)
        return F_e + F_nn

    def eigenstates(self, n: int, spin: int = 0, v_extra=None):
        """Lowest n eigenstates of the current effective one-electron Hamiltonian."""
        X = self._rng.standard_normal((n,) + self.grid.shape) * self._envelope()
        H = self._hamiltonian(self.v_nuc if v_extra is None else self.v_nuc + v_extra)
        return lobpcg(self.b, H, self.b.asarray(X), self._precond,
                      tol=2e-4 if self.b.is_gpu else 1e-6, maxiter=200)
