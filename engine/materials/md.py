"""Molecular dynamics of many atoms with the learned potential.

Classical nuclei moving on the energy surface learned from DFT (eam.py). Periodic
orthorhombic box, neighbour lists from a periodic k-d tree (rebuilt with a safety skin),
velocity Verlet, a Bussi thermostat (samples the canonical ensemble exactly) and a
Berendsen barostat for constant pressure. Hartree atomic units throughout.

Experiments built on it:
  thermal expansion   lattice spacing against temperature at zero pressure
  heat capacity       enthalpy against temperature
  melting point       solid and liquid in one box: below T_m the crystal grows, above it melts
  latent heat         enthalpy jump between liquid and solid at T_m
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from ..core.units import AMU_ME, AU_TIME_FS, KELVIN_HARTREE
from .eam import EAM, R_CUT

HA_PER_BOHR3_GPA = 29421.02648438959
SKIN = 1.0
MAX_BOX_STEP = 2e-4      # largest fractional change of the box per step under the barostat
RUNAWAY = 2.0            # volume ratio at which a run is declared lost rather than reported


@dataclass
class State:
    pos: np.ndarray
    vel: np.ndarray
    box: np.ndarray
    mass: float


def fcc_lattice(a: float, n: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    cells = np.array([[i, j, k] for i in range(n[0]) for j in range(n[1]) for k in range(n[2])])
    pos = (cells[:, None, :] + basis[None]).reshape(-1, 3) * a
    return pos, a * np.asarray(n, float)


class EAMForceField:
    def __init__(self, model: EAM) -> None:
        self.m = model
        t = model.tabulate(6000)
        self.r, self.phi, self.dphi, self.f, self.df = t["r"], t["phi"], t["dphi"], t["f"], t["df"]
        self._tree_pos = None
        self._pairs = None

    def _neighbours(self, pos, box):
        if self._tree_pos is not None and self._tree_box is not None and np.allclose(box, self._tree_box):
            disp = (pos - self._tree_pos + box / 2) % box - box / 2
            if np.max(np.linalg.norm(disp, axis=1)) < SKIN / 2:
                return self._pairs
        tree = cKDTree(pos % box, boxsize=box)
        pairs = tree.query_pairs(R_CUT + SKIN, output_type="ndarray")
        self._pairs, self._tree_pos, self._tree_box = pairs, pos.copy(), box.copy()
        return pairs

    def compute(self, pos, box):
        """Energy, forces and virial Σ r·f (for the pressure)."""
        pairs = self._neighbours(pos, box)
        i, j = pairs[:, 0], pairs[:, 1]
        d = pos[j] - pos[i]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        keep = r < R_CUT
        i, j, d, r = i[keep], j[keep], d[keep], r[keep]
        n = len(pos)
        fr = np.interp(r, self.r, self.f)
        rho = np.bincount(i, fr, n) + np.bincount(j, fr, n)
        dF = self.m.dF(rho)
        E = float(np.sum(self.m.F(rho)) + np.sum(np.interp(r, self.r, self.phi))) + self.m.e0 * n
        dEdr = np.interp(r, self.r, self.dphi) + (dF[i] + dF[j]) * np.interp(r, self.r, self.df)
        fvec = (dEdr / r)[:, None] * d               # force on i is +fvec, on j is −fvec
        F = np.zeros((n, 3))
        np.add.at(F, i, fvec)
        np.add.at(F, j, -fvec)
        virial = float(-np.sum(dEdr * r))           # Σ r_ij · f_ij
        return E, F, virial


def kinetic(state: State) -> float:
    return 0.5 * state.mass * float(np.sum(state.vel ** 2))


def temperature(state: State) -> float:
    return 2 * kinetic(state) / (3 * len(state.pos)) / KELVIN_HARTREE


def pressure(state: State, virial: float) -> float:
    V = float(np.prod(state.box))
    return (2 * kinetic(state) + virial) / (3 * V)


class MD:
    def __init__(self, model: EAM, state: State, dt_fs: float = 2.0, seed: int = 0) -> None:
        self.ff = EAMForceField(model)
        self.s = state
        self.dt = dt_fs / AU_TIME_FS
        self.rng = np.random.default_rng(seed)
        self.E, self.F, self.W = self.ff.compute(state.pos, state.box)

    def thermalise(self, T: float) -> None:
        kT = T * KELVIN_HARTREE
        self.s.vel = self.rng.normal(0, math.sqrt(kT / self.s.mass), self.s.pos.shape)
        self.s.vel -= self.s.vel.mean(axis=0)

    def step(self, T: float | None = None, P_GPa: float | None = None, tau_T_fs: float = 100.0,
             tau_P_fs: float = 1000.0, frozen: np.ndarray | None = None) -> None:
        s, dt = self.s, self.dt
        s.vel += 0.5 * dt * self.F / s.mass
        if frozen is not None:
            s.vel[frozen] = 0.0
        s.pos += dt * s.vel
        if P_GPa is not None:
            # Berendsen: scale the box toward the target pressure (compressibility of a metal ~1/76 GPa).
            # The per-step limit is deliberately tiny: at 0.5% a run that never reaches equilibrium
            # inflates the cell without bound, and the metal boils off into vacuum.
            P = pressure(s, self.W) * HA_PER_BOHR3_GPA
            mu = 1 + dt / (tau_P_fs / AU_TIME_FS) * (P - P_GPa) / 76.0 / 3
            mu = float(np.clip(mu, 1 - MAX_BOX_STEP, 1 + MAX_BOX_STEP))
            s.pos *= mu
            s.box *= mu
        s.pos %= s.box
        self.E, self.F, self.W = self.ff.compute(s.pos, s.box)
        s.vel += 0.5 * dt * self.F / s.mass
        if frozen is not None:
            s.vel[frozen] = 0.0
        if T is not None:
            self._bussi(T, tau_T_fs, frozen)

    def _bussi(self, T, tau_fs, frozen):
        """Bussi–Donadio–Parrinello stochastic velocity rescaling."""
        s = self.s
        n_free = len(s.pos) if frozen is None else int((~frozen).sum())
        dof = 3 * n_free - 3
        K = kinetic(s)
        if K <= 0:
            return
        Kt = 0.5 * dof * T * KELVIN_HARTREE
        c = math.exp(-self.dt / (tau_fs / AU_TIME_FS))
        R = self.rng.normal()
        S = float(np.sum(self.rng.normal(size=dof - 1) ** 2))
        Knew = K * c + Kt / dof * (1 - c) * (S + R * R) + 2 * R * math.sqrt(c * (1 - c) * K * Kt / dof)
        s.vel *= math.sqrt(max(Knew, 0) / K)

    def run(self, steps: int, T=None, P_GPa=None, sample_every: int = 10, frozen=None, callback=None):
        rows = []
        V0 = float(np.prod(self.s.box))
        for k in range(steps):
            self.step(T, P_GPa, frozen=frozen)
            V = float(np.prod(self.s.box))
            if V > RUNAWAY * V0 or V < V0 / RUNAWAY:
                raise RuntimeError(f"the cell ran away under the barostat (volume ×{V / V0:.2g} after "
                                   f"{k} steps): the state is not a condensed metal, so nothing measured "
                                   f"from it would mean anything")
            if k % sample_every == 0:
                row = {"step": k, "E": self.E + kinetic(self.s), "U": self.E, "T": temperature(self.s),
                       "P": pressure(self.s, self.W) * HA_PER_BOHR3_GPA, "V": float(np.prod(self.s.box))}
                rows.append(row)
                if callback:
                    callback(row, self.s)
        return rows


# ------------------------------------------------------------------ experiments
def make_md(model: EAM, state: State, seed: int = 0, engine: str = "numpy", dt_fs: float = 2.0):
    """The integrator for ``engine``: "numpy" (md.MD, the reference) or "torch" (md_torch.MDTorch,
    the same dynamics on a GPU, reproducing md.MD's seeded trajectories; for 10⁵–10⁶ atoms)."""
    if engine == "torch":
        from .md_torch import MDTorch
        return MDTorch(model, state, dt_fs=dt_fs, seed=seed)
    if engine != "numpy":
        raise ValueError(f"unknown MD engine {engine!r}")
    return MD(model, state, dt_fs=dt_fs, seed=seed)


def al_state(model: EAM, a: float, n=(6, 6, 6)) -> State:
    pos, box = fcc_lattice(a, n)
    return State(pos, np.zeros_like(pos), box, 26.9815385 * AMU_ME)


def npt_lattice_constant(model: EAM, a0: float, T: float, n=(6, 6, 6), steps=3000, seed=0,
                         engine: str = "numpy") -> dict:
    """Mean lattice constant and enthalpy per atom at temperature T and zero pressure."""
    md = make_md(model, al_state(model, a0, n), seed=seed, engine=engine)
    md.thermalise(T)
    rows = md.run(steps, T=T, P_GPa=0.0, sample_every=10)
    tail = rows[len(rows) // 2:]
    V = np.mean([r["V"] for r in tail])
    N = len(md.s.pos)
    H = np.mean([r["E"] for r in tail]) / N               # P ≈ 0, so H ≈ E
    return {"T": T, "a": (V / (N / 4)) ** (1 / 3), "H_per_atom": H,
            "T_measured": float(np.mean([r["T"] for r in tail]))}


def coexistence(model: EAM, a_T: float, T: float, n=(5, 5, 12), steps=6000, seed=0,
                engine: str = "numpy") -> dict:
    """Half crystal, half liquid at temperature T, at fixed volume.

    The solid and the liquid share a box whose density is the solid's at T, opened by the few
    per cent that melting costs. Which phase grows is then decided by the energy: below the
    melting point the crystal advances into the liquid and the potential energy falls; above it
    the liquid eats the crystal and the energy rises. Fixed volume on purpose — a barostat on a
    two-phase cell chases a pressure that neither phase alone defines."""
    state = al_state(model, a_T, n)
    state.box = state.box * (1 + 0.02)                 # liquid is a few per cent less dense
    state.pos = state.pos * (1 + 0.02)
    md = make_md(model, state, seed=seed, engine=engine)
    N = len(state.pos)
    top = state.pos[:, 2] > state.box[2] / 2
    # melt the top half while holding the bottom half fixed, then release at T
    md.thermalise(1800.0)
    md.run(1500, T=1800.0, frozen=~top, sample_every=50)
    md.thermalise(T)
    rows = md.run(steps, T=T, sample_every=20)
    t = np.array([r["step"] for r in rows]) * md.dt * AU_TIME_FS
    U = np.array([r["U"] for r in rows]) / N
    k = len(t) // 5
    slope = np.polyfit(t[k:], U[k:], 1)[0]                   # Hartree per atom per fs
    return {"T": T, "slope": float(slope), "U_start": float(U[k]), "U_end": float(U[-1]),
            "positions": md.s.pos.copy(), "box": md.s.box.copy()}
