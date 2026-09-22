"""A learned embedded-atom potential for metals, fitted only to this engine's own DFT.

Form (the embedded-atom method, the standard physical picture of a metal):

    E = Σ_i F(ρ_i) + ½ Σ_{i≠j} φ(r_ij),        ρ_i = Σ_{j≠i} f(r_ij)

Each atom sits in the electron density ρ_i its neighbours provide and pays an embedding
energy F for it; φ is the remaining pairwise interaction. The functions φ and f are
expansions in smooth radial basis functions and F is a small polynomial in √ρ — all of
their coefficients are learned from DFT energies and forces (engine/crystal), never from
experiment or from published potentials.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

R_CUT = 10.0        # bohr (≈ 5.3 Å): first three neighbour shells of fcc aluminium
R_MIN = 3.6
N_BASIS = 12


def cutoff(r):
    x = np.clip(r / R_CUT, 0, 1)
    return (1 - x * x) ** 3


def basis(r):
    """(len(r), N_BASIS) smooth radial functions that vanish at the cutoff."""
    centres = np.linspace(R_MIN, R_CUT - 0.8, N_BASIS)
    width = (centres[1] - centres[0]) * 1.1
    return np.exp(-((r[:, None] - centres[None, :]) / width) ** 2) * cutoff(r)[:, None]


def basis_deriv(r, eps=1e-5):
    return (basis(r + eps) - basis(r - eps)) / (2 * eps)


@dataclass
class EAM:
    a: np.ndarray          # pair coefficients
    b: np.ndarray          # density coefficients
    c: np.ndarray          # embedding: F(ρ) = c0 √ρ + c1 ρ + c2 ρ² + c3 ρ³
    e0: float              # energy per atom offset (sets the zero; physics unaffected)

    # ------------------------------------------------------------ functions
    def phi(self, r):
        return basis(np.atleast_1d(r)) @ self.a

    def dens(self, r):
        return basis(np.atleast_1d(r)) @ self.b

    def F(self, rho):
        rho = np.maximum(rho, 1e-12)
        c = self.c
        return c[0] * np.sqrt(rho) + c[1] * rho + c[2] * rho ** 2 + c[3] * rho ** 3

    def dF(self, rho):
        rho = np.maximum(rho, 1e-12)
        c = self.c
        return 0.5 * c[0] / np.sqrt(rho) + c[1] + 2 * c[2] * rho + 3 * c[3] * rho ** 2

    def tabulate(self, n: int = 4000):
        """Dense tables for fast molecular dynamics."""
        r = np.linspace(0.5, R_CUT, n)
        return {"r": r, "phi": self.phi(r), "dphi": basis_deriv(r) @ self.a,
                "f": self.dens(r), "df": basis_deriv(r) @ self.b}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "EAM":
        return pickle.loads(path.read_bytes())


# ------------------------------------------------------------------ neighbours in small periodic cells
def pairs_with_images(cell, positions, rc=R_CUT):
    """All (i, j, d_vec) with |d| < rc, including periodic images (works for cells smaller than rc)."""
    cell = np.asarray(cell, float)
    n = len(positions)
    reps = np.ceil(rc / cell).astype(int)
    I, J, D = [], [], []
    for shift in product(*[range(-m, m + 1) for m in reps]):
        T = np.array(shift) * cell
        d = positions[None, :, :] + T - positions[:, None, :]          # r_j + T − r_i
        r = np.linalg.norm(d, axis=-1)
        mask = (r < rc) & (r > 1e-8)
        ii, jj = np.nonzero(mask)
        I.append(ii); J.append(jj); D.append(d[ii, jj])
    return np.concatenate(I), np.concatenate(J), np.concatenate(D)


def energy_forces(model: EAM, cell, positions):
    I, J, D = pairs_with_images(cell, positions)
    r = np.linalg.norm(D, axis=1)
    n = len(positions)
    B = basis(r)
    dB = basis_deriv(r)
    rho = np.bincount(I, weights=B @ model.b, minlength=n)
    E = float(np.sum(model.F(rho)) + 0.5 * np.sum(B @ model.a)) + model.e0 * n
    dE_dr = 0.5 * (dB @ model.a) * 2 / 2 + (model.dF(rho)[I] + model.dF(rho)[J]) * (dB @ model.b) / 2
    # each unordered pair appears twice (i→j and j→i); dE/dr_ij per ordered entry:
    F = np.zeros((n, 3))
    unit = D / r[:, None]
    np.add.at(F, I, (dE_dr * 1.0)[:, None] * unit)
    np.add.at(F, J, -(dE_dr * 1.0)[:, None] * unit)
    return E, F


# ------------------------------------------------------------------ fitting (MLX autodiff)
def fit(configs, iters: int = 4000, w_force: float = 30.0, seed: int = 0, log=None) -> EAM:
    """configs: list of dicts {cell, positions, energy, forces} (Hartree, bohr)."""
    import mlx.core as mx
    import mlx.optimizers as optim

    data = []
    for c in configs:
        I, J, D = pairs_with_images(c["cell"], c["positions"])
        r = np.linalg.norm(D, axis=1)
        data.append({
            "n": len(c["positions"]), "I": mx.array(I), "J": mx.array(J),
            "B": mx.array(basis(r).astype(np.float32)), "dB": mx.array(basis_deriv(r).astype(np.float32)),
            "unit": mx.array((D / r[:, None]).astype(np.float32)),
            "E": float(c["energy"]) / len(c["positions"]),
            "F": mx.array(np.asarray(c["forces"], np.float32)),
        })
    e_mean = float(np.mean([d["E"] for d in data]))
    rng = np.random.default_rng(seed)
    params = {"a": mx.array(rng.normal(0, 1e-3, N_BASIS).astype(np.float32)),
              "b": mx.array(np.abs(rng.normal(0.05, 0.01, N_BASIS)).astype(np.float32)),
              "c": mx.array(np.array([-0.2, 0.0, 0.0, 0.0], np.float32)),
              "e0": mx.array(np.array([0.0], np.float32))}

    def predict(p, d):
        rho_pair = d["B"] @ p["b"]
        rho = mx.zeros((d["n"],)).at[d["I"]].add(rho_pair)
        rho = mx.maximum(rho, 1e-8)
        c = p["c"]
        Femb = c[0] * mx.sqrt(rho) + c[1] * rho + c[2] * rho ** 2 + c[3] * rho ** 3
        dF = 0.5 * c[0] / mx.sqrt(rho) + c[1] + 2 * c[2] * rho + 3 * c[3] * rho ** 2
        E = (mx.sum(Femb) + 0.5 * mx.sum(d["B"] @ p["a"])) / d["n"] + p["e0"][0]
        dEdr = 0.5 * (d["dB"] @ p["a"]) + 0.5 * (dF[d["I"]] + dF[d["J"]]) * (d["dB"] @ p["b"])
        vec = dEdr[:, None] * d["unit"]
        F = mx.zeros((d["n"], 3)).at[d["I"]].add(vec).at[d["J"]].add(-vec)
        return E, F

    def loss_fn(p):
        tot = 0.0
        for d in data:
            E, F = predict(p, d)
            tot = tot + (E - (d["E"] - e_mean)) ** 2 * 1e4 + w_force * mx.mean((F - d["F"]) ** 2) * 1e2
        return tot / len(data)

    opt = optim.Adam(learning_rate=3e-3)
    step = mx.value_and_grad(loss_fn)
    for it in range(iters):
        loss, g = step(params)
        opt.update(params, g)
        mx.eval(params, opt.state)
        if log and it % 250 == 0:
            log(it, float(loss))
        if it == iters // 2:
            opt.learning_rate = 1e-3
    to_np = lambda x: np.array(x, dtype=np.float64)
    return EAM(to_np(params["a"]), to_np(params["b"]), to_np(params["c"]), float(to_np(params["e0"])[0]) + e_mean)
