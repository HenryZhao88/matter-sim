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
N_BASIS = 12
RHO_FLOOR = 0.02    # the embedding energy's √ρ slope is capped below this density
# Aluminium's values, the defaults (every model saved before these were per model is aluminium's).
# Another metal takes its own from short_range_for(): the element's nuclear charge, and the basis
# and the splice set by where its training data stops, by the same rule these follow.
R_MIN = 3.6         # bohr: first basis centre, the shortest distance aluminium's data sampled
Z_NUC = 13.0        # aluminium: the nuclear charge the cores repel with at short range
CORE_IN, CORE_OUT = 2.2, 3.4     # bohr: pure nuclear repulsion below, pure learned fit above
SPLICE_GAP, SPLICE_WIDTH = R_MIN - CORE_OUT, CORE_OUT - CORE_IN     # 0.2 and 1.2 bohr


def short_range_for(configs, z: float) -> dict:
    """The per-element short-range settings, by aluminium's rule: the basis starts at the shortest
    distance the training data sampled, the learned part ends SPLICE_GAP below it, and the nuclear
    repulsion of charge z takes over across SPLICE_WIDTH below that. Nothing is fitted here: the
    data only says where it stops being evidence."""
    r_min = min(float(np.linalg.norm(pairs_with_images(c["cell"], c["positions"], rc=6.0)[2], axis=1).min())
                for c in configs)
    core_out = r_min - SPLICE_GAP
    return {"z": float(z), "r_min": r_min, "core_in": core_out - SPLICE_WIDTH, "core_out": core_out}


def cutoff(r):
    x = np.clip(r / R_CUT, 0, 1)
    return (1 - x * x) ** 3


def basis(r, r_min=R_MIN):
    """(len(r), N_BASIS) smooth radial functions that vanish at the cutoff."""
    centres = np.linspace(r_min, R_CUT - 0.8, N_BASIS)
    width = (centres[1] - centres[0]) * 1.1
    return np.exp(-((r[:, None] - centres[None, :]) / width) ** 2) * cutoff(r)[:, None]


def basis_deriv(r, eps=1e-5, r_min=R_MIN):
    return (basis(r + eps, r_min) - basis(r - eps, r_min)) / (2 * eps)


def _zbl(r, z=Z_NUC):
    """Ziegler–Biersack–Littmark screened nuclear repulsion (Hartree, bohr).

    Two nuclei of charge Z pushed close together repel as Z²/r, screened by the electrons that
    remain between them. The screening function is universal — no parameters to choose — and it
    is what keeps atoms from passing through each other at high temperature. The training
    configurations never sample distances this short, so nothing here is fitted: it is the part
    of the physics the data could not teach."""
    a = 0.8853 / (2 * z ** 0.23)
    x = r / a
    c, d = (0.18175, 0.50986, 0.28022, 0.02817), (3.19980, 0.94229, 0.40290, 0.20162)
    phi = sum(ci * np.exp(-di * x) for ci, di in zip(c, d))
    dphi = sum(-ci * di / a * np.exp(-di * x) for ci, di in zip(c, d))
    return z ** 2 / r * phi, z ** 2 * (dphi / r - phi / r ** 2)


def _blend(r, core_in=CORE_IN, core_out=CORE_OUT):
    """0 below core_in (all nuclear repulsion), 1 above core_out (all learned), smooth between."""
    t = np.clip((np.asarray(r, float) - core_in) / (core_out - core_in), 0.0, 1.0)
    w = t * t * (3 - 2 * t)
    dw = np.where((t > 0) & (t < 1), 6 * t * (1 - t) / (core_out - core_in), 0.0)
    return w, dw


def dens_with_core(r, b, core_out=CORE_OUT, r_min=R_MIN):
    """The background density an atom sits in, held flat below the shortest distance the training
    data sampled.

    Left to the fitted basis alone the density would fall back to zero as two atoms approach,
    and the embedding energy's √ρ would then pull them together without limit. Holding it at its
    value at CORE_OUT removes that, and leaves the short-range repulsion where it belongs: in
    the screened nuclear term, which is what actually keeps nuclei apart. Letting the density
    instead grow inward (as a real electron density does) is defensible physics but ruins the
    dynamics here — through a cubic embedding function it produced forces a thousand times
    stiffer than the nuclear repulsion itself."""
    r = np.atleast_1d(np.asarray(r, float))
    f = basis(r, r_min) @ b
    df = basis_deriv(r, r_min=r_min) @ b
    f0 = float((basis(np.array([core_out]), r_min) @ b)[0])
    inner = r < core_out
    f = np.where(inner, f0, f)
    df = np.where(inner, 0.0, df)
    return np.maximum(f, 0.0), np.where(f > 0, df, 0.0)


def core_pair(r, z=Z_NUC, core_in=CORE_IN, core_out=CORE_OUT):
    """The short-range part of the pair term, and its derivative: nuclear repulsion faded out
    before the first distance the training data ever sampled."""
    r = np.atleast_1d(np.asarray(r, float))
    e, de = _zbl(np.maximum(r, 1e-6), z)
    w, dw = _blend(r, core_in, core_out)
    return (1 - w) * e, (1 - w) * de - dw * e


@dataclass
class EAM:
    a: np.ndarray          # pair coefficients
    b: np.ndarray          # density coefficients
    c: np.ndarray          # embedding: F(ρ) = c0 √ρ + c1 ρ + c2 ρ² + c3 ρ³
    e0: float              # energy per atom offset (sets the zero; physics unaffected)
    # short range (see short_range_for); the defaults are aluminium's, so older pickles load as they were
    z: float = Z_NUC
    r_min: float = R_MIN
    core_in: float = CORE_IN
    core_out: float = CORE_OUT

    # ------------------------------------------------------------ functions
    def short_range(self) -> dict:
        return {"z": self.z, "r_min": self.r_min, "core_in": self.core_in, "core_out": self.core_out}

    def B(self, r):
        return basis(r, self.r_min)

    def dB(self, r):
        return basis_deriv(r, r_min=self.r_min)

    def core(self, r):
        return core_pair(r, self.z, self.core_in, self.core_out)

    def dens_core(self, r):
        return dens_with_core(r, self.b, self.core_out, self.r_min)

    def phi(self, r):
        r = np.atleast_1d(r)
        return self.B(r) @ self.a + self.core(r)[0]

    def dens(self, r):
        return self.dens_core(r)[0]

    def F(self, rho):
        rho = np.maximum(rho, RHO_FLOOR)
        c = self.c
        return c[0] * np.sqrt(rho) + c[1] * rho + c[2] * rho ** 2 + c[3] * rho ** 3

    def dF(self, rho):
        # √ρ has an infinite slope at the origin: an atom whose neighbours happen to sum to
        # almost nothing would otherwise feel a force of 10⁵ eV/bohr from a perfectly ordinary
        # arrangement. Below RHO_FLOOR the embedding term is flat.
        rho = np.maximum(rho, RHO_FLOOR)
        c = self.c
        return 0.5 * c[0] / np.sqrt(rho) + c[1] + 2 * c[2] * rho + 3 * c[3] * rho ** 2

    def tabulate(self, n: int = 4000):
        """Dense tables for fast molecular dynamics."""
        r = np.linspace(0.5, R_CUT, n)
        f, df = self.dens_core(r)
        return {"r": r, "phi": self.phi(r), "dphi": self.dB(r) @ self.a + self.core(r)[1],
                "f": f, "df": df}

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
    B = model.B(r)
    dB = model.dB(r)
    f_pair, df_pair = model.dens_core(r)
    rho = np.bincount(I, weights=f_pair, minlength=n)
    core, dcore = model.core(r)
    E = float(np.sum(model.F(rho)) + 0.5 * np.sum(B @ model.a + core)) + model.e0 * n
    dE_dr = 0.5 * (dB @ model.a + dcore) + (model.dF(rho)[I] + model.dF(rho)[J]) * df_pair / 2
    # each unordered pair appears twice (i→j and j→i); dE/dr_ij per ordered entry:
    F = np.zeros((n, 3))
    unit = D / r[:, None]
    np.add.at(F, I, (dE_dr * 1.0)[:, None] * unit)
    np.add.at(F, J, -(dE_dr * 1.0)[:, None] * unit)
    return E, F


# ------------------------------------------------------------------ fitting (MLX autodiff)
def fit(configs, iters: int = 4000, w_force: float = 30.0, seed: int = 0, log=None,
        e_scale_eV: float | None = 0.3, short_range: dict | None = None) -> EAM:
    """configs: list of dicts {cell, positions, energy, forces} (Hartree, bohr).

    All configurations are packed into one graph (atoms numbered globally, energies summed
    per configuration), so each training step is a handful of GPU kernels. Without MLX this
    returns a plain starting guess: :func:`refine` (float64 least squares, NumPy and SciPy only)
    does the real work from there on any machine.

    ``e_scale_eV`` weights each configuration by 1 / (1 + (ΔE / e_scale)²), ΔE its energy per
    atom above the lowest one: the potential should be most accurate where the metal actually
    spends its time (solid and liquid near melting lie within a few tenths of an eV).

    ``short_range``: the element's settings from :func:`short_range_for` (default: aluminium's).
    The fit sees only the learned part, so every training distance must lie above core_out."""
    from ..core.accel import have_mlx
    sr = short_range or {}
    r_min = sr.get("r_min", R_MIN)
    rng0 = np.random.default_rng(seed)
    if not have_mlx():
        return EAM(rng0.normal(0, 1e-3, N_BASIS), np.abs(rng0.normal(0.05, 0.01, N_BASIS)),
                   np.array([-0.2, 0.0, 0.0, 0.0]),
                   float(np.mean([c["energy"] / len(c["positions"]) for c in configs])), **sr)

    import mlx.core as mx
    import mlx.optimizers as optim

    mx.random.seed(seed)          # so a refit (and anything derived from it) is reproducible
    I_all, J_all, B_all, dB_all, U_all, cfg_of_atom, E_ref, F_ref, n_at = [], [], [], [], [], [], [], [], []
    off = 0
    for k, c in enumerate(configs):
        I, J, D = pairs_with_images(c["cell"], c["positions"])
        r = np.linalg.norm(D, axis=1)
        n = len(c["positions"])
        I_all.append(I + off); J_all.append(J + off)
        B_all.append(basis(r, r_min)); dB_all.append(basis_deriv(r, r_min=r_min)); U_all.append(D / r[:, None])
        cfg_of_atom.append(np.full(n, k)); n_at.append(n)
        E_ref.append(float(c["energy"]) / n); F_ref.append(np.asarray(c["forces"]))
        off += n
    f32 = lambda x: mx.array(np.concatenate(x).astype(np.float32))
    I, J = mx.array(np.concatenate(I_all)), mx.array(np.concatenate(J_all))
    B, dB, U = f32(B_all), f32(dB_all), f32(U_all)
    cfg = mx.array(np.concatenate(cfg_of_atom))
    n_at = mx.array(np.array(n_at, np.float32))
    e_mean = float(np.mean(E_ref))
    E_t = mx.array((np.array(E_ref) - e_mean).astype(np.float32))
    F_t = f32(F_ref)
    n_atoms, n_cfg = off, len(configs)
    dE = (np.array(E_ref) - min(E_ref)) * 27.211386
    w_np = np.ones(n_cfg) if e_scale_eV is None else 1 / (1 + (dE / e_scale_eV) ** 2)
    w_cfg = mx.array((w_np / w_np.mean()).astype(np.float32))
    w_atom = w_cfg[cfg][:, None]

    rng = np.random.default_rng(seed)
    params = {"a": mx.array(rng.normal(0, 1e-3, N_BASIS).astype(np.float32)),
              "b": mx.array(np.abs(rng.normal(0.05, 0.01, N_BASIS)).astype(np.float32)),
              "c": mx.array(np.array([-0.2, 0.0, 0.0, 0.0], np.float32)),
              "e0": mx.array(np.array([0.0], np.float32))}

    def predict(p):
        rho = mx.maximum(mx.zeros((n_atoms,)).at[I].add(B @ mx.abs(p["b"])), 1e-8)
        c = p["c"]
        Femb = c[0] * mx.sqrt(rho) + c[1] * rho + c[2] * rho ** 2 + c[3] * rho ** 3
        dF = 0.5 * c[0] / mx.sqrt(rho) + c[1] + 2 * c[2] * rho + 3 * c[3] * rho ** 2
        pair_atom = mx.zeros((n_atoms,)).at[I].add(0.5 * (B @ p["a"]))
        E = mx.zeros((n_cfg,)).at[cfg].add(Femb + pair_atom) / n_at + p["e0"][0]
        dEdr = 0.5 * (dB @ p["a"]) + 0.5 * (dF[I] + dF[J]) * (dB @ mx.abs(p["b"]))
        vec = dEdr[:, None] * U
        F = mx.zeros((n_atoms, 3)).at[I].add(vec).at[J].add(-vec)
        return E, F

    def loss_fn(p):
        E, F = predict(p)
        return mx.mean(w_cfg * (E - E_t) ** 2) * 1e4 + w_force * mx.mean(w_atom * (F - F_t) ** 2) * 1e2

    opt = optim.Adam(learning_rate=3e-3)
    step = mx.value_and_grad(loss_fn)
    for it in range(iters):
        loss, g = step(params)
        opt.update(params, g)
        mx.eval(params, opt.state)
        if log and it % 500 == 0:
            log(it, float(loss))
        if it == iters // 2:
            opt.learning_rate = 1e-3
        if it == (3 * iters) // 4:
            opt.learning_rate = 3e-4
    to_np = lambda x: np.array(x, dtype=np.float64)
    return EAM(to_np(params["a"]), np.abs(to_np(params["b"])), to_np(params["c"]),
               float(to_np(params["e0"])[0]) + e_mean, **sr)


def refine(model: EAM, configs, w_force: float = 30.0, e_scale_eV: float | None = 0.3, log=None) -> EAM:
    """Polish a fit in float64 with Levenberg–Marquardt least squares (29 parameters: small
    enough for exact second-order convergence, which a stochastic optimiser does not reach).
    The model's short-range settings are kept; the residuals are the learned part alone, which is
    the whole model only while every training distance lies above core_out (checked)."""
    from scipy.optimize import least_squares

    pre = []
    E_ref = []
    for c in configs:
        I, J, D = pairs_with_images(c["cell"], c["positions"])
        r = np.linalg.norm(D, axis=1)
        if r.min() < model.core_out:
            raise ValueError(f"a training distance {r.min():.3f} bohr lies inside the splice (core_out "
                             f"{model.core_out:.3f}): take the settings from short_range_for(configs, z)")
        pre.append((len(c["positions"]), I, J, model.B(r), model.dB(r), D / r[:, None], np.asarray(c["forces"])))
        E_ref.append(float(c["energy"]) / len(c["positions"]))
    E_ref = np.array(E_ref)
    dE = (E_ref - E_ref.min()) * 27.211386
    w = np.ones(len(configs)) if e_scale_eV is None else 1 / (1 + (dE / e_scale_eV) ** 2)
    w = w / w.mean()
    na, nb = N_BASIS, N_BASIS

    def unpack(x):
        return x[:na], x[na:na + nb], x[na + nb:na + nb + 4], x[-1]

    def residuals(x):
        a, b, cc, e0 = unpack(x)
        b = np.maximum(b, 0.0)
        out = []
        for k, (n, I, J, B, dB, U, Fr) in enumerate(pre):
            rho = np.maximum(np.bincount(I, B @ b, n), RHO_FLOOR)
            Femb = cc[0] * np.sqrt(rho) + cc[1] * rho + cc[2] * rho ** 2 + cc[3] * rho ** 3
            dF = 0.5 * cc[0] / np.sqrt(rho) + cc[1] + 2 * cc[2] * rho + 3 * cc[3] * rho ** 2
            E = (Femb.sum() + 0.5 * (B @ a).sum()) / n + e0
            dEdr = 0.5 * (dB @ a) + 0.5 * (dF[I] + dF[J]) * (dB @ b)
            vec = dEdr[:, None] * U
            F = np.zeros((n, 3))
            np.add.at(F, I, vec)
            np.add.at(F, J, -vec)
            out.append([np.sqrt(w[k]) * (E - E_ref[k]) * 100.0])
            out.append(np.sqrt(w[k] * w_force / (3 * n)) * (F - Fr).ravel() * 10.0)
        return np.concatenate([np.ravel(o) for o in out])

    x0 = np.concatenate([model.a, np.abs(model.b), model.c, [model.e0]])
    lo = np.concatenate([np.full(na, -np.inf), np.zeros(nb), np.full(5, -np.inf)])
    hi = np.full(len(x0), np.inf)
    # the density coefficients stay non-negative: a negative electron density is not a thing,
    # and where the fitted density crossed zero the embedding term's slope blew up
    res = least_squares(residuals, np.clip(x0, lo + 1e-12, None), bounds=(lo, hi),
                        method="trf", max_nfev=4000, x_scale="jac")
    if log:
        log(f"least squares: cost {res.cost:.4g} after {res.nfev} evaluations ({res.message})")
    a, b, cc, e0 = unpack(res.x)
    return EAM(a, b, cc, float(e0), **model.short_range())
