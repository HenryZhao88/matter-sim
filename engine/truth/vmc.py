"""Truth mode: a neural-network wavefunction trained on the Schrödinger equation alone.

Variational Monte Carlo with a FermiNet-style ansatz (Pfau et al., 2020). The many-electron
wavefunction is

    ψ(r₁…r_N) = Σ_k det[φ_k↑(r_i; all electrons)] · det[φ_k↓(…)] · e^{J}

where the orbitals φ are outputs of a small neural network that sees every electron (so
the determinant is no longer a mean-field product), and J is a Jastrow factor that
enforces the exact electron–electron cusp conditions (½ for opposite spins, ¼ for equal
spins), which follow from the Hamiltonian itself. Antisymmetry comes from the determinants.

Training minimises ⟨H⟩ over walkers sampled from |ψ|² (Metropolis). The local energy
E_L = −½ ∇²ψ/ψ + V uses exact derivatives by automatic differentiation. No exchange–
correlation functional, no basis set, no pseudopotentials: every electron, the exact
Hamiltonian. Accuracy grows with network size and training time — compute, not new rules.

Runs on the Metal GPU through MLX.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
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


class FermiNetLite(nn.Module):
    def __init__(self, mol: Molecule, hidden: int = 32, layers: int = 3, dets: int = 4) -> None:
        super().__init__()
        self.mol = mol
        self.R = mx.array(np.asarray(mol.positions, np.float32))
        self.Z = mx.array(np.asarray(mol.charges, np.float32))
        nI = len(mol.charges)
        self.dets = dets
        self.inp = nn.Linear(4 * nI, hidden)
        self.pair_in = nn.Linear(4, 16)
        self.layers = [nn.Linear(3 * hidden + 16, hidden) for _ in range(layers)]
        n_orb = max(mol.n_up, mol.n_dn, 1)
        self.orb_up = nn.Linear(hidden, dets * n_orb)
        self.orb_dn = nn.Linear(hidden, dets * n_orb)
        self.env_sigma = mx.ones((dets * n_orb, nI))
        self.env_pi = mx.ones((dets * n_orb, nI))
        self.n_orb = n_orb

    def log_psi(self, r):
        """log|ψ| for one configuration r of shape (N, 3)."""
        mol = self.mol
        N, nu = mol.n_elec, mol.n_up
        ei = r[:, None, :] - self.R[None, :, :]                       # (N, nI, 3)
        ed = mx.sqrt(mx.sum(ei * ei, axis=-1) + 1e-12)               # (N, nI)
        h = mx.tanh(self.inp(mx.concatenate([ei, ed[..., None]], axis=-1).reshape(N, -1)))
        rij = r[:, None, :] - r[None, :, :]
        dij = mx.sqrt(mx.sum(rij * rij, axis=-1) + 1e-12)
        g = mx.tanh(self.pair_in(mx.concatenate([rij, dij[..., None]], axis=-1)))   # (N, N, 16)
        gmean = mx.mean(g, axis=1)
        for layer in self.layers:
            up = mx.broadcast_to(mx.mean(h[:nu], axis=0, keepdims=True), h.shape) if nu > 0 else mx.zeros_like(h)
            dn = mx.broadcast_to(mx.mean(h[nu:], axis=0, keepdims=True), h.shape) if N - nu > 0 else mx.zeros_like(h)
            h = h + mx.tanh(layer(mx.concatenate([h, up, dn, gmean], axis=-1)))
        env = mx.exp(-mx.abs(self.env_sigma)[None, :, :] * ed[:, None, :])        # (N, K·n_orb, nI)
        env = mx.sum(self.env_pi[None] * env, axis=-1)                              # (N, K·n_orb)
        total = None
        for spin, lo, hi, head in (("up", 0, nu, self.orb_up), ("dn", nu, N, self.orb_dn)):
            n = hi - lo
            if n == 0:
                continue
            phi = head(h[lo:hi]) * env[lo:hi]                                        # (n, K·n_orb)
            phi = phi.reshape(n, self.dets, self.n_orb)[:, :, :n].transpose(1, 0, 2)  # (K, n, n)
            d = _det(phi)
            total = d if total is None else total * d
        psi = mx.sum(total)
        # Jastrow with exact electron–electron cusps: ½ for opposite spins, ¼ for equal spins
        spin = np.array([0] * nu + [1] * (N - nu))
        same = mx.array((spin[:, None] == spin[None, :]).astype(np.float32))
        cusp = 0.25 * same + 0.5 * (1 - same)
        mask = mx.array(np.triu(np.ones((N, N), np.float32), 1))
        J = mx.sum(mask * cusp * dij / (1 + dij))
        return mx.log(mx.abs(psi) + 1e-30) + J


def _det(A):
    """Determinants of a batch of small square matrices (K, n, n) by explicit expansion."""
    n = A.shape[-1]
    if n == 1:
        return A[:, 0, 0]
    if n == 2:
        return A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
    if n == 3:
        return (A[:, 0, 0] * (A[:, 1, 1] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 1])
                - A[:, 0, 1] * (A[:, 1, 0] * A[:, 2, 2] - A[:, 1, 2] * A[:, 2, 0])
                + A[:, 0, 2] * (A[:, 1, 0] * A[:, 2, 1] - A[:, 1, 1] * A[:, 2, 0]))
    raise NotImplementedError("up to 3 electrons per spin")


class VMC:
    def __init__(self, mol: Molecule, walkers: int = 1024, seed: int = 0, **net) -> None:
        mx.random.seed(seed)
        self.mol = mol
        self.net = FermiNetLite(mol, **net)
        self.nw = walkers
        rng = np.random.default_rng(seed)
        # walkers start near the nuclei, weighted by charge
        idx = rng.choice(len(mol.charges), size=(walkers, mol.n_elec), p=np.array(mol.charges) / sum(mol.charges))
        start = mol.positions[idx] + rng.normal(0, 0.8, (walkers, mol.n_elec, 3))
        self.r = mx.array(start.astype(np.float32))
        self.step_size = 0.3
        self.Enn = mol.nuclear_repulsion()
        self.opt = optim.Adam(learning_rate=2e-3)
        self._logpsi_batch = mx.vmap(self.net.log_psi)

    def local_energy(self, params, r):
        """E_L for a batch of walkers: −½(∇² log ψ + |∇ log ψ|²) + V."""
        def logpsi_flat(x):
            self.net.update(params)
            return self.net.log_psi(x.reshape(self.mol.n_elec, 3))

        def single(x):
            x = x.reshape(-1)
            grad_fn = mx.grad(logpsi_flat)
            g = grad_fn(x)
            lap = 0.0
            eye = mx.eye(x.shape[0])
            for k in range(x.shape[0]):
                _, t = mx.jvp(grad_fn, (x,), (eye[k],))
                lap = lap + t[0][k]
            kin = -0.5 * (lap + mx.sum(g * g))
            return kin

        kin = mx.vmap(single)(r.reshape(r.shape[0], -1))
        return kin + self._potential(r)

    def _potential(self, r):
        R = mx.array(np.asarray(self.mol.positions, np.float32))
        Z = mx.array(np.asarray(self.mol.charges, np.float32))
        ei = mx.sqrt(mx.sum((r[:, :, None, :] - R[None, None]) ** 2, axis=-1) + 1e-12)
        v = -mx.sum(Z[None, None] / ei, axis=(1, 2))
        N = self.mol.n_elec
        mask = mx.array(np.triu(np.ones((N, N), np.float32), 1))
        dij = mx.sqrt(mx.sum((r[:, :, None] - r[:, None]) ** 2, axis=-1) + 1e-12)
        v = v + mx.sum(mask[None] / dij, axis=(1, 2))
        return v + self.Enn

    def mcmc(self, steps: int = 10) -> float:
        acc = 0.0
        lp = self._logpsi_batch(self.r)
        for _ in range(steps):
            prop = self.r + self.step_size * mx.random.normal(self.r.shape)
            lp_new = self._logpsi_batch(prop)
            accept = mx.log(mx.random.uniform(shape=lp.shape)) < 2 * (lp_new - lp)
            self.r = mx.where(accept[:, None, None], prop, self.r)
            lp = mx.where(accept, lp_new, lp)
            acc += float(mx.mean(accept.astype(mx.float32)))
        rate = acc / steps
        self.step_size *= 1.1 if rate > 0.55 else 0.9 if rate < 0.45 else 1.0
        return rate

    def train(self, iters: int = 1500, log=None) -> dict:
        params = self.net.trainable_parameters()

        def loss_fn(p, r, el):
            self.net.update(p)
            lp = mx.vmap(self.net.log_psi)(r)
            centred = el - mx.mean(el)
            return 2 * mx.mean(mx.stop_gradient(centred) * lp)

        grad = mx.grad(loss_fn)
        history = []
        t0 = time.perf_counter()
        for it in range(iters):
            self.mcmc(5)
            el = self.local_energy(params, self.r)
            # robust: clip outliers of the local energy (standard in VMC)
            med = mx.median(el) if hasattr(mx, "median") else mx.mean(el)
            spread = mx.mean(mx.abs(el - med))
            el = mx.clip(el, med - 5 * spread, med + 5 * spread)
            g = grad(params, self.r, el)
            self.opt.update(self.net, g)
            params = self.net.trainable_parameters()
            mx.eval(params, self.r)
            e = float(mx.mean(el))
            history.append(e)
            if log and it % 50 == 0:
                log(it, e, time.perf_counter() - t0)
        return {"history": history}

    def evaluate(self, blocks: int = 20, per_block: int = 10) -> tuple[float, float]:
        """Energy and statistical error from decorrelated blocks."""
        params = self.net.trainable_parameters()
        means = []
        for _ in range(blocks):
            self.mcmc(per_block)
            el = self.local_energy(params, self.r)
            means.append(float(mx.mean(el)))
        m = np.array(means)
        return float(m.mean()), float(m.std() / math.sqrt(len(m)))
