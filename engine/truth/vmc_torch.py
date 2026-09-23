"""Truth mode on PyTorch: the same network, sampler and optimiser as vmc_mlx.py.

A line-for-line port of the MLX implementation so the two produce the same physics; only the
random streams differ, so results agree within their statistical errors, not bit for bit.
Runs on CUDA, on Apple's Metal through MPS, or on the CPU (engine/core/accel.py picks).

Three details are kept identical on purpose, because each changes the result:

- Every array the MLX module holds is trainable, the nuclear centres ``R`` inside the network
  included (the potential always uses the true positions, so this is extra variational
  freedom, not a change to the Hamiltonian).
- MLX's Adam does not bias-correct its moment estimates; torch's does. The update is written
  out below as MLX does it.
- Everything is float32, as on the Metal GPU.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, jacrev, vmap

from ..core.accel import torch_device
from .molecule import Molecule


class FermiNetLite(nn.Module):
    def __init__(self, mol: Molecule, hidden: int = 32, layers: int = 3, dets: int = 4) -> None:
        super().__init__()
        self.mol = mol
        nI = len(mol.charges)
        self.dets = dets
        self.R = nn.Parameter(torch.tensor(np.asarray(mol.positions, np.float32)))
        self.register_buffer("Z", torch.tensor(np.asarray(mol.charges, np.float32)))   # unused by log ψ
        self.inp = nn.Linear(4 * nI, hidden)
        self.pair_in = nn.Linear(4, 16)
        self.layers = nn.ModuleList([nn.Linear(3 * hidden + 16, hidden) for _ in range(layers)])
        n_orb = max(mol.n_up, mol.n_dn, 1)
        self.orb_up = nn.Linear(hidden, dets * n_orb)
        self.orb_dn = nn.Linear(hidden, dets * n_orb)
        self.env_sigma = nn.Parameter(torch.ones(dets * n_orb, nI))
        self.env_pi = nn.Parameter(torch.ones(dets * n_orb, nI))
        self.n_orb = n_orb
        N, nu = mol.n_elec, mol.n_up
        spin = np.array([0] * nu + [1] * (N - nu))
        same = (spin[:, None] == spin[None, :]).astype(np.float32)
        self.register_buffer("cusp", torch.tensor(0.25 * same + 0.5 * (1 - same)))
        self.register_buffer("mask", torch.tensor(np.triu(np.ones((N, N), np.float32), 1)))
        _mlx_init(self)

    def forward(self, r):
        """log|ψ| for one configuration r of shape (N, 3)."""
        mol = self.mol
        N, nu = mol.n_elec, mol.n_up
        ei = r[:, None, :] - self.R[None, :, :]                       # (N, nI, 3)
        ed = torch.sqrt(torch.sum(ei * ei, dim=-1) + 1e-12)          # (N, nI)
        h = torch.tanh(self.inp(torch.cat([ei, ed[..., None]], dim=-1).reshape(N, -1)))
        rij = r[:, None, :] - r[None, :, :]
        dij = torch.sqrt(torch.sum(rij * rij, dim=-1) + 1e-12)
        g = torch.tanh(self.pair_in(torch.cat([rij, dij[..., None]], dim=-1)))   # (N, N, 16)
        gmean = torch.mean(g, dim=1)
        for layer in self.layers:
            up = torch.mean(h[:nu], dim=0, keepdim=True).expand_as(h) if nu > 0 else torch.zeros_like(h)
            dn = torch.mean(h[nu:], dim=0, keepdim=True).expand_as(h) if N - nu > 0 else torch.zeros_like(h)
            h = h + torch.tanh(layer(torch.cat([h, up, dn, gmean], dim=-1)))
        env = torch.exp(-torch.abs(self.env_sigma)[None, :, :] * ed[:, None, :])  # (N, K·n_orb, nI)
        env = torch.sum(self.env_pi[None] * env, dim=-1)                          # (N, K·n_orb)
        total = None
        for lo, hi, head in ((0, nu, self.orb_up), (nu, N, self.orb_dn)):
            n = hi - lo
            if n == 0:
                continue
            phi = head(h[lo:hi]) * env[lo:hi]                                      # (n, K·n_orb)
            phi = phi.reshape(n, self.dets, self.n_orb)[:, :, :n].permute(1, 0, 2)  # (K, n, n)
            d = _det(phi)
            total = d if total is None else total * d
        psi = torch.sum(total)
        # Jastrow with exact electron–electron cusps: ½ for opposite spins, ¼ for equal spins
        J = torch.sum(self.mask * self.cusp * dij / (1 + dij))
        return torch.log(torch.abs(psi) + 1e-30) + J


@torch.no_grad()
def _mlx_init(net: nn.Module) -> None:
    """MLX's Linear initialisation: weight and bias both U(−1/√fan_in, 1/√fan_in).

    torch's default draws the weight from the same interval, so this only pins the bias down
    explicitly rather than relying on the coincidence.
    """
    for m in net.modules():
        if isinstance(m, nn.Linear):
            s = 1.0 / math.sqrt(m.in_features)
            m.weight.uniform_(-s, s)
            m.bias.uniform_(-s, s)


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


class _AdamNoBiasCorrection:
    """Adam exactly as mlx.optimizers.Adam applies it by default (no bias correction)."""

    def __init__(self, params, learning_rate: float, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps = learning_rate, betas, eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, grads) -> None:
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.sub_(self.lr * m / (torch.sqrt(v) + self.eps))


class VMC:
    def __init__(self, mol: Molecule, walkers: int = 1024, seed: int = 0, device: str | None = None,
                 **net) -> None:
        self.device = torch.device(device or torch_device())
        torch.manual_seed(seed)
        self.mol = mol
        self.net = FermiNetLite(mol, **net).to(self.device)
        self.nw = walkers
        rng = np.random.default_rng(seed)
        # walkers start near the nuclei, weighted by charge
        idx = rng.choice(len(mol.charges), size=(walkers, mol.n_elec), p=np.array(mol.charges) / sum(mol.charges))
        start = mol.positions[idx] + rng.normal(0, 0.8, (walkers, mol.n_elec, 3))
        self.r = torch.tensor(start.astype(np.float32), device=self.device)
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.step_size = 0.3
        self.Enn = mol.nuclear_repulsion()
        self.opt = _AdamNoBiasCorrection(self.net.parameters(), learning_rate=2e-3)
        self._buffers = dict(self.net.named_buffers())
        self._R = torch.tensor(np.asarray(mol.positions, np.float32), device=self.device)
        self._Zn = torch.tensor(np.asarray(mol.charges, np.float32), device=self.device)

    def _params(self, detach: bool = True) -> dict:
        return {n: (p.detach() if detach else p) for n, p in self.net.named_parameters()}

    def _logpsi_fn(self, params):
        bufs = self._buffers

        def f(x):
            return functional_call(self.net, (params, bufs), (x,))
        return f

    def _logpsi_batch(self, r, params=None):
        return vmap(self._logpsi_fn(params if params is not None else self._params()))(r)

    def local_energy(self, r):
        """E_L for a batch of walkers: −½(∇² log ψ + |∇ log ψ|²) + V."""
        N = self.mol.n_elec
        f = self._logpsi_fn(self._params())

        def logpsi_flat(x):
            return f(x.reshape(N, 3))

        g = grad(logpsi_flat)

        def g_twice(x):
            gx = g(x)
            return gx, gx

        def single(x):
            # reverse-over-reverse: forward-over-reverse (jacfwd) is what MLX does, but torch
            # 2.11's forward mode promotes a tangent to float64 inside this network and fails.
            hess, gx = jacrev(g_twice, has_aux=True)(x)
            return -0.5 * (torch.diagonal(hess).sum() + torch.sum(gx * gx))

        kin = vmap(single)(r.reshape(r.shape[0], -1))
        return kin.detach() + self._potential(r)

    def _potential(self, r):
        R, Z = self._R, self._Zn
        ei = torch.sqrt(torch.sum((r[:, :, None, :] - R[None, None]) ** 2, dim=-1) + 1e-12)
        v = -torch.sum(Z[None, None] / ei, dim=(1, 2))
        mask = self._buffers["mask"]
        dij = torch.sqrt(torch.sum((r[:, :, None] - r[:, None]) ** 2, dim=-1) + 1e-12)
        v = v + torch.sum(mask[None] / dij, dim=(1, 2))
        return v + self.Enn

    @torch.no_grad()
    def mcmc(self, steps: int = 10) -> float:
        acc = torch.zeros((), device=self.device)
        params = self._params()
        lp = self._logpsi_batch(self.r, params)
        for _ in range(steps):
            prop = self.r + self.step_size * torch.randn(self.r.shape, generator=self.gen, device=self.device)
            lp_new = self._logpsi_batch(prop, params)
            u = torch.rand(lp.shape, generator=self.gen, device=self.device)
            accept = torch.log(u) < 2 * (lp_new - lp)
            self.r = torch.where(accept[:, None, None], prop, self.r)
            lp = torch.where(accept, lp_new, lp)
            acc += accept.float().mean()
        rate = float(acc) / steps
        self.step_size *= 1.1 if rate > 0.55 else 0.9 if rate < 0.45 else 1.0
        return rate

    def train(self, iters: int = 1500, log=None) -> dict:
        history = []
        t0 = time.perf_counter()
        for it in range(iters):
            self.mcmc(5)
            el = self.local_energy(self.r)
            # robust: clip outliers of the local energy (standard in VMC)
            med = torch.median(el)
            spread = torch.mean(torch.abs(el - med))
            el = torch.clamp(el, med - 5 * spread, med + 5 * spread)
            params = self._params(detach=False)
            lp = self._logpsi_batch(self.r, params)
            centred = (el - torch.mean(el)).detach()
            loss = 2 * torch.mean(centred * lp)
            grads = torch.autograd.grad(loss, list(params.values()), allow_unused=True)
            grads = [torch.zeros_like(p) if g is None else g for p, g in zip(params.values(), grads)]
            self.opt.step(grads)
            e = float(torch.mean(el))
            history.append(e)
            if log and it % 50 == 0:
                log(it, e, time.perf_counter() - t0)
        return {"history": history}

    def evaluate(self, blocks: int = 20, per_block: int = 10) -> tuple[float, float]:
        """Energy and statistical error from decorrelated blocks."""
        means = []
        for _ in range(blocks):
            self.mcmc(per_block)
            el = self.local_energy(self.r)
            means.append(float(torch.mean(el)))
        m = np.array(means)
        return float(m.mean()), float(m.std() / math.sqrt(len(m)))
