"""The learned potential's forces on a GPU, for molecular dynamics of 10⁵–10⁶ atoms.

The same embedded-atom energy as md.EAMForceField — the same tables, the same half neighbour
list with a skin, the same minimum-image convention and the same energy, forces and virial —
evaluated with PyTorch on CUDA (or MPS/CPU). ``compute(pos, box)`` takes and returns NumPy arrays,
so it drops into md.MD; the NumPy force field stays the reference it is tested against.

Neighbours come from a cell list built on the device: atoms are binned into cells at least
R_CUT + SKIN wide, and each atom looks only in its own and the 26 surrounding cells, in chunks so
that a million atoms fit in memory. Arithmetic is float64 unless asked otherwise: the work is
gathers and scatters, bound by memory traffic more than by the GPU's (slow) float64 units.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from ..core.accel import torch_device
from .eam import EAM, R_CUT, RHO_FLOOR
from .md import SKIN


class EAMForceFieldTorch:
    def __init__(self, model: EAM, device: str | None = None, dtype=torch.float64, chunk: int = 32768) -> None:
        self.m = model
        self.dev = torch.device(device or torch_device())
        if self.dev.type == "mps" and dtype == torch.float64:
            self.dev = torch.device("cpu")          # MPS has no float64; the CPU does it instead
        self.dt = dtype
        self.chunk = chunk
        t = model.tabulate(6000)
        self.r0 = float(t["r"][0])
        self.dr = float(t["r"][1] - t["r"][0])
        self.nt = len(t["r"])
        as_t = lambda a: torch.tensor(np.asarray(a), dtype=dtype, device=self.dev)
        self.tab = {k: as_t(t[k]) for k in ("phi", "dphi", "f", "df")}
        self.c = [float(x) for x in model.c]
        self._pairs = None
        self._ref_pos = None
        self._ref_box = None
        self.rebuilds = 0

    # -------------------------------------------------------------- tables and embedding
    def _interp(self, name, r):
        """np.interp on the uniform table (r never leaves its range: the table starts at 0.5 bohr)."""
        x = (r - self.r0) / self.dr
        i = torch.clamp(x.floor().long(), 0, self.nt - 2)
        w = torch.clamp(x - i, 0.0, 1.0)
        t = self.tab[name]
        return t[i] * (1 - w) + t[i + 1] * w

    def _F(self, rho):
        rho = torch.clamp(rho, min=RHO_FLOOR)
        c = self.c
        return c[0] * torch.sqrt(rho) + c[1] * rho + c[2] * rho ** 2 + c[3] * rho ** 3

    def _dF(self, rho):
        rho = torch.clamp(rho, min=RHO_FLOOR)
        c = self.c
        return 0.5 * c[0] / torch.sqrt(rho) + c[1] + 2 * c[2] * rho + 3 * c[3] * rho ** 2

    # -------------------------------------------------------------- neighbours
    @staticmethod
    def _min_image(d, box):
        return d - box * torch.round(d / box)

    def _build(self, pos, box):
        """All pairs i < j closer than R_CUT + SKIN, by cell list (or all pairs for a small box)."""
        n = len(pos)
        reach = R_CUT + SKIN
        ncell = torch.floor(box / reach).long()
        if bool((ncell < 3).any()):
            # fewer than three cells across: every pair, minimum image (the box must exceed 2·R_CUT,
            # as for the NumPy force field)
            i, j = torch.triu_indices(n, n, 1, device=self.dev)
            r = torch.linalg.norm(self._min_image(pos[j] - pos[i], box), dim=1)
            keep = r < reach
            return i[keep], j[keep]
        size = box / ncell
        cxyz = torch.minimum(torch.floor((pos % box) / size).long(), ncell - 1)
        nx, ny, nz = (int(v) for v in ncell)
        cid = (cxyz[:, 0] * ny + cxyz[:, 1]) * nz + cxyz[:, 2]
        order = torch.argsort(cid)
        counts = torch.bincount(cid, minlength=nx * ny * nz)
        starts = torch.cumsum(counts, 0) - counts
        M = int(counts.max())
        table = torch.full((nx * ny * nz, M), -1, dtype=torch.long, device=self.dev)
        cs = cid[order]
        table[cs, torch.arange(n, device=self.dev) - starts[cs]] = order
        offs = torch.tensor([[a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)],
                            device=self.dev)
        I, J = [], []
        for s in range(0, n, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, n), device=self.dev)
            nb = (cxyz[idx][:, None, :] + offs[None]) % ncell                  # (c, 27, 3)
            nid = (nb[..., 0] * ny + nb[..., 1]) * nz + nb[..., 2]
            cand = table[nid].reshape(len(idx), -1)                              # (c, 27·M)
            ii = idx[:, None].expand_as(cand)
            ok = cand > ii                                                       # real atoms, each pair once
            ii, jj = ii[ok], cand[ok]
            r = torch.linalg.norm(self._min_image(pos[jj] - pos[ii], box), dim=1)
            keep = r < reach
            I.append(ii[keep])
            J.append(jj[keep])
        return torch.cat(I), torch.cat(J)

    def _neighbours(self, pos, box):
        if self._ref_pos is not None and torch.equal(box, self._ref_box):
            disp = self._min_image(pos - self._ref_pos, box)
            if float(torch.linalg.norm(disp, dim=1).max()) < SKIN / 2:
                return self._pairs
        self._pairs = self._build(pos, box)
        self._ref_pos, self._ref_box = pos.clone(), box.clone()
        self.rebuilds += 1
        return self._pairs

    # -------------------------------------------------------------- energy, forces, virial
    def compute_t(self, pos, box):
        """As compute(), on device tensors: returns (E float, F tensor, virial float)."""
        i, j = self._neighbours(pos, box)
        d = self._min_image(pos[j] - pos[i], box)
        r = torch.linalg.norm(d, dim=1)
        keep = r < R_CUT
        i, j, d, r = i[keep], j[keep], d[keep], r[keep]
        n = len(pos)
        fr = self._interp("f", r)
        rho = torch.zeros(n, dtype=self.dt, device=self.dev)
        rho.index_add_(0, i, fr)
        rho.index_add_(0, j, fr)
        dF = self._dF(rho)
        E = float(self._F(rho).sum() + self._interp("phi", r).sum()) + self.m.e0 * n
        dEdr = self._interp("dphi", r) + (dF[i] + dF[j]) * self._interp("df", r)
        fvec = (dEdr / r)[:, None] * d
        F = torch.zeros((n, 3), dtype=self.dt, device=self.dev)
        F.index_add_(0, i, fvec)
        F.index_add_(0, j, -fvec)
        return E, F, float(-(dEdr * r).sum())

    def compute(self, pos, box):
        """Energy, forces and virial Σ r·f, as md.EAMForceField.compute (NumPy in, NumPy out)."""
        p = torch.as_tensor(np.asarray(pos), dtype=self.dt, device=self.dev)
        b = torch.as_tensor(np.asarray(box), dtype=self.dt, device=self.dev)
        E, F, W = self.compute_t(p, b)
        return E, F.cpu().numpy().astype(np.float64), W


class MDTorch:
    """md.MD with the state kept on the device: velocity Verlet, the Bussi thermostat and the
    Berendsen barostat, step for step as in md.py. Random numbers come from the same NumPy
    generator, drawn in the same order, so a seeded run reproduces md.MD's trajectory exactly
    (the test for this class), while a million atoms never leave the GPU between steps."""

    def __init__(self, model: EAM, state, dt_fs: float = 2.0, seed: int = 0, device: str | None = None) -> None:
        from ..core.units import AU_TIME_FS
        self.ff = EAMForceFieldTorch(model, device=device)
        dev, dt = self.ff.dev, self.ff.dt
        self.pos = torch.tensor(state.pos, dtype=dt, device=dev)
        self.vel = torch.tensor(state.vel, dtype=dt, device=dev)
        self.box = torch.tensor(state.box, dtype=dt, device=dev)
        self.mass = state.mass
        self.dt = dt_fs / AU_TIME_FS
        self.rng = np.random.default_rng(seed)
        self.E, self.F, self.W = self.ff.compute_t(self.pos, self.box)

    # the same observables as md.kinetic / temperature / pressure
    def kinetic(self) -> float:
        return 0.5 * self.mass * float((self.vel ** 2).sum())

    def temperature(self) -> float:
        from ..core.units import KELVIN_HARTREE
        return 2 * self.kinetic() / (3 * len(self.pos)) / KELVIN_HARTREE

    def pressure(self) -> float:
        return (2 * self.kinetic() + self.W) / (3 * float(torch.prod(self.box)))

    def thermalise(self, T: float) -> None:
        from ..core.units import KELVIN_HARTREE
        kT = T * KELVIN_HARTREE
        v = self.rng.normal(0, math.sqrt(kT / self.mass), tuple(self.pos.shape))
        v -= v.mean(axis=0)
        self.vel = torch.tensor(v, dtype=self.ff.dt, device=self.ff.dev)

    def step(self, T: float | None = None, P_GPa: float | None = None, tau_T_fs: float = 100.0,
             tau_P_fs: float = 1000.0, frozen=None) -> None:
        from ..core.units import AU_TIME_FS
        from .md import HA_PER_BOHR3_GPA, MAX_BOX_STEP
        dt = self.dt
        fz = None if frozen is None else torch.as_tensor(np.asarray(frozen), device=self.ff.dev)
        self.vel += 0.5 * dt * self.F / self.mass
        if fz is not None:
            self.vel[fz] = 0.0
        self.pos += dt * self.vel
        if P_GPa is not None:
            P = self.pressure() * HA_PER_BOHR3_GPA
            mu = 1 + dt / (tau_P_fs / AU_TIME_FS) * (P - P_GPa) / 76.0 / 3
            mu = float(np.clip(mu, 1 - MAX_BOX_STEP, 1 + MAX_BOX_STEP))
            self.pos *= mu
            self.box *= mu
        self.pos = torch.remainder(self.pos, self.box)
        self.E, self.F, self.W = self.ff.compute_t(self.pos, self.box)
        self.vel += 0.5 * dt * self.F / self.mass
        if fz is not None:
            self.vel[fz] = 0.0
        if T is not None:
            self._bussi(T, tau_T_fs, frozen)

    def _bussi(self, T, tau_fs, frozen):
        from ..core.units import AU_TIME_FS, KELVIN_HARTREE
        n_free = len(self.pos) if frozen is None else int((~np.asarray(frozen)).sum())
        dof = 3 * n_free - 3
        K = self.kinetic()
        if K <= 0:
            return
        Kt = 0.5 * dof * T * KELVIN_HARTREE
        c = math.exp(-self.dt / (tau_fs / AU_TIME_FS))
        R = self.rng.normal()
        S = float(np.sum(self.rng.normal(size=dof - 1) ** 2))
        Knew = K * c + Kt / dof * (1 - c) * (S + R * R) + 2 * R * math.sqrt(c * (1 - c) * K * Kt / dof)
        self.vel *= math.sqrt(max(Knew, 0) / K)

    def run(self, steps: int, T=None, P_GPa=None, sample_every: int = 10, frozen=None, callback=None):
        """As md.MD.run: rows of step, E, U, T, P (GPa), V; the same runaway guard."""
        from .md import HA_PER_BOHR3_GPA, RUNAWAY
        rows = []
        V0 = float(torch.prod(self.box))
        for k in range(steps):
            self.step(T, P_GPa, frozen=frozen)
            V = float(torch.prod(self.box))
            if V > RUNAWAY * V0 or V < V0 / RUNAWAY:
                raise RuntimeError(f"the cell ran away under the barostat (volume x{V / V0:.2g} after {k} steps)")
            if k % sample_every == 0:
                row = {"step": k, "E": self.E + self.kinetic(), "U": self.E, "T": self.temperature(),
                       "P": self.pressure() * HA_PER_BOHR3_GPA, "V": V}
                rows.append(row)
                if callback:
                    callback(row, self)
        return rows
