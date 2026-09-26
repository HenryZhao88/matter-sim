"""Periodic DFT with the Kohn–Sham eigenproblem on a GPU in single precision (prototype).

Same physics, same algorithm and the same SCF loop as periodic.py; only where the arithmetic
happens and in what precision changes:

* on the device, complex64: the Bloch functions u_nk, H_k applied by FFT, the projectors, and
  LOBPCG's block products. This is where nearly all the time goes.
* in float64: every grid-sized quantity that feeds the energy (density, Hartree, XC, mixing,
  Ewald), the small Rayleigh–Ritz matrices of LOBPCG (formed from complex64 vectors but
  accumulated in complex128), and every sum that becomes an energy or a force. That work runs
  on the "wide" device: the GPU itself on CUDA, the CPU on Apple's MPS, which has no float64.

Projectors are rebuilt for each k-point as it is needed (``cache_projectors`` keeps them
resident instead, when the device has room): memory then scales with one k-point, not the mesh.

The NumPy path in periodic.py is the reference this must reproduce, to ~1 meV/atom in energy
and ~1e-3 Ha/bohr in forces, before any timing means anything (tests/test_crystal.py).
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ..core.accel import torch_device
from .periodic import CrystalResult, PeriodicDFT, fermi_all

C64, C128, F32, F64 = torch.complex64, torch.complex128, torch.float32, torch.float64
RESIDUAL_FLOOR = 1.0       # LOBPCG stops refining a band at this many ε₃₂·‖H‖ (see lobpcg_dev)


def _interp(x, xp, fp):
    """np.interp for a sorted table xp (clamped at the ends)."""
    i = torch.clamp(torch.searchsorted(xp, x), 1, len(xp) - 1)
    x0, x1 = xp[i - 1], xp[i]
    w = torch.clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return fp[i - 1] * (1 - w) + fp[i] * w


def _harmonics(l: int, x, y, z):
    """real_harmonics_k from atoms/pseudo.py, on tensors (complex128)."""
    if l == 0:
        return [(torch.ones_like(x) / math.sqrt(4 * math.pi)).to(C128)]
    if l == 1:
        c = math.sqrt(3 / (4 * math.pi))
        return [-1j * c * x.to(C128), -1j * c * y.to(C128), -1j * c * z.to(C128)]
    if l == 2:
        c = math.sqrt(15 / (4 * math.pi))
        return [(-(c * x * y)).to(C128), (-(c * y * z)).to(C128), (-(c * x * z)).to(C128),
                (-(math.sqrt(5 / (16 * math.pi)) * (3 * z * z - (x * x + y * y + z * z)))).to(C128),
                (-(0.5 * c * (x * x - y * y))).to(C128)]
    raise NotImplementedError(l)


class PeriodicDFTTorch(PeriodicDFT):
    def __init__(self, crystal, *args, device: str | None = None, wide_device: str | None = None,
                 small_device: str | None = "cpu", cache_projectors: bool = False, **kw) -> None:
        super().__init__(crystal, *args, **kw)
        self.dev = torch.device(device or torch_device())
        # float64 work stays on the device unless it cannot do float64 (MPS)
        self.wdev = torch.device(wide_device or ("cpu" if self.dev.type == "mps" else self.dev.type))
        # the small Rayleigh–Ritz eigenproblems (complex128, ~150²): faster on the CPU than on a GPU
        self.sdev = torch.device(small_device or self.wdev)
        self.cache_projectors = cache_projectors
        w = self.wdev
        self._Gt = torch.tensor(self.G, dtype=F64, device=w)                   # (Nx,Ny,Nz,3)
        self._filter = torch.tensor(self.filter, dtype=F64, device=w)
        self._tabs = {key: (torch.tensor(qt, dtype=F64, device=w), torch.tensor(ft, dtype=F64, device=w))
                      for key, (qt, ft) in self.proj_tab.items()}
        self._Bcache: dict = {}

    def _wide(self, x, dtype):
        return x.to(self.wdev).to(dtype)

    def _narrow(self, x):
        return x.to(C64).to(self.dev)

    def _kin(self, k):
        """½|k+G|², float64 on the wide device."""
        return 0.5 * torch.sum((self._Gt + torch.tensor(k, dtype=F64, device=self.wdev)) ** 2, dim=-1)

    # -------------------------------------------------------------- projectors
    def _projectors_dev(self, ik: int, k, keep_g: bool = False):
        """B (P, Ntot) complex64 on the device, energies E (P,) float32, atom index per row; with
        ``keep_g`` also the G-space forms (P, Nx, Ny, Nz), complex128 on the wide device."""
        if not keep_g and ik in self._Bcache:
            return self._Bcache[ik]
        c, w = self.c, self.wdev
        kG = self._Gt + torch.tensor(k, dtype=F64, device=w)
        q = torch.linalg.norm(kG, dim=-1)
        safe = torch.where(q > 0, q, torch.ones_like(q))
        n = torch.where((q > 0)[..., None], kG / safe[..., None], torch.zeros_like(kG))
        rows, Fs, E, atom = [], [], [], []
        for i, (Z, R) in enumerate(zip(c.charges, c.positions)):
            pp = self.pp[Z]
            phase = torch.exp(-1j * (kG @ torch.tensor(R, dtype=F64, device=w)).to(C128))
            for l in pp.nonlocal_channels:
                qt, ft = self._tabs[(Z, l)]
                radial = _interp(q, qt, ft) * self._filter
                for ang in _harmonics(l, n[..., 0], n[..., 1], n[..., 2]):
                    F = radial * ang * phase / c.volume
                    rows.append(self._narrow((torch.fft.ifftn(F) * self.Ntot).reshape(-1)))
                    if keep_g:
                        Fs.append(F)
                    E.append(pp.kb_energy[l])
                    atom.append(i)
        B = torch.stack(rows) if rows else torch.zeros((0, self.Ntot), dtype=C64, device=self.dev)
        out = (B, torch.tensor(E, dtype=F32, device=self.dev), atom)
        if keep_g:
            return out + (torch.stack(Fs) if Fs else None,)
        if self.cache_projectors:
            self._Bcache[ik] = out
        return out

    # -------------------------------------------------------------- Hamiltonian
    def _overlap(self, U, B):
        """⟨b_p|u_n⟩ (without dV) accumulated in complex128 on the wide device. The nonlocal energy is
        most of a transition metal's total (−238 of −242 Ha for a 4-atom Cu cell), so complex64
        accumulation of these overlaps alone biased copper's energy by 0.1 meV/atom."""
        return self._wide(U, C128) @ self._wide(B, C128).conj().T

    def _apply_dev(self, U, kin32, Veff32, B, E):
        nb = U.shape[0]
        Ug = torch.fft.fftn(U.reshape((nb,) + self.N), dim=(1, 2, 3))
        out = torch.fft.ifftn(kin32 * Ug, dim=(1, 2, 3)).reshape(nb, -1) + Veff32 * U
        if len(E):
            cvec = self._narrow(self._overlap(U, B) * self.dV * self._wide(E, F64))
            out = out + cvec @ B
        return out

    def _eig_dev(self, k, Veff32, B, E, U0, iters):
        x = self._kin(k)
        num = 27 + 18 * x + 12 * x ** 2 + 8 * x ** 3
        P = (num / (num + 16 * x ** 4)).to(F32).to(self.dev)
        kin32 = x.to(F32).to(self.dev)

        def precond(R):
            nb = R.shape[0]
            return torch.fft.ifftn(P * torch.fft.fftn(R.reshape((nb,) + self.N), dim=(1, 2, 3)),
                                   dim=(1, 2, 3)).reshape(nb, -1)

        # the smallest residual complex64 can resolve is ~ε × the size of the terms that make up H·x:
        # kinetic and local potential by their largest values, the nonlocal term by its actual size on
        # these vectors. Its operator norm E·|b|²dV is no guide: 3e4 Ha for copper's p channel
        # (E = 469 Ha), 500x the rounding it really causes, since bound states barely overlap a
        # strongly repulsive projector; used as the scale it froze copper's bands 500x too early
        h_norm = float(x.max()) + float(Veff32.abs().max()) + self._nonlocal_scale(U0, B, E)
        floor = RESIDUAL_FLOOR * torch.finfo(F32).eps * h_norm
        return lobpcg_dev(lambda X: self._apply_dev(X, kin32, Veff32, B, E), U0, precond, iters, floor, wide=self.wdev, small=self.sdev)

    def _nonlocal_scale(self, U, B, E):
        """Largest ‖Σ_p E_p ⟨b_p|u⟩ b_p‖ over the (normalised) vectors u: the nonlocal term's real size."""
        if not len(E):
            return 0.0
        Un = U / torch.clamp(torch.linalg.norm(U, dim=1), min=1e-30)[:, None]
        return float(torch.linalg.norm(self._narrow(self._overlap(Un, B) * self.dV * self._wide(E, F64)) @ B, dim=1).max())

    # -------------------------------------------------------------- SCF
    def run(self, max_iter: int = 60, tol: float = 1e-6, forces: bool = False, verbose: bool = False,
            rho_tol: float = 1e-4) -> CrystalResult:
        """As PeriodicDFT.run; ``rho_tol`` is the density-change test (the NumPy path's fixed 1e-4)."""
        if self.nspin == 2:
            return self._run_spin_dev(max_iter, tol, forces, verbose, rho_tol)
        t0 = time.perf_counter()
        c = self.c
        ne = c.valence
        nk = len(self.kpts)
        rho = np.full(self.N, ne / c.volume)
        # the same random start as the NumPy path, drawn in the same order, moved over one k at a time
        U = []
        for _ in range(nk):
            u = self._rng.standard_normal((self.n_bands, self.Ntot)) + 1j * self._rng.standard_normal((self.n_bands, self.Ntot))
            U.append(torch.tensor(u.astype(np.complex64), device=self.dev))
        hist_x, hist_f = [], []
        E_prev = np.inf
        converged = False
        for it in range(1, max_iter + 1):
            vh = self._hartree(rho)
            _, vxc, _ = self._xc(rho)
            Veff = self.Vloc + vh + vxc
            Veff32 = torch.tensor(Veff.reshape(1, -1).astype(np.float32), device=self.dev)
            evals = []
            for ik, k in enumerate(self.kpts):
                B, E, _ = self._projectors_dev(ik, k)
                lam, U[ik] = self._eig_dev(k, Veff32, B, E, U[ik], 30 if it == 1 else 5)
                evals.append(lam)
            occ, mu, S = fermi_all(evals, self.wk, ne, self.T_e, self.smearing)
            rho_out = self._symmetrize(self._density_dev(U, occ, 2))
            comps = self._energy(rho_out, U, occ, None)
            E = sum(comps.values())
            F = E - self.T_e * S
            drho = float(np.sum(np.abs(rho_out - rho)) * self.dV)
            if verbose:
                print(f"  it {it:2d}  E = {E:.8f}  dρ = {drho:.2e}", flush=True)
            tol_eff = self._energy_tol(tol, comps)
            if abs(F - E_prev) < tol_eff and drho < rho_tol:
                rho = rho_out
                converged = True
                break
            E_prev = F
            rho = self._mix(rho, rho_out, hist_x, hist_f)
        result = CrystalResult(
            energy=0.5 * (E + F), free_energy=F, components=comps, fermi=mu,
            eigenvalues=evals, occupations=occ, kpoints=self.kpts, weights=self.wk, rho=rho,
            converged=converged, iterations=it, seconds=time.perf_counter() - t0,
            notes={"precision": "complex64 eigensolver", "energy_tol": tol_eff, "final_drho": drho},
        )
        if forces:
            result.forces = self._forces(rho, U, occ, None)
        return result

    def _energy_tol(self, tol, comps):
        """The energy test cannot be tighter than complex64 rounding of the energy itself (measured:
        ~1e-6 Ha/cell jitter once converged on fcc Al), or passing it becomes a matter of luck; the
        density test, ~15x above its own noise, is what guarantees self-consistency."""
        return max(tol, torch.finfo(F32).eps * sum(abs(v) for v in comps.values()))

    def _density_dev(self, U, occ, g):
        """Σ_k g w_k Σ_n f_nk |u_nk|² / dV in float64 (NumPy, unsymmetrised); g states per band."""
        rho_t = torch.zeros(self.Ntot, dtype=F64, device=self.wdev)
        for ik in range(len(self.kpts)):
            o = torch.tensor(occ[ik], dtype=F64, device=self.wdev)
            rho_t += (g * self.wk[ik] / self.dV) * (o[:, None] * self._wide(U[ik].abs() ** 2, F64)).sum(dim=0)
        return rho_t.cpu().numpy().reshape(self.N)

    # -------------------------------------------------------------- spin-polarised SCF
    def _run_spin_dev(self, max_iter, tol, forces, verbose, rho_tol) -> CrystalResult:
        """PeriodicDFT._run_spin with the eigenproblems on the device: LSDA, one Fermi level. The mixing,
        XC, energy terms and forces are the NumPy path's own, through _band_terms and _forces_from."""
        t0 = time.perf_counter()
        c = self.c
        ne = c.valence
        nk = len(self.kpts)
        rho = self._initial_spin_density(ne)
        # the NumPy path's random start, shared by both spins (so ↑ = ↓ holds exactly without a moment)
        U0 = []
        for _ in range(nk):
            u = self._rng.standard_normal((self.n_bands, self.Ntot)) + 1j * self._rng.standard_normal((self.n_bands, self.Ntot))
            U0.append(torch.tensor(u.astype(np.complex64), device=self.dev))
        U = [U0, [u.clone() for u in U0]]
        hist_x, hist_f = [], []
        E_prev = np.inf
        converged = False
        for it in range(1, max_iter + 1):
            vh = self._hartree(rho[0] + rho[1])
            _, (vu, vd), _ = self._xc_spin(rho)
            evals = [[], []]
            for s, vxc in enumerate((vu, vd)):
                Veff32 = torch.tensor((self.Vloc + vh + vxc).reshape(1, -1).astype(np.float32), device=self.dev)
                for ik, k in enumerate(self.kpts):
                    B, E, _ = self._projectors_dev(ik, k)
                    lam, U[s][ik] = self._eig_dev(k, Veff32, B, E, U[s][ik], 30 if it == 1 else 5)
                    evals[s].append(lam)
            occ_all, mu, S = fermi_all(evals[0] + evals[1], np.concatenate([self.wk, self.wk]), ne, self.T_e,
                                       self.smearing, g=1)
            occ = [occ_all[:nk], occ_all[nk:]]
            rho_out = np.stack([self._symmetrize(self._density_dev(U[s], occ[s], 1)) for s in range(2)])
            comps = self._energy_spin(rho_out, U, occ, None)
            E = sum(comps.values())
            F = E - self.T_e * S
            drho = float(np.sum(np.abs(rho_out - rho)) * self.dV)
            if verbose:
                m = float(np.sum(rho_out[0] - rho_out[1]) * self.dV)
                print(f"  it {it:2d}  E = {E:.8f}  dρ = {drho:.2e}  M = {m:+.4f} μB", flush=True)
            tol_eff = self._energy_tol(tol, comps)
            if abs(F - E_prev) < tol_eff and drho < rho_tol:
                rho = rho_out
                converged = True
                break
            E_prev = F
            rho = self._mix_spin(rho, rho_out, hist_x, hist_f)
        m = rho[0] - rho[1]
        result = CrystalResult(
            energy=0.5 * (E + F), free_energy=F, components=comps, fermi=mu,
            eigenvalues=[np.stack(p) for p in zip(*evals)], occupations=[np.stack(p) for p in zip(*occ)],
            kpoints=self.kpts, weights=self.wk, rho=rho[0] + rho[1],
            converged=converged, iterations=it, seconds=time.perf_counter() - t0,
            notes={"precision": "complex64 eigensolver", "energy_tol": tol_eff, "final_drho": drho,
                   "top_band_occupation": max(float(o[-1]) for o in occ[0] + occ[1])},
            rho_spin=rho, moment=float(np.sum(m) * self.dV), abs_moment=float(np.sum(np.abs(m)) * self.dV),
        )
        if forces:
            result.forces = self._forces_spin(rho, U, occ)
        return result

    # -------------------------------------------------------------- energy and forces
    # PeriodicDFT's _energy, _energy_spin, _forces and _forces_spin call these two, so overriding them
    # moves every band sum onto the device for both the unpolarised and the spin-polarised paths
    def _band_terms(self, U, occ, projs, g):
        """Kinetic and nonlocal energy of the occupied bands, g states per band; float64 sums."""
        kin = 0.0
        enl = 0.0
        for ik, k in enumerate(self.kpts):
            o = torch.tensor(occ[ik], dtype=F64, device=self.wdev)
            Ug = torch.fft.fftn(U[ik].reshape((-1,) + self.N), dim=(1, 2, 3))
            per_band = torch.sum(self._kin(k) * self._wide(Ug.abs() ** 2, F64), dim=(1, 2, 3)) / self.Ntot
            kin += g * self.wk[ik] * float(o @ per_band)
            B, E, _ = self._projectors_dev(ik, k)
            if len(E):
                cc = self._overlap(U[ik], B) * math.sqrt(self.dV)
                enl += g * self.wk[ik] * float(torch.sum(o[:, None] * self._wide(E, F64)[None, :] * cc.abs() ** 2))
        return kin, enl

    def _forces_from(self, rho, v_core, bands):
        """As PeriodicDFT._forces_from: ``bands`` is (U, occ, states per band) per spin channel."""
        F = self._forces_local(rho, v_core)
        for ik, k in enumerate(self.kpts):
            B, E, atom, Fs = self._projectors_dev(ik, k, keep_g=True)
            if not len(E):
                continue
            kG = self._Gt + torch.tensor(k, dtype=F64, device=self.wdev)
            E64 = self._wide(E, F64)[None, :]
            for U, occ, g in bands:
                o = torch.tensor(occ[ik], dtype=F64, device=self.wdev)
                cc = self._overlap(U[ik], B) * math.sqrt(self.dV)                       # (nb, P)
                for a in range(3):
                    dB = self._narrow((torch.fft.ifftn(-1j * kG[..., a] * Fs, dim=(1, 2, 3)) * self.Ntot).reshape(len(E), -1))
                    dc = self._overlap(U[ik], dB) * math.sqrt(self.dV)                    # (nb, P)
                    dE = g * self.wk[ik] * torch.sum(o[:, None] * E64 * 2 * torch.real(cc.conj() * dc), dim=0)
                    for p, i in enumerate(atom):
                        F[i, a] -= float(dE[p])
        return F

    def _forces_local(self, rho, v_core=None):
        """Ewald, local-pseudopotential and core-correction forces: grid-sized, float64, as in periodic.py."""
        c = self.c
        F = self.F_ewald.copy()
        rg = np.fft.fftn(rho) / self.Ntot
        for i, (Z, R) in enumerate(zip(c.charges, c.positions)):
            vI = self.vloc_q[Z] * np.exp(-1j * (self.G @ R)) * self.filter
            F[i] -= np.real(np.sum(np.conj(rg)[..., None] * (-1j * self.G) * vI[..., None], axis=(0, 1, 2)))
        return F + self._core_forces(rho, v_core)


def lobpcg_dev(apply_H, X, precond, maxiter, floor: float, keep_tol: float | None = None, wide=None, small="cpu"):
    """lobpcg_complex from periodic.py with the blocks in complex64 on X's device and the small
    Rayleigh–Ritz matrices accumulated in complex128 on ``wide`` (default: the same device) and
    diagonalised in complex128 on ``small`` (default: the CPU; a 150² complex128 eigh takes 11–16 ms
    on a GeForce GPU, whose float64 runs at 1/64 rate, and 2–5 ms on a Ryzen 5900X). Returns float64
    NumPy eigenvalues and the vectors.

    Soft locking, which float64 can do without and complex64 cannot: a band whose residual has
    fallen to ``floor`` gets no further search direction, and the solve stops when every band is
    there. Below that floor a residual is complex64 rounding noise; fed back as a search direction
    it makes the subspace nearly singular, Rayleigh–Ritz amplifies the rounding, and eigenvalues
    appear far below the true spectrum (measured: −91 Ha against −0.06 on fcc Al, 1 k-point).

    The same limit applies to the subspace: a direction whose overlap eigenvalue is below the
    blocks' own resolution (ε of their precision, relative to the largest) is rounding, not a search
    direction, and is dropped (``keep_tol``; float64 LOBPCG can keep 1e-10). Kept at 1e-10 in complex64,
    one such direction (1.2e-10) on perfect fcc copper broke the bands' orthonormality (8e-5) and put the
    lowest band at −45.5 Ha against +0.16; the SCF never recovered, and 5 of copper's 71 labels were
    refused that way."""
    nb = X.shape[0]
    if keep_tol is None:
        keep_tol = torch.finfo(X.real.dtype).eps
    dev = X.device
    wide = wide or dev
    small = torch.device(small or wide)

    def gram(A, Bm):
        return A.to(wide).to(C128).conj() @ Bm.to(wide).to(C128).T

    def narrow(M):
        return M.to(C64).to(dev)

    def herm(M):
        return 0.5 * (M + M.conj().T)

    def orth(A):
        s, V = torch.linalg.eigh(herm(gram(A, A)).to(small))
        keep = s > keep_tol * s.max()
        return narrow((V[:, keep] / torch.sqrt(s[keep])).to(wide).T @ A.to(wide).to(C128))

    X = orth(X)[:nb]
    HX = apply_H(X)
    lam, C = torch.linalg.eigh(herm(gram(X, HX)).to(small))
    Cc = narrow(C)
    X, HX = Cc.T @ X, Cc.T @ HX
    P = HP = None
    for _ in range(maxiter):
        R = HX - lam.to(F32).to(dev)[:, None] * X
        active = torch.linalg.norm(R, dim=1) > floor
        if not bool(active.any()):
            break
        W = precond(R[active])
        W = W - narrow(gram(X, W).T) @ X
        W = W / torch.clamp(torch.linalg.norm(W, dim=1), min=1e-30)[:, None]
        HW = apply_H(W)
        blocks, hblocks = [X, W], [HX, HW]
        if P is not None:
            blocks.append(P[active])
            hblocks.append(HP[active])
        for attempt in range(2):
            S, HS = torch.cat(blocks), torch.cat(hblocks)
            Gm, Hm = herm(gram(S, S)).to(small), herm(gram(S, HS)).to(small)
            try:
                s, V = torch.linalg.eigh(Gm)
                keep = s > keep_tol * s.max()
                T = V[:, keep] / torch.sqrt(s[keep])
                e, Cr = torch.linalg.eigh(herm(T.conj().T @ Hm @ T))
                break
            except torch.linalg.LinAlgError:
                # ill-conditioned subspace: drop the history block for this step (standard LOBPCG restart)
                blocks, hblocks = blocks[:2], hblocks[:2]
                P = HP = None
        else:
            raise torch.linalg.LinAlgError("LOBPCG subspace could not be diagonalised")
        coef = T @ Cr[:, :nb]
        lam = e[:nb]
        cp = coef.clone()
        cp[:nb] = 0
        coef, cp = narrow(coef), narrow(cp)
        X, HX = coef.T @ S, coef.T @ HS
        P, HP = cp.T @ S, cp.T @ HS
        pn = torch.clamp(torch.linalg.norm(P, dim=1), min=1e-30)[:, None]
        P, HP = P / pn, HP / pn
    return lam.cpu().numpy(), X
