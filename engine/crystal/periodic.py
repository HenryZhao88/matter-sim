"""Crystals: electrons in an infinite periodic solid (Kohn–Sham DFT with Bloch waves).

A crystal is a cell that repeats forever. Each electron state is a Bloch wave
ψ_nk(r) = e^{ik·r} u_nk(r) with u periodic, labelled by a crystal momentum k in the
Brillouin zone; sums over the whole infinite crystal become averages over a mesh of k.

Same physics as the molecule solver (engine/electrons), in periodic form:
* kinetic energy ½|k+G|² applied exactly in Fourier space;
* Hartree potential 4πρ(G)/G² (the G = 0 term cancels against the ions);
* the engine's own pseudopotentials: local part in Fourier space, Kleinman–Bylander
  projectors with Bloch phases;
* LDA exchange–correlation;
* ion–ion energy by Ewald summation (exact for point charges in a neutralising lattice);
* Fermi–Dirac smearing, because metals have no gap.

Only the nuclei, their positions and the cell are input. Which structure is stable,
the lattice spacing, the stiffness and whether it conducts all come out.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import permutations, product

import numpy as np
from scipy.special import erfc

from ..atoms import species
from ..atoms.pseudo import R_GAUSS, real_harmonics_k
from ..core.grid import fft_friendly
from ..electrons.xc import lda_xc


# ------------------------------------------------------------------ geometry
@dataclass
class Crystal:
    cell: np.ndarray                 # (3,) orthorhombic side lengths, bohr
    charges: list[int]
    positions: np.ndarray            # (n, 3) cartesian, bohr

    def __post_init__(self) -> None:
        self.cell = np.asarray(self.cell, float)
        self.positions = np.asarray(self.positions, float).reshape(-1, 3) % self.cell
        for Z in self.charges:
            if not species.is_pseudized(Z):
                raise ValueError("crystals use pseudopotential elements (Li and heavier)")

    @property
    def volume(self) -> float:
        return float(np.prod(self.cell))

    @property
    def valence(self) -> float:
        return float(sum(species.valence_charge(Z) for Z in self.charges))


def cubic(kind: str, a: float, Z: int, strain: tuple[float, float, float] = (0, 0, 0)) -> Crystal:
    """Conventional cubic cell of simple-cubic, BCC or FCC with lattice constant a (bohr)."""
    basis = {
        "sc": [(0, 0, 0)],
        "bcc": [(0, 0, 0), (0.5, 0.5, 0.5)],
        "fcc": [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
    }[kind]
    cell = a * (1 + np.asarray(strain, float))
    return Crystal(cell, [Z] * len(basis), np.array(basis) * cell)


# ------------------------------------------------------------------ symmetry
def cubic_operations() -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """The 48 point operations of a cube: axis permutation + sign flips."""
    return [(p, s) for p in permutations(range(3)) for s in product((1, -1), repeat=3)]


def crystal_symmetries(c: Crystal, tol: float = 1e-6):
    """Cube operations (about the origin) that map the cell and every atom onto the crystal."""
    ops = []
    frac = c.positions / c.cell
    for perm, sign in cubic_operations():
        if not np.allclose(c.cell[list(perm)], c.cell):
            continue
        moved = (frac[:, list(perm)] * np.array(sign)) % 1.0
        ok = True
        for m, Z in zip(moved, c.charges):
            d = np.abs(((frac - m) + 0.5) % 1.0 - 0.5).max(axis=1)
            if not any(dd < tol and Zj == Z for dd, Zj in zip(d, c.charges)):
                ok = False
                break
        if ok:
            ops.append((perm, sign))
    return ops


def kpoint_mesh(c: Crystal, n: int | tuple[int, int, int], ops, shift: bool = True):
    """Monkhorst–Pack mesh reduced by symmetry (and time reversal). Returns (k cartesian, weights)."""
    ns = (n, n, n) if isinstance(n, int) else n
    pts = []
    for idx in product(*[range(m) for m in ns]):
        f = np.array([(i + (0.5 if shift else 0.0)) / m - 0.5 for i, m in zip(idx, ns)])
        pts.append(f)
    pts = np.array(pts)
    key = lambda f: tuple(np.round(((f + 0.5) % 1.0 - 0.5) * 1e6).astype(int))
    rep: dict = {}
    ops_all = ops + [(p, tuple(-x for x in s)) for p, s in ops]          # time reversal: k → −k
    for f in pts:
        images = {key(f[list(p)] * np.array(s)) for p, s in ops_all}
        canon = min(images)
        rep.setdefault(canon, [f, 0])[1] += 1
    fr = np.array([v[0] for v in rep.values()])
    w = np.array([v[1] for v in rep.values()], float)
    return 2 * np.pi * fr / c.cell, w / w.sum()


# ------------------------------------------------------------------ the solver
@dataclass
class CrystalResult:
    energy: float                 # E (Hartree per cell), extrapolated to zero smearing
    free_energy: float
    components: dict
    fermi: float
    eigenvalues: list             # per k
    occupations: list
    kpoints: np.ndarray
    weights: np.ndarray
    rho: np.ndarray
    converged: bool
    iterations: int
    seconds: float
    forces: np.ndarray | None = None
    notes: dict = field(default_factory=dict)


class PeriodicDFT:
    def __init__(self, crystal: Crystal, h: float = 0.3, kmesh: int | tuple = 6, T_e: float = 0.005,
                 symmetry: bool = True, extra_bands: int = 6) -> None:
        self.c = crystal
        self.T_e = T_e
        self.N = tuple(fft_friendly(math.ceil(L / h)) for L in crystal.cell)
        self.Ntot = int(np.prod(self.N))
        self.dV = crystal.volume / self.Ntot
        gs = [2 * np.pi * np.fft.fftfreq(n, d=L / n) for n, L in zip(self.N, crystal.cell)]
        self.G = np.stack(np.meshgrid(*gs, indexing="ij"), axis=-1)      # (Nx,Ny,Nz,3)
        self.G2 = np.sum(self.G ** 2, axis=-1)
        gmax = min(np.pi * n / L for n, L in zip(self.N, crystal.cell))
        self.filter = np.exp(-36.0 * np.minimum(np.sqrt(self.G2) / gmax, 1.5) ** 36)
        self.ops = crystal_symmetries(crystal) if symmetry else [((0, 1, 2), (1, 1, 1))]
        self.kpts, self.wk = kpoint_mesh(crystal, kmesh, self.ops)
        self.n_bands = int(math.ceil(crystal.valence / 2)) + extra_bands
        self._form_factors()
        self._rng = np.random.default_rng(0)

    # -------------------------------------------------------------- ions
    def _form_factors(self) -> None:
        c = self.c
        q = np.sqrt(self.G2)
        qtab = np.linspace(0, q.max() * 1.001 + 1, 3000)
        self.pp = {Z: species.pseudopotential(Z) for Z in set(c.charges)}
        self.vloc_q: dict = {}
        self.proj_tab: dict = {}
        self.core_q: dict = {}
        for Z, pp in self.pp.items():
            short = np.interp(q, qtab, pp.local_short_range_q(qtab))
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = -4 * np.pi * pp.Z_val * np.exp(-self.G2 * R_GAUSS ** 2 / 4) / self.G2
            v = lr + short
            v[0, 0, 0] = pp.local_short_range_q(np.array([0.0]))[0] + math.pi * pp.Z_val * R_GAUSS ** 2
            self.vloc_q[Z] = v
            for l in pp.nonlocal_channels:
                self.proj_tab[(Z, l)] = (qtab, pp.projector_q(l, qtab))
            if pp.rho_core is not None:
                self.core_q[Z] = np.interp(q, qtab, pp.core_density_q(qtab))
        self._set_local()

    def _set_local(self) -> None:
        c = self.c
        Vg = np.zeros(self.N, complex)
        for Z, R in zip(c.charges, c.positions):
            Vg += self.vloc_q[Z] * np.exp(-1j * (self.G @ R))
        Vg *= self.filter / c.volume
        self.Vloc_g = Vg
        self.Vloc = np.real(np.fft.ifftn(Vg) * self.Ntot)
        self.rho_core = 0.0
        if self.core_q:
            Cg = np.zeros(self.N, complex)
            for Z, R in zip(c.charges, c.positions):
                if Z in self.core_q:
                    Cg += self.core_q[Z] * np.exp(-1j * (self.G @ R))
            self.rho_core = np.real(np.fft.ifftn(Cg * self.filter / c.volume) * self.Ntot)
        self.E_ewald, self.F_ewald = ewald(c)

    def _projectors(self, k):
        """Bloch-periodic projector parts b(r) for crystal momentum k, plus their G-space forms."""
        c = self.c
        kG = self.G + k
        q = np.linalg.norm(kG, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            n = np.where(q[..., None] > 0, kG / q[..., None], 0.0)
        rows, Fs, E, atom = [], [], [], []
        for i, (Z, R) in enumerate(zip(c.charges, c.positions)):
            pp = self.pp[Z]
            phase = np.exp(-1j * (kG @ R))
            for l in pp.nonlocal_channels:
                qt, ft = self.proj_tab[(Z, l)]
                radial = np.interp(q, qt, ft) * self.filter
                angs = real_harmonics_k(l, [n[..., 0], n[..., 1], n[..., 2]])
                for ang in angs:
                    F = radial * ang * phase / c.volume
                    rows.append(np.fft.ifftn(F) * self.Ntot)
                    Fs.append(F)
                    E.append(pp.kb_energy[l])
                    atom.append(i)
        B = np.array(rows).reshape(len(rows), -1) if rows else np.zeros((0, self.Ntot), complex)
        return B, Fs, np.array(E), atom

    # -------------------------------------------------------------- Hamiltonian
    def _apply(self, U, k, Veff, B, E):
        """H_k u for a block U of shape (nb, Ntot)."""
        nb = U.shape[0]
        Ug = np.fft.fftn(U.reshape((nb,) + self.N), axes=(1, 2, 3))
        kin = 0.5 * np.sum((self.G + k) ** 2, axis=-1)
        T = np.fft.ifftn(kin * Ug, axes=(1, 2, 3)).reshape(nb, -1)
        out = T + Veff.reshape(1, -1) * U
        if len(E):
            cvec = (U @ B.conj().T) * self.dV
            out = out + (cvec * E) @ B
        return out

    def _eig(self, k, Veff, B, E, U0, iters):
        kin = 0.5 * np.sum((self.G + k) ** 2, axis=-1)
        x = kin / 1.0
        num = 27 + 18 * x + 12 * x ** 2 + 8 * x ** 3
        P = num / (num + 16 * x ** 4)

        def precond(R):
            nb = R.shape[0]
            Rg = np.fft.fftn(R.reshape((nb,) + self.N), axes=(1, 2, 3))
            return np.fft.ifftn(P * Rg, axes=(1, 2, 3)).reshape(nb, -1)

        return lobpcg_complex(lambda X: self._apply(X, k, Veff, B, E), U0, precond, iters)

    # -------------------------------------------------------------- SCF
    def run(self, max_iter: int = 60, tol: float = 1e-6, forces: bool = False, verbose: bool = False) -> CrystalResult:
        t0 = time.perf_counter()
        c = self.c
        ne = c.valence
        nk = len(self.kpts)
        # start from overlapping neutral-atom-like densities (numerical starting point only)
        rho = np.full(self.N, ne / c.volume)
        U = [self._rng.standard_normal((self.n_bands, self.Ntot)) + 1j * self._rng.standard_normal((self.n_bands, self.Ntot))
             for _ in range(nk)]
        projs = [self._projectors(k) for k in self.kpts]
        hist_x, hist_f = [], []
        E_prev = np.inf
        converged = False
        for it in range(1, max_iter + 1):
            vh = self._hartree(rho)
            rx = rho + self.rho_core                         # + partial core (nonlinear core correction)
            _, vxc, _ = lda_xc(rx / 2, rx / 2)
            Veff = self.Vloc + vh + vxc
            evals = []
            for ik, k in enumerate(self.kpts):
                B, _, E, _ = projs[ik]
                lam, U[ik] = self._eig(k, Veff, B, E, U[ik], 30 if it == 1 else 5)
                evals.append(lam)
            occ, mu, S = fermi_all(evals, self.wk, ne, self.T_e)
            rho_out = np.zeros(self.N)
            for ik in range(nk):
                dens = np.sum(occ[ik][:, None] * np.abs(U[ik]) ** 2, axis=0).reshape(self.N)
                rho_out += 2 * self.wk[ik] * dens / self.dV
            rho_out = self._symmetrize(rho_out)
            comps = self._energy(rho_out, U, occ, projs)
            E = sum(comps.values())
            F = E - self.T_e * S
            drho = float(np.sum(np.abs(rho_out - rho)) * self.dV)
            if verbose:
                print(f"  it {it:2d}  E = {E:.8f}  dρ = {drho:.2e}", flush=True)
            if abs(F - E_prev) < tol and drho < 1e-4:
                rho = rho_out
                converged = True
                break
            E_prev = F
            rho = self._mix(rho, rho_out, hist_x, hist_f)
        result = CrystalResult(
            energy=0.5 * (E + F), free_energy=F, components=comps, fermi=mu,
            eigenvalues=evals, occupations=occ, kpoints=self.kpts, weights=self.wk, rho=rho,
            converged=converged, iterations=it, seconds=time.perf_counter() - t0,
        )
        if forces:
            result.forces = self._forces(rho, U, occ, projs)
        return result

    def _hartree(self, rho):
        rg = np.fft.fftn(rho) / self.Ntot
        with np.errstate(divide="ignore", invalid="ignore"):
            vg = np.where(self.G2 > 0, 4 * np.pi * rg / self.G2, 0.0)
        return np.real(np.fft.ifftn(vg) * self.Ntot)

    def _symmetrize(self, rho):
        if len(self.ops) <= 1:
            return rho
        acc = np.zeros_like(rho)
        for perm, sign in self.ops:
            r = np.transpose(rho, perm)
            for ax, s in enumerate(sign):
                if s < 0:
                    r = np.roll(np.flip(r, axis=ax), 1, axis=ax)
            acc += r
        return acc / len(self.ops)

    def _mix(self, rho_in, rho_out, hist_x, hist_f, beta=0.3, q0=1.2, keep=7):
        """Pulay mixing with Kerker preconditioning (damps long-wavelength charge sloshing in metals)."""
        res = rho_out - rho_in
        rg = np.fft.fftn(res)
        kerker = self.G2 / (self.G2 + q0 * q0)
        kerker[0, 0, 0] = 0.0
        pres = np.real(np.fft.ifftn(kerker * rg))
        hist_x.append(rho_in.ravel().copy())
        hist_f.append(res.ravel().copy())
        if len(hist_x) > keep:
            hist_x.pop(0)
            hist_f.pop(0)
        Fm = np.array(hist_f)
        A = Fm @ Fm.T
        m = len(hist_f)
        M = np.zeros((m + 1, m + 1))
        M[:m, :m] = A / max(np.abs(A).max(), 1e-300)
        M[m, :m] = M[:m, m] = 1
        rhs = np.zeros(m + 1)
        rhs[m] = 1
        try:
            coef = np.linalg.solve(M, rhs)[:m]
        except np.linalg.LinAlgError:
            coef = np.zeros(m)
            coef[-1] = 1
        x = coef @ np.array(hist_x)
        f = coef @ Fm
        fg = np.fft.fftn(f.reshape(self.N))
        new = x.reshape(self.N) + beta * np.real(np.fft.ifftn(kerker * fg))
        new = np.maximum(new, 0)
        return new * (self.c.valence / (new.sum() * self.dV))

    def _energy(self, rho, U, occ, projs):
        c = self.c
        kin = 0.0
        enl = 0.0
        for ik, k in enumerate(self.kpts):
            Ug = np.fft.fftn(U[ik].reshape((-1,) + self.N), axes=(1, 2, 3))
            kk = 0.5 * np.sum((self.G + k) ** 2, axis=-1)
            per_band = np.sum(kk * np.abs(Ug) ** 2, axis=(1, 2, 3)) / self.Ntot
            kin += 2 * self.wk[ik] * float(np.dot(occ[ik], per_band))
            B, _, E, _ = projs[ik]
            if len(E):
                cc = (U[ik] @ B.conj().T) * math.sqrt(self.dV)
                enl += 2 * self.wk[ik] * float(np.sum(occ[ik][:, None] * E[None, :] * np.abs(cc) ** 2))
        rg = np.fft.fftn(rho) / self.Ntot
        with np.errstate(divide="ignore", invalid="ignore"):
            eh = 0.5 * c.volume * float(np.sum(np.where(self.G2 > 0, 4 * np.pi * np.abs(rg) ** 2 / self.G2, 0)))
        rx = rho + self.rho_core
        exc = float(np.sum(lda_xc(rx / 2, rx / 2)[0]) * self.dV)
        eloc = float(np.sum(self.Vloc * rho) * self.dV)
        return {"kinetic": kin, "electron_ion": eloc + enl, "hartree": eh, "exchange_correlation": exc,
                "ion_ion": self.E_ewald}

    def _forces(self, rho, U, occ, projs):
        c = self.c
        n = len(c.charges)
        F = self.F_ewald.copy()
        rg = np.fft.fftn(rho) / self.Ntot
        for i, (Z, R) in enumerate(zip(c.charges, c.positions)):
            vI = self.vloc_q[Z] * np.exp(-1j * (self.G @ R)) * self.filter
            # E_loc = Σ_G conj(ρ_G) v_I(G): dE/dR = Σ conj(ρ_G)(−iG) v_I
            dE = np.real(np.sum(np.conj(rg)[..., None] * (-1j * self.G) * vI[..., None], axis=(0, 1, 2)))
            F[i] -= dE
        if self.core_q:
            rx = rho + self.rho_core
            _, vxc, _ = lda_xc(rx / 2, rx / 2)
            vg = np.fft.fftn(vxc) / self.Ntot
            for i, (Z, R) in enumerate(zip(c.charges, c.positions)):
                if Z in self.core_q:
                    cI = self.core_q[Z] * np.exp(-1j * (self.G @ R)) * self.filter
                    F[i] -= np.real(np.sum(np.conj(vg)[..., None] * (-1j * self.G) * cI[..., None], axis=(0, 1, 2)))
        for ik, k in enumerate(self.kpts):
            B, Fs, E, atom = projs[ik]
            if not len(E):
                continue
            cc = (U[ik] @ B.conj().T) * math.sqrt(self.dV)                   # (nb, P)
            kG = self.G + k
            for p in range(len(E)):
                for a in range(3):
                    dB = (np.fft.ifftn(-1j * kG[..., a] * Fs[p]) * self.Ntot).ravel()   # ∂b/∂R_a
                    dc = (U[ik] @ dB.conj()) * math.sqrt(self.dV)
                    dE = 2 * self.wk[ik] * float(np.sum(occ[ik] * E[p] * 2 * np.real(np.conj(cc[:, p]) * dc)))
                    F[atom[p], a] -= dE
        return F


# ------------------------------------------------------------------ helpers
def fermi_all(evals, wk, ne, T):
    """Occupations (0..1 per spin state) with one chemical potential across all k."""
    flat = np.concatenate(evals)

    def count(mu):
        return sum(2 * w * np.sum(0.5 * (1 - np.tanh((e - mu) / (2 * T)))) for e, w in zip(evals, wk))

    lo, hi = flat.min() - 1, flat.max() + 1
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if count(mu) > ne:
            hi = mu
        else:
            lo = mu
    occ = [0.5 * (1 - np.tanh((e - mu) / (2 * T))) for e in evals]
    S = 0.0
    for f, w in zip(occ, wk):
        fc = np.clip(f, 1e-300, 1 - 1e-16)
        S += -2 * w * float(np.sum(fc * np.log(fc) + (1 - fc) * np.log1p(-fc)))
    return occ, mu, S


def lobpcg_complex(apply_H, X, precond, maxiter):
    """Block LOBPCG for a Hermitian operator on complex vectors (rows of X)."""
    nb = X.shape[0]

    def orth(A):
        Gm = A.conj() @ A.T
        s, V = np.linalg.eigh(0.5 * (Gm + Gm.conj().T))
        keep = s > 1e-12 * s.max()
        return (V[:, keep] / np.sqrt(s[keep])).T @ A

    X = orth(X)[:nb]
    HX = apply_H(X)
    Hm = X.conj() @ HX.T
    lam, C = np.linalg.eigh(0.5 * (Hm + Hm.conj().T))
    X, HX = C.T @ X, C.T @ HX
    P = HP = None
    for _ in range(maxiter):
        R = HX - lam[:, None] * X
        if np.max(np.linalg.norm(R, axis=1)) < 1e-7:
            break
        W = precond(R)
        W -= (X.conj() @ W.T).T @ X
        W /= np.maximum(np.linalg.norm(W, axis=1), 1e-30)[:, None]
        HW = apply_H(W)
        blocks = [X, W] + ([P] if P is not None else [])
        hblocks = [HX, HW] + ([HP] if P is not None else [])
        for attempt in range(2):
            S, HS = np.concatenate(blocks), np.concatenate(hblocks)
            Gm = S.conj() @ S.T
            Hm = S.conj() @ HS.T
            Gm, Hm = 0.5 * (Gm + Gm.conj().T), 0.5 * (Hm + Hm.conj().T)
            try:
                s, V = np.linalg.eigh(Gm)
                keep = s > 1e-12 * s.max()
                T = V[:, keep] / np.sqrt(s[keep])
                e, Cr = np.linalg.eigh(T.conj().T @ Hm @ T)
                break
            except np.linalg.LinAlgError:
                # ill-conditioned subspace: drop the history block for this step (standard LOBPCG restart)
                blocks, hblocks = blocks[:2], hblocks[:2]
                P = HP = None
        else:
            raise np.linalg.LinAlgError("LOBPCG subspace could not be diagonalised")
        coef = T @ Cr[:, :nb]
        lam = e[:nb]
        cp = coef.copy()
        cp[:nb] = 0
        X, HX = coef.T @ S, coef.T @ HS
        P, HP = cp.T @ S, cp.T @ HS
        pn = np.maximum(np.linalg.norm(P, axis=1), 1e-30)[:, None]
        P, HP = P / pn, HP / pn
    return lam, X


def ewald(c: Crystal, eta: float | None = None):
    """Ion–ion energy and forces of point valence charges in a neutralising background."""
    Z = np.array([species.valence_charge(z) for z in c.charges])
    R = c.positions
    L = c.cell
    V = c.volume
    eta = eta or math.sqrt(math.pi) / (V ** (1 / 3))
    rcut = 6.0 / eta
    gcut = 12.0 * eta
    nmax = np.ceil(rcut / L).astype(int)
    E = 0.0
    F = np.zeros_like(R)
    n = len(Z)
    for shift in product(*[range(-m, m + 1) for m in nmax]):
        T = np.array(shift) * L
        for i in range(n):
            for j in range(n):
                if i == j and not any(shift):
                    continue
                d = R[i] - R[j] + T
                r = float(np.linalg.norm(d))
                if r > rcut:
                    continue
                E += 0.5 * Z[i] * Z[j] * erfc(eta * r) / r
                dEdr = Z[i] * Z[j] * (-erfc(eta * r) / r ** 2 - 2 * eta / math.sqrt(math.pi) * math.exp(-(eta * r) ** 2) / r)
                F[i] -= dEdr * d / r
    gmax = np.ceil(gcut * L / (2 * math.pi)).astype(int)
    for m in product(*[range(-g, g + 1) for g in gmax]):
        if not any(m):
            continue
        G = 2 * math.pi * np.array(m) / L
        G2 = float(G @ G)
        if G2 > gcut ** 2:
            continue
        ph = G @ R.T
        S = np.sum(Z * np.exp(1j * ph))
        pref = 4 * math.pi / V * math.exp(-G2 / (4 * eta * eta)) / G2
        E += 0.5 * pref * abs(S) ** 2
        # dE/dR_i = pref · Re[ i G Z_i e^{iG·R_i} conj(S) ]
        for i in range(n):
            F[i] -= pref * np.real(1j * Z[i] * np.exp(1j * ph[i]) * np.conj(S)) * G
    E -= eta / math.sqrt(math.pi) * float(np.sum(Z * Z))
    E -= math.pi * float(Z.sum()) ** 2 / (2 * V * eta * eta)
    return E, F
