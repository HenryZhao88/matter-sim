"""Tree-level scattering amplitudes for any process, from the model's vertices.

Berends–Giele recursion: for every subset of external particles, build the off-shell
"current" of each field the subset can turn into, by joining smaller subsets through
every vertex in the Lagrangian, then attaching a propagator. The amplitude is the
full current contracted with the last particle. No process or diagram is written down
anywhere; if the vertices allow a final state, it appears with the right rate.

Currents live in the model's real field basis:
  ψ  (column spinors)   shape (B, n_fstates, 4)
  ψ̄  (row spinors)      shape (B, n_fstates, 4)
  V  (vectors, upper μ) shape (B, n_vectors, 4)
  S  (the Higgs)        shape (B,)
B runs over every combination of external spins and colours, summed at the end.

Conventions: metric (+,−,−,−); Weyl basis; incoming momenta positive. Feynman rules are
i × (coefficient in the Lagrangian) with the symmetry factors written out beside each
vertex below.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .model import Model

METRIC = np.diag([1.0, -1.0, -1.0, -1.0])
_S = [np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], complex),
      np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]
GAMMA = np.zeros((4, 4, 4), complex)          # γ^μ (upper index)
for _mu in range(4):
    sig = _S[_mu]
    sigbar = _S[_mu] if _mu == 0 else -_S[_mu]
    GAMMA[_mu, :2, 2:] = sig
    GAMMA[_mu, 2:, :2] = sigbar
PL = np.diag([1, 1, 0, 0]).astype(complex)
PR = np.diag([0, 0, 1, 1]).astype(complex)
GL = np.einsum("mab,bc->mac", GAMMA, PL)       # γ^μ P_L
GR = np.einsum("mab,bc->mac", GAMMA, PR)
GAMMA0 = GAMMA[0]


def slash(p: np.ndarray) -> np.ndarray:
    """p̸ = γ^μ p_μ for momenta p of shape (..., 4) (upper index)."""
    p_low = p @ METRIC
    return np.einsum("...m,mab->...ab", p_low, GAMMA)


def mdot(a, b):
    return a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1] - a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3]


# ------------------------------------------------------------------ external wavefunctions
def _sqrt_pdotsigma(p, sign):
    """√(p·σ) (sign=+1) or √(p·σ̄) (sign=−1) as a 2×2 matrix (Peskin 3.62)."""
    E, pv = p[0], p[1:]
    A = E * np.eye(2) - sign * sum(pv[i] * _S[i + 1] for i in range(3))
    w, U = np.linalg.eigh(A)
    return U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ U.conj().T


def spinors(p):
    """Complete sets {u(p,s)}, {v(p,s)}, s = 1, 2 (Σuū = p̸+m, Σvv̄ = p̸−m)."""
    a, b = _sqrt_pdotsigma(p, +1), _sqrt_pdotsigma(p, -1)
    basis = [np.array([1, 0], complex), np.array([0, 1], complex)]
    u = [np.concatenate([a @ x, b @ x]) for x in basis]
    v = [np.concatenate([a @ x, -(b @ x)]) for x in basis]
    return u, v


def polarisations(p, mass):
    """Real polarisation vectors (upper index): 2 transverse (+1 longitudinal if massive)."""
    k = p[1:]
    kk = np.linalg.norm(k)
    n = k / kk if kk > 1e-12 else np.array([0.0, 0.0, 1.0])
    trial = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, trial)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    out = [np.concatenate([[0.0], e1]), np.concatenate([[0.0], e2])]
    if mass > 0:
        out.append(np.concatenate([[kk / mass], p[0] * n / mass]))
    return out


@dataclass
class Leg:
    """One external particle: what it is, whether it comes in or goes out, its momentum."""
    kind: str               # "f" (fermion), "fbar" (antifermion), "v" (vector), "s" (scalar)
    states: list            # flavour×colour states (fermions) or species mixtures (vectors)
    incoming: bool
    mass: float
    label: str

    @property
    def is_fermion(self) -> bool:
        return self.kind in ("f", "fbar")


def external_states(model: Model, leg: Leg, p: np.ndarray):
    """[(current type, component array)] for every spin/colour state of this leg."""
    nF, nV = model.n_fstates, len(model.vectors)
    out = []
    if leg.is_fermion:
        u, v = spinors(p)
        for state in leg.states:
            for s in range(2):
                comp = np.zeros((nF, 4), complex)
                if leg.kind == "f" and leg.incoming:
                    comp[state] = u[s];                      out.append(("psi", comp))
                elif leg.kind == "fbar" and not leg.incoming:
                    comp[state] = v[s];                      out.append(("psi", comp))
                elif leg.kind == "f" and not leg.incoming:
                    comp[state] = u[s].conj() @ GAMMA0;       out.append(("psibar", comp))
                else:  # incoming antifermion
                    comp[state] = v[s].conj() @ GAMMA0;       out.append(("psibar", comp))
    elif leg.kind == "v":
        for mix in leg.states:                # mix: complex coefficients over vector species
            mix = np.asarray(mix)
            if not leg.incoming:
                mix = mix.conj()
            for eps in polarisations(p, leg.mass):
                out.append(("vec", np.outer(mix, eps)))
    else:
        out.append(("scalar", np.ones(())))
    return out


# ------------------------------------------------------------------ recursion
TYPES = ("psi", "psibar", "vec", "scalar")


def _perm_sign(seq: list[int]) -> int:
    s = 1
    seq = list(seq)
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                s = -s
    return s


def reachable(model: Model, legs: list[Leg]) -> tuple[np.ndarray, np.ndarray]:
    """Fermion states and vector species reachable from the external legs through the vertices.

    Fields that can never appear in any diagram for this process are dropped, which is exact
    and makes each amplitude cheap.
    """
    coup = (np.abs(model.ffv_L) + np.abs(model.ffv_R)) > 0          # [v, i, j]
    F = set()
    V = set()
    for lg in legs:
        if lg.is_fermion:
            F.update(lg.states)
        elif lg.kind == "v":
            for mix in lg.states:
                V.update(np.nonzero(np.abs(np.asarray(mix)) > 0)[0].tolist())
    vvv = np.abs(model.vvv) > 0
    vvs = np.abs(model.vvs) > 0
    changed = True
    while changed:
        changed = False
        for v in range(coup.shape[0]):
            if v not in V and F and coup[v][np.ix_(sorted(F), range(coup.shape[2]))].any():
                V.add(v); changed = True
        for v in list(V):
            new = set(np.nonzero(coup[v][sorted(F)].any(axis=0))[0].tolist()) - F if F else set()
            if new:
                F |= new; changed = True
        for a in range(vvv.shape[0]):
            if a not in V and V and vvv[a][np.ix_(sorted(V), sorted(V))].any():
                V.add(a); changed = True
        for a in range(vvs.shape[0]):   # through a Higgs line
            if a not in V and (F or V) and vvs[a].any() and (F or V):
                if any(vvs[a, b] for b in V) or F:
                    V.add(a); changed = True
    return np.array(sorted(F), dtype=int), np.array(sorted(V), dtype=int)


class Amplitude:
    def __init__(self, model: Model, legs: list[Leg], widths: dict[str, float] | None = None) -> None:
        self.m = model
        self.legs = legs
        self.n = len(legs)
        self.fsel, self.vsel = reachable(model, legs)
        self.mF = model.masses_fstate
        self.mV = model.masses_vector
        self.wV = np.zeros(len(model.vectors))
        self.wF = np.zeros(model.n_fstates)
        self.wH = 0.0
        for name, w in (widths or {}).items():
            if name in ("W",):
                for i in model.w_pair:
                    self.wV[i] = w
            elif name == "Z":
                self.wV[model.vector("Z").index] = w
            elif name == "H":
                self.wH = w
            else:
                f = model.fermion(name)
                self.wF[f.index:f.index + f.colours] = w
        self.mH = model.higgs_mass
        self._fermions = [i for i, l in enumerate(legs) if l.is_fermion]
        fs, vs = self.fsel, self.vsel
        self.mF, self.wF = self.mF[fs], self.wF[fs]
        self.mV, self.wV = self.mV[vs], self.wV[vs]
        # vertex tensors restricted to the reachable fields
        self.L = model.ffv_L[np.ix_(vs, fs, fs)]
        self.R = model.ffv_R[np.ix_(vs, fs, fs)]
        self.Y = model.ffs[np.ix_(fs, fs)]
        self.vvv = model.vvv[np.ix_(vs, vs, vs)]
        self.q4 = np.einsum("eab,ecd->abcd", self.vvv, self.vvv)
        self.vvs2 = 2 * model.vvs[np.ix_(vs, vs)]
        self.vvss4 = 4 * model.vvss[np.ix_(vs, vs)]
        self.sss6 = 6 * model.sss
        self.ssss24 = 24 * model.ssss

    # --------------------------------------------------------------- batch
    def _batch(self, momenta):
        per_leg = [external_states(self.m, leg, p) for leg, p in zip(self.legs, momenta)]
        combos = list(itertools.product(*[range(len(x)) for x in per_leg]))
        B = len(combos)
        ext = []
        for i, states in enumerate(per_leg):
            typ = states[0][0]
            if typ == "scalar":
                arr = np.ones(B, complex)
            else:
                arr = np.stack([states[c[i]][1] for c in combos])
                arr = arr[:, self.vsel] if typ == "vec" else arr[:, self.fsel]
            ext.append((typ, arr))
        return ext, B

    # --------------------------------------------------------------- propagators
    def _prop(self, typ, J, P):
        P2 = mdot(P, P)
        if typ == "psi":
            num = slash(P)[None, None] + self.mF[None, :, None, None] * np.eye(4)[None, None]
            den = P2 - self.mF ** 2 + 1j * self.mF * self.wF
            return 1j * np.einsum("fab,Bfb->Bfa", num[0], J) / den[None, :, None]
        if typ == "psibar":
            num = -slash(P)[None] + self.mF[:, None, None] * np.eye(4)[None]
            den = P2 - self.mF ** 2 + 1j * self.mF * self.wF
            return 1j * np.einsum("Bfa,fab->Bfb", J, num) / den[None, :, None]
        if typ == "vec":
            m2 = self.mV ** 2
            den = P2 - m2 + 1j * self.mV * self.wV
            # J carries a lower index here; raise and apply −i(g − PP/m²)/den (unitary gauge if massive)
            Jup = J @ METRIC
            PJ = np.einsum("m,Bam->Ba", P @ METRIC, Jup)
            massive = m2 > 0
            corr = np.where(massive, 1.0 / np.where(massive, m2, 1.0), 0.0)
            out = Jup - corr[None, :, None] * PJ[:, :, None] * P[None, None, :]
            return -1j * out / den[None, :, None]
        den = P2 - self.mH ** 2 + 1j * self.mH * self.wH
        return 1j * J / den

    # --------------------------------------------------------------- vertices → unpropagated currents
    def _join2(self, J1, J2, P1, P2):
        """All currents made by joining two sub-currents (dicts type → array). Lower index for vec."""
        out = {}

        def add(t, x):
            out[t] = out[t] + x if t in out else x

        g_low = METRIC
        if "psibar" in J1 and "psi" in J2:
            self._ffx(J1["psibar"], J2["psi"], add)
        if "psibar" in J2 and "psi" in J1:
            self._ffx(J2["psibar"], J1["psi"], add)
        for a, b in (("vec", "psi"), ("psi", "vec")):
            if a in J1 and b in J2:
                V, psi = (J1[a], J2[b]) if a == "vec" else (J2["vec"], J1["psi"])
                add("psi", 1j * self._v_on_psi(V, psi))
        for a, b in (("vec", "psibar"), ("psibar", "vec")):
            if a in J1 and b in J2:
                V, pb = (J1["vec"], J2["psibar"]) if a == "vec" else (J2["vec"], J1["psibar"])
                add("psibar", 1j * self._psibar_on_v(pb, V))
        for a, b in (("scalar", "psi"), ("psi", "scalar")):
            if a in J1 and b in J2:
                S, psi = (J1["scalar"], J2["psi"]) if a == "scalar" else (J2["scalar"], J1["psi"])
                add("psi", 1j * S[:, None, None] * np.einsum("ij,Bja->Bia", self.Y, psi))
        for a, b in (("scalar", "psibar"), ("psibar", "scalar")):
            if a in J1 and b in J2:
                S, pb = (J1["scalar"], J2["psibar"]) if a == "scalar" else (J2["scalar"], J1["psibar"])
                add("psibar", 1j * S[:, None, None] * np.einsum("Bia,ij->Bja", pb, self.Y))
        if "vec" in J1 and "vec" in J2:
            add("vec", self._vvv(J1["vec"], J2["vec"], P1, P2))
            # h from V V:  L = vvs h V·V  →  i·2·vvs g^{μν}
            add("scalar", 1j * np.einsum("ab,Bam,Bbm->B", self.vvs2, J1["vec"] @ g_low, J2["vec"]))
        for a, b in (("vec", "scalar"), ("scalar", "vec")):
            if a in J1 and b in J2:
                V, S = (J1["vec"], J2["scalar"]) if a == "vec" else (J2["vec"], J1["scalar"])
                add("vec", 1j * np.einsum("ab,Bbm->Bam", self.vvs2, V @ g_low) * S[:, None, None])
        if "scalar" in J1 and "scalar" in J2:
            add("scalar", 1j * self.sss6 * J1["scalar"] * J2["scalar"])
        return out

    def _join3(self, J1, J2, J3):
        out = {}
        if "vec" in J1 and "vec" in J2 and "vec" in J3:
            out["vec"] = self._vvvv(J1["vec"], J2["vec"], J3["vec"])
        # V V S S
        combos = [(J1, J2, J3), (J2, J3, J1), (J3, J1, J2)]
        for A, Bc, C in combos:
            if "vec" in A and "scalar" in Bc and "scalar" in C:
                x = 1j * np.einsum("ab,Bbm->Bam", self.vvss4, A["vec"] @ METRIC) * (Bc["scalar"] * C["scalar"])[:, None, None]
                out["vec"] = out.get("vec", 0) + x
            if "vec" in A and "vec" in Bc and "scalar" in C:
                x = 1j * np.einsum("ab,Bam,Bbm->B", self.vvss4, A["vec"] @ METRIC, Bc["vec"]) * C["scalar"]
                out["scalar"] = out.get("scalar", 0) + x
        if "scalar" in J1 and "scalar" in J2 and "scalar" in J3:
            out["scalar"] = out.get("scalar", 0) + 1j * self.ssss24 * J1["scalar"] * J2["scalar"] * J3["scalar"]
        return out

    def _ffx(self, pb, psi, add):
        """ψ̄ Γ ψ: a vector current (FFV) and a scalar current (FFS)."""
        XL = np.einsum("Bia,mab,Bjb->Bmij", pb, GL, psi)
        XR = np.einsum("Bia,mab,Bjb->Bmij", pb, GR, psi)
        Jv = np.einsum("vij,Bmij->Bvm", self.L, XL) + np.einsum("vij,Bmij->Bvm", self.R, XR)
        add("vec", 1j * (Jv @ METRIC))               # vertex outputs carry a lower index
        add("scalar", 1j * np.einsum("ij,Bia,Bja->B", self.Y, pb, psi))

    def _v_on_psi(self, V, psi):
        Vl = V @ METRIC
        tL = np.einsum("vij,Bvm,mab,Bjb->Bia", self.L, Vl, GL, psi)
        tR = np.einsum("vij,Bvm,mab,Bjb->Bia", self.R, Vl, GR, psi)
        return tL + tR

    def _psibar_on_v(self, pb, V):
        Vl = V @ METRIC
        tL = np.einsum("Bia,vij,Bvm,mab->Bjb", pb, self.L, Vl, GL)
        tR = np.einsum("Bia,vij,Bvm,mab->Bjb", pb, self.R, Vl, GR)
        return tL + tR

    def _vvv(self, A, Bv, k1, k2):
        """Triple-gauge vertex; A, Bv upper index; returns lower-index current for the third leg."""
        k3 = -(k1 + k2)
        g = METRIC
        Al, Bl = A @ g, Bv @ g
        k1l, k2l, k3l = k1 @ g, k2 @ g, k3 @ g
        AB = np.einsum("Bam,Bbm->Bab", Al, Bv)                        # A·B
        # vertex (c,ρ ← k3), (a,μ ← k1), (b,ν ← k2):
        # V_{μνρ} = g_{μν}(k1−k2)_ρ + g_{νρ}(k2−k3)_μ + g_{ρμ}(k3−k1)_ν
        A_k2k3 = np.einsum("Bam,m->Ba", A, (k2l - k3l))                 # A^μ (k2−k3)_μ
        B_k3k1 = np.einsum("Bbm,m->Bb", Bv, (k3l - k1l))
        t1 = AB[:, :, :, None] * (k1l - k2l)[None, None, None, :]
        t2 = A_k2k3[:, :, None, None] * Bl[:, None, :, :]
        t3 = B_k3k1[:, None, :, None] * Al[:, :, None, :]
        T = t1 + t2 + t3                                                  # (B, a, b, ρ)
        # The rule is  vvv_abc [g^{μν}(k1−k2)^ρ + ...]  with no factor of i: the i from the
        # path integral cancels the −i from ∂ → −ik (Peskin & Schroeder fig. 16.7).
        return np.einsum("cab,Babr->Bcr", self.vvv, T)

    def _vvvv(self, A, Bv, C):
        """Quartic gauge vertex from −¼ Σ_e (f_eab A^a·A^b)(f_ecd A^c·A^d).

        Rule for legs (a,μ)(b,ν)(c,ρ)(d,σ):
          −i [ f^{abe}f^{cde}(g^{μρ}g^{νσ} − g^{μσ}g^{νρ}) + f^{ace}f^{bde}(g^{μν}g^{ρσ} − g^{μσ}g^{νρ})
               + f^{ade}f^{bce}(g^{μν}g^{ρσ} − g^{μρ}g^{νσ}) ]
        contracted with A^a_μ B^b_ν C^c_ρ; returns the free leg (d, σ) with a lower index.
        """
        g = METRIC
        Al, Bl, Cl = A @ g, Bv @ g, C @ g
        AB = np.einsum("Bam,Bbm->Bab", Al, Bv)
        AC = np.einsum("Bam,Bcm->Bac", Al, C)
        BC = np.einsum("Bbm,Bcm->Bbc", Bl, C)
        q = self.q4                                   # q[a,b,c,d] = f^{abe} f^{cde}
        t = (np.einsum("abcd,Bac,Bbs->Bds", q, AC, Bl) - np.einsum("abcd,Bbc,Bas->Bds", q, BC, Al)
             + np.einsum("acbd,Bab,Bcs->Bds", q, AB, Cl) - np.einsum("acbd,Bbc,Bas->Bds", q, BC, Al)
             + np.einsum("adbc,Bab,Bcs->Bds", q, AB, Cl) - np.einsum("adbc,Bac,Bbs->Bds", q, AC, Bl))
        return -1j * t

    # --------------------------------------------------------------- evaluate
    def evaluate(self, momenta_in_out: list[np.ndarray]) -> np.ndarray:
        """Amplitudes for all spin/colour combinations. Momenta are physical (E > 0)."""
        P = [p if leg.incoming else -p for leg, p in zip(self.legs, momenta_in_out)]
        ext, B = self._batch(momenta_in_out)
        n = self.n
        last = n - 1
        cur: dict[frozenset, dict] = {}
        mom: dict[frozenset, np.ndarray] = {}
        for i in range(last):
            cur[frozenset([i])] = {ext[i][0]: ext[i][1]}
            mom[frozenset([i])] = P[i]
        full = None
        for size in range(2, n):
            for S in itertools.combinations(range(last), size):
                S = frozenset(S)
                acc: dict = {}
                elems = sorted(S)
                first = elems[0]
                rest = elems[1:]
                # unordered 2-splits: first element always in S1
                for r in range(0, len(rest)):
                    for extra in itertools.combinations(rest, r):
                        S1 = frozenset((first,) + extra)
                        S2 = S - S1
                        if not S2 or S1 not in cur or S2 not in cur:
                            continue
                        sign = self._sign(S1, S2)
                        for t, x in self._join2(cur[S1], cur[S2], mom[S1], mom[S2]).items():
                            acc[t] = acc.get(t, 0) + sign * x
                # unordered 3-splits
                if len(S) >= 3:
                    for S1, S2, S3 in _three_splits(elems):
                        if S1 in cur and S2 in cur and S3 in cur:
                            sign = self._sign3(S1, S2, S3)
                            for t, x in self._join3(cur[S1], cur[S2], cur[S3]).items():
                                acc[t] = acc.get(t, 0) + sign * x
                acc = {t: x for t, x in acc.items() if np.any(np.abs(x) > 1e-300)}
                if size == n - 1:
                    full = acc
                    continue
                PS = sum(P[i] for i in S)
                mom[S] = PS
                cur[S] = {t: self._prop(t, x, PS) for t, x in acc.items()}
                if not cur[S]:
                    del cur[S]
        if n == 2:
            full = cur[frozenset([0])]
        return self._close(full or {}, ext[last], B)

    def _close(self, full, ext_last, B):
        typ, w = ext_last
        amp = np.zeros(B, complex)
        if typ == "psibar" and "psi" in full:
            amp += np.einsum("Bia,Bia->B", w, full["psi"])
        elif typ == "psi" and "psibar" in full:
            amp += np.einsum("Bia,Bia->B", full["psibar"], w)
        elif typ == "vec" and "vec" in full:
            amp += np.einsum("Bam,Bam->B", full["vec"], w)       # lower (vertex) · upper (ε)
        elif typ == "scalar" and "scalar" in full:
            amp += full["scalar"]
        return amp

    def _sign(self, S1, S2):
        f1 = [i for i in sorted(S1) if i in self._fermions]
        f2 = [i for i in sorted(S2) if i in self._fermions]
        return _perm_sign(f1 + f2)

    def _sign3(self, S1, S2, S3):
        fs = [[i for i in sorted(S) if i in self._fermions] for S in (S1, S2, S3)]
        return _perm_sign(fs[0] + fs[1] + fs[2])


def _three_splits(elems):
    """Unordered partitions of ``elems`` into three non-empty blocks, each yielded once."""
    elems = list(elems)
    seen = set()
    for labels in itertools.product(range(3), repeat=len(elems)):
        if len(set(labels)) < 3:
            continue
        parts = [frozenset(e for e, l in zip(elems, labels) if l == k) for k in range(3)]
        key = frozenset(parts)
        if key in seen:
            continue
        seen.add(key)
        parts.sort(key=min)
        yield parts[0], parts[1], parts[2]
