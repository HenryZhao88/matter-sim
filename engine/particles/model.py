"""The Standard Model, built from its Lagrangian's structure.

Input (written here): the gauge group SU(3)×SU(2)×U(1) and its couplings, how each field
transforms (colour, weak isospin, hypercharge), the Higgs potential, and the Yukawa
couplings (fixed by measured fermion masses and the CKM matrix). The couplings g, g', v
are fixed by three measurements (α, G_F, m_Z).

Output (computed here, not written anywhere): which vector bosons exist after the Higgs
field takes its vacuum value and what their masses are, a massless photon, every
particle's electric charge (the photon's coupling), the W/Z couplings of each fermion,
the gauge self-couplings (from the Lie algebra), and the Higgs couplings.

Everything is kept in a *real* field basis: the 8 gluons, the electroweak mass
eigenstates (two degenerate charged-sector fields "W1", "W2", then Z and γ, whatever
order the diagonalisation returns them in), one Higgs scalar, and 12 Dirac fermions ×
colour. External W± states are complex combinations of the two degenerate real fields.
Natural units, GeV.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ------------------------------------------------------------------ measured inputs (PDG 2024)
ALPHA_0 = 1 / 137.035999177    # electromagnetic coupling at zero momentum transfer (CODATA 2022)
ALPHA_MZ = 1 / 127.951          # electromagnetic coupling at the Z mass
G_FERMI = 1.1663788e-5          # GeV⁻² (muon decay)
M_Z_INPUT = 91.1876             # GeV
ALPHA_S = 0.1180                # strong coupling at the Z mass
M_HIGGS = 125.25
FERMION_MASS = {                 # GeV
    "u": 2.16e-3, "c": 1.27, "t": 172.69,
    "d": 4.67e-3, "s": 0.0934, "b": 4.18,
    "e": 0.51099895e-3, "mu": 0.1056583755, "tau": 1.77686,
    "nu_e": 0.0, "nu_mu": 0.0, "nu_tau": 0.0,
}
# CKM (Wolfenstein parameters, PDG) — the misalignment of up- and down-type Yukawa matrices.
_LAM, _A, _RHOB, _ETAB = 0.22500, 0.826, 0.159, 0.348


def _ckm() -> np.ndarray:
    lam, A = _LAM, _A
    rho_eta = complex(_RHOB, _ETAB) * math.sqrt(1 - A * A * lam ** 4) / (
        math.sqrt(1 - lam * lam) * (1 - A * A * lam ** 4 * complex(_RHOB, _ETAB)))
    s12, s23 = lam, A * lam * lam
    s13e = A * lam ** 3 * rho_eta.conjugate()          # s13 e^{-iδ}
    s13 = abs(s13e)
    d = -np.angle(s13e)
    c12, c23, c13 = (math.sqrt(1 - x * x) for x in (s12, s23, s13))
    e = np.exp(1j * d)
    return np.array([
        [c12 * c13, s12 * c13, s13 / e],
        [-s12 * c23 - c12 * s23 * s13 * e, c12 * c23 - s12 * s23 * s13 * e, s23 * c13],
        [s12 * s23 - c12 * c23 * s13 * e, -c12 * s23 - s12 * c23 * s13 * e, c23 * c13],
    ])


# ------------------------------------------------------------------ Lagrangian structure
# Per generation: (name, SU(3) rep dim, SU(2) rep dim, hypercharge Y, chirality, components)
MULTIPLETS = [
    ("Q", 3, 2, 1 / 6, "L", ("u", "d")),
    ("u_R", 3, 1, 2 / 3, "R", ("u",)),
    ("d_R", 3, 1, -1 / 3, "R", ("d",)),
    ("L", 1, 2, -1 / 2, "L", ("nu", "e")),
    ("e_R", 1, 1, -1, "R", ("e",)),
]
HIGGS_Y = 1 / 2                  # the Higgs is an SU(2) doublet with hypercharge ½
GENERATIONS = [("u", "d", "nu_e", "e"), ("c", "s", "nu_mu", "mu"), ("t", "b", "nu_tau", "tau")]

PAULI = [np.array([[0, 1], [1, 0]], complex), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]], complex)]


def gell_mann() -> list[np.ndarray]:
    l = [np.zeros((3, 3), complex) for _ in range(8)]
    l[0][0, 1] = l[0][1, 0] = 1
    l[1][0, 1], l[1][1, 0] = -1j, 1j
    l[2][0, 0], l[2][1, 1] = 1, -1
    l[3][0, 2] = l[3][2, 0] = 1
    l[4][0, 2], l[4][2, 0] = -1j, 1j
    l[5][1, 2] = l[5][2, 1] = 1
    l[6][1, 2], l[6][2, 1] = -1j, 1j
    l[7][0, 0] = l[7][1, 1] = 1 / math.sqrt(3)
    l[7][2, 2] = -2 / math.sqrt(3)
    return l


def structure_constants(gens: list[np.ndarray]) -> np.ndarray:
    """f_abc from [T_a, T_b] = i f_abc T_c, with Tr(T_a T_b) = ½ δ_ab."""
    n = len(gens)
    f = np.zeros((n, n, n))
    for a in range(n):
        for b in range(n):
            comm = gens[a] @ gens[b] - gens[b] @ gens[a]
            for c in range(n):
                f[a, b, c] = np.real(-2j * np.trace(comm @ gens[c]))
    return f


# ------------------------------------------------------------------ the built model
@dataclass
class Fermion:
    name: str
    mass: float
    charge: float        # computed: photon coupling / e
    colours: int         # 3 for quarks, 1 for leptons
    index: int           # first slot in the fermion state space


@dataclass
class Vector:
    name: str
    mass: float
    kind: str            # "gluon" or "electroweak"
    index: int           # slot in the vector species space


@dataclass
class Model:
    e: float
    g: float
    gp: float
    gs: float
    v: float
    sin2w: float
    fermions: list[Fermion]
    vectors: list[Vector]
    higgs_mass: float
    n_fstates: int                       # fermion flavour × colour states
    ffv_L: np.ndarray                    # [vector, fbar_state, f_state]
    ffv_R: np.ndarray
    ffs: np.ndarray                      # [fbar_state, f_state] (Higgs Yukawa, both chiralities)
    vvv: np.ndarray                      # [a, b, c] real, totally antisymmetric
    vvs: np.ndarray                      # [a, b]  coefficient of h V_a V_b in L
    vvss: np.ndarray                     # [a, b]  coefficient of h h V_a V_b in L
    sss: float                           # coefficient of h³ in L
    ssss: float                          # coefficient of h⁴ in L
    w_pair: tuple[int, int]              # the two degenerate charged-sector fields
    notes: dict = field(default_factory=dict)

    def fermion(self, name: str) -> Fermion:
        for f in self.fermions:
            if f.name == name:
                return f
        raise KeyError(name)

    def vector(self, name: str) -> Vector:
        for v in self.vectors:
            if v.name == name:
                return v
        raise KeyError(name)

    @property
    def masses_fstate(self) -> np.ndarray:
        m = np.zeros(self.n_fstates)
        for f in self.fermions:
            m[f.index:f.index + f.colours] = f.mass
        return m

    @property
    def masses_vector(self) -> np.ndarray:
        return np.array([v.mass for v in self.vectors])


def build(alpha_s: float = ALPHA_S) -> Model:
    # --- couplings from three measurements (tree level): e, G_F, m_Z
    e = math.sqrt(4 * math.pi * ALPHA_MZ)
    v = (math.sqrt(2) * G_FERMI) ** -0.5
    gsum2 = (2 * M_Z_INPUT / v) ** 2                   # g² + g'² from m_Z = v√(g²+g'²)/2
    # e = g g'/√(g²+g'²):  g² g'² = e² (g² + g'²)
    prod = e * e * gsum2
    disc = math.sqrt(gsum2 * gsum2 - 4 * prod)
    g = math.sqrt((gsum2 + disc) / 2)
    gp = math.sqrt((gsum2 - disc) / 2)
    gs = math.sqrt(4 * math.pi * alpha_s)

    # --- electroweak gauge-boson masses from |D_μ⟨H⟩|² (real fields W¹, W², W³, B)
    H0 = np.array([0, v / math.sqrt(2)], complex)
    G = [g * PAULI[a] / 2 for a in range(3)] + [gp * HIGGS_Y * np.eye(2)]
    M2 = np.array([[np.real(H0.conj() @ (Gi @ Gj + Gj @ Gi) @ H0) for Gj in G] for Gi in G])
    m2, O = np.linalg.eigh(M2)                       # columns = mass eigenstates
    m2 = np.clip(m2, 0, None)

    # Classify eigenstates by what they do, not by name: the massless one is the photon;
    # the pair that is degenerate and carries charge is the W; the rest is the Z.
    order = np.argsort(m2)
    photon = order[0]
    massive = order[1:]
    # the two degenerate massive eigenvalues form the charged sector
    pairs = [(i, j) for i in massive for j in massive if i < j and abs(m2[i] - m2[j]) < 1e-9 * m2.max()]
    w1, w2 = pairs[0]
    z = [i for i in massive if i not in (w1, w2)][0]

    # couplings of each EW mass eigenstate to a multiplet: C_k = Σ_i O[i,k] G_i(rep)
    def ew_coupling(k, su2_dim, Y):
        T = [np.zeros((1, 1))] * 3 if su2_dim == 1 else [PAULI[a] / 2 for a in range(3)]
        mats = [g * T[a] for a in range(3)] + [gp * Y * np.eye(su2_dim)]
        return sum(O[i, k] * mats[i] for i in range(4))

    # Electric charge = photon coupling / e. Fix the photon field's sign so the electron's is negative
    # (a sign convention for the field, not a physical choice).
    C_e = ew_coupling(photon, 2, -1 / 2)[1, 1] + 0 * 1j
    if np.real(C_e) > 0:
        O[:, photon] *= -1

    # --- fermion state space: flavour × colour
    names = [n for gen in GENERATIONS for n in gen]
    fermions, idx = [], 0
    for gen in GENERATIONS:
        for name in gen:
            col = 3 if name in ("u", "c", "t", "d", "s", "b") else 1
            fermions.append(Fermion(name, FERMION_MASS[name], 0.0, col, idx))
            idx += col
    nF = idx
    fidx = {f.name: f for f in fermions}

    vectors = [Vector(f"g{a + 1}", 0.0, "gluon", a) for a in range(8)]
    ew_slots = {}
    for label, k in (("W1", w1), ("W2", w2), ("Z", z), ("photon", photon)):
        ew_slots[k] = len(vectors)
        vectors.append(Vector(label, math.sqrt(m2[k]), "electroweak", len(vectors)))
    nV = len(vectors)

    L = np.zeros((nV, nF, nF), complex)
    R = np.zeros((nV, nF, nF), complex)
    lam = gell_mann()
    V_ckm = _ckm()

    def put(mat, vec, fa, fb, value):
        """Coupling between flavours fa (bar) and fb, diagonal in colour."""
        A, B = fidx[fa], fidx[fb]
        for c in range(A.colours):
            mat[vec, A.index + c, B.index + c] += value

    # gluons: g_s T^a on every quark flavour, both chiralities
    for a in range(8):
        for q in ("u", "c", "t", "d", "s", "b"):
            Q = fidx[q]
            blk = gs * lam[a] / 2
            L[a, Q.index:Q.index + 3, Q.index:Q.index + 3] += blk
            R[a, Q.index:Q.index + 3, Q.index:Q.index + 3] += blk

    # electroweak: read couplings off each multiplet
    ups, downs = ["u", "c", "t"], ["d", "s", "b"]
    for k, slot in ew_slots.items():
        for gi, gen in enumerate(GENERATIONS):
            up, down, nu, lep = gen
            for (mname, _c3, su2, Y, chir, comps) in MULTIPLETS:
                C = ew_coupling(k, su2, Y)
                target = L if chir == "L" else R
                flav = {"u": up, "d": down, "nu": nu, "e": lep}
                if su2 == 1:
                    put(target, slot, flav[comps[0]], flav[comps[0]], C[0, 0])
                    continue
                top, bot = flav[comps[0]], flav[comps[1]]
                put(target, slot, top, top, C[0, 0])
                put(target, slot, bot, bot, C[1, 1])
                # off-diagonal (charged-current) pieces; quarks mix through the CKM matrix
                if comps == ("u", "d"):
                    for j, dj in enumerate(downs):
                        put(target, slot, top, dj, C[0, 1] * V_ckm[gi, j])
                        put(target, slot, dj, top, C[1, 0] * np.conj(V_ckm[gi, j]))
                else:
                    put(target, slot, top, bot, C[0, 1])
                    put(target, slot, bot, top, C[1, 0])

    # electric charges (computed: the photon's coupling in units of e)
    ph = ew_slots[photon]
    for f in fermions:
        f.charge = float(np.real(L[ph, f.index, f.index]) / e)

    # Higgs Yukawa: mass terms m ψ̄ψ come from y ψ̄ H ψ → coefficient of h ψ̄ψ is −m/v
    ffs = np.zeros((nF, nF), complex)
    for f in fermions:
        for c in range(f.colours):
            ffs[f.index + c, f.index + c] = -f.mass / v

    # gauge self-couplings from the Lie algebra, rotated to the mass basis
    f3 = structure_constants([l / 2 for l in lam])
    vvv = np.zeros((nV, nV, nV))
    vvv[:8, :8, :8] = gs * f3
    eps = np.zeros((4, 4, 4))
    for (i, j, kk) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        eps[i, j, kk], eps[j, i, kk] = 1.0, -1.0
    eps *= g
    ew = np.einsum("ijk,ia,jb,kc->abc", eps, O, O, O)
    for a_k, a_s in ew_slots.items():
        for b_k, b_s in ew_slots.items():
            for c_k, c_s in ew_slots.items():
                vvv[a_s, b_s, c_s] = ew[a_k, b_k, c_k]

    # Higgs–vector couplings: ½ M²(v) V V with v → v + h
    vvs = np.zeros((nV, nV))
    vvss = np.zeros((nV, nV))
    M2mass = O.T @ M2 @ O
    for a_k, a_s in ew_slots.items():
        for b_k, b_s in ew_slots.items():
            vvs[a_s, b_s] = M2mass[a_k, b_k] / v            # ½M²·2h/v
            vvss[a_s, b_s] = M2mass[a_k, b_k] / (2 * v * v)  # ½M²·h²/v²
    lam_h = M_HIGGS ** 2 / (2 * v * v)
    return Model(
        e=e, g=g, gp=gp, gs=gs, v=v, sin2w=gp * gp / (g * g + gp * gp),
        fermions=fermions, vectors=vectors, higgs_mass=M_HIGGS, n_fstates=nF,
        ffv_L=L, ffv_R=R, ffs=ffs, vvv=vvv, vvs=vvs, vvss=vvss,
        sss=-lam_h * v, ssss=-lam_h / 4,
        w_pair=(ew_slots[w1], ew_slots[w2]),
        notes={"ckm": V_ckm, "m2_eigen": m2},
    )
