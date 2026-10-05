"""One-loop quantum corrections, from the same Feynman rules as the tree-level amplitudes.

Two loops of QED, each evaluated the way the tree amplitudes are: explicit 4×4 Dirac matrices and
numbers, no algebra done by hand and no loop result written in.

* The vertex correction: a photon exchanged across the lepton–photon vertex. Its magnetic form
  factor F₂ at zero momentum transfer is the lepton's anomalous magnetic moment a = (g − 2)/2.
  F₂ is finite (no ultraviolet or infrared regulator enters it).
* The vacuum polarisation: a fermion loop in the photon propagator, which screens charge and
  makes α grow with the energy. Renormalised at q² = 0 (so α(0) is the Thomson-limit coupling),
  it is finite too.

Method, for both. The loop integrand's denominators are combined with Feynman parameters, the
loop momentum is shifted so the combined denominator is (l² − Δ)ⁿ, and the l-independent part of
the numerator is evaluated numerically, as a Dirac matrix (vertex) or a Dirac trace (vacuum
polarisation). Terms odd in l vanish; terms in l^α l^β are proportional to g^{αβ}, so they feed
only the γ^μ (vertex) and g^{μν} (vacuum polarisation) structures, never the ones read off here.
The remaining l-integrals are the standard Wick-rotated ones,
    ∫ d⁴l/(2π)⁴ (l² − Δ)⁻³ = −i / (32π² Δ),
    ∫ d⁴l/(2π)⁴ [(l² − Δ)⁻² − (l² − Δ₀)⁻²] = (i / 16π²) ln(Δ₀/Δ),
and the Feynman parameters are integrated numerically.

Conventions as amplitudes.py: metric (+,−,−,−), Weyl basis, vertex −i e Q γ^μ, fermion
propagator i(k̸ + m)/(k² − m²), photon propagator −i g_{μν}/k² (Feynman gauge).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

from .amplitudes import GAMMA, METRIC, mdot, slash, spinors
from .model import ALPHA_0

GAMMA_LOW = np.einsum("nm,mab->nab", METRIC, GAMMA)          # γ_μ (lower index)


# ------------------------------------------------------------------ the vertex: F₂(0) = a
def _vertex_numerator(k, kp, m):
    """γ^ν (k̸' + m) γ^μ (k̸ + m) γ_ν for each μ: shape (points, 4, 4, 4) [point, μ, a, b]."""
    one = np.eye(4)
    S, Sp = slash(k) + m * one, slash(kp) + m * one
    return np.einsum("nac,pcd,mde,pef,nfb->pmab", GAMMA, Sp, GAMMA, S, GAMMA_LOW, optimize=True)


def _feynman_simplex(n: int):
    """Gauss–Legendre points and weights on {x + y + z = 1}, as (x, y, z, w) with ∫dF 1 = ½."""
    t, w = np.polynomial.legendre.leggauss(n)
    t, w = 0.5 * (t + 1), 0.5 * w
    z = np.repeat(t, n)
    wz = np.repeat(w, n)
    s = np.tile(t, n)
    ws = np.tile(w, n)
    x = (1 - z) * s
    return x, (1 - z) - x, z, wz * ws * (1 - z)


def vertex_form_factors(m: float, q_over_m: float, alpha: float = ALPHA_0, n: int = 96) -> dict:
    """The one-loop QED vertex correction for a lepton of mass m at momentum transfer q² = −(q_over_m·m)².

    Breit frame: p = (E, 0, 0, −k), p' = (E, 0, 0, k), q = p' − p. The correction δΓ^μ is written
    A γ^μ + B (p + p')^μ + C q^μ, found by fitting ū(p', s') δΓ^μ u(p, s) over all spins and μ;
    the Gordon identity then gives F₂ = −2m B. Only B and C are finite: A carries the ultraviolet
    and infrared divergences of F₁ and is not returned."""
    k = 0.5 * q_over_m * m
    E = math.sqrt(m * m + k * k)
    p, pp = np.array([E, 0, 0, -k]), np.array([E, 0, 0, k])
    q = pp - p
    x, y, z, w = _feynman_simplex(n)
    # shift of the loop momentum: k = l − y q + z p, k' = k + q  (x ↔ k, y ↔ k', z ↔ the photon)
    k0 = z[:, None] * p[None] - y[:, None] * q[None]
    kp0 = k0 + q[None]
    delta = -x * y * mdot(q, q) + (1 - z) ** 2 * m * m
    N = _vertex_numerator(k0, kp0, m)                                   # (points, μ, 4, 4)
    # −i e² × 2 (Feynman) × (−i/(32π² Δ)) = −e²/(16π² Δ), and ∫dF over the simplex
    e2 = 4 * math.pi * alpha
    dG = -e2 / (16 * math.pi ** 2) * np.einsum("p,pmab->mab", w / delta, N)
    u, _ = spinors(p)
    up, _ = spinors(pp)
    ubar = [x.conj() @ GAMMA[0] for x in up]
    rows, basis = [], []
    for a in range(2):
        for b in range(2):
            for mu in range(4):
                rows.append(ubar[a] @ dG[mu] @ u[b])
                basis.append([ubar[a] @ GAMMA[mu] @ u[b], (p + pp)[mu] * (ubar[a] @ u[b]), q[mu] * (ubar[a] @ u[b])])
    coef, *_ = np.linalg.lstsq(np.array(basis), np.array(rows), rcond=None)
    A, B, C = coef
    return {"F2": float(np.real(-2 * m * B)), "F2_imag": float(np.imag(-2 * m * B)), "C": complex(C),
            "q2": float(mdot(q, q))}


def anomalous_moment(m: float, alpha: float = ALPHA_0, n: int = 96) -> float:
    """a = F₂(0), by Richardson extrapolation in q² from two small momentum transfers."""
    q1, q2 = 1e-2, 2e-2
    f1 = vertex_form_factors(m, q1, alpha, n)["F2"]
    f2 = vertex_form_factors(m, q2, alpha, n)["F2"]
    return f1 - (f2 - f1) * q1 ** 2 / (q2 ** 2 - q1 ** 2)


# ------------------------------------------------------------------ vacuum polarisation: α(q²)
def _pi_coefficient(x: float, m: float) -> float:
    """The q^μ q^ν coefficient of tr[γ^μ (k̸₀ + m) γ^ν (k̸₀ + q̸ + m)] at the shifted k₀ = −x q, per q^μ q^ν."""
    q = np.array([0.3, 0.0, 0.0, 0.7])                    # any q with q⁰q³ ≠ 0 (g^{03} = 0 isolates q^μ q^ν)
    k0 = -x * q
    T = np.einsum("mab,bc,ncd,da->mn", GAMMA, slash(k0) + m * np.eye(4), GAMMA, slash(k0 + q) + m * np.eye(4))
    return float(np.real(T[0, 3]) / (q[0] * q[3]))


def vacuum_polarisation(q2: float, m: float, charge: float, colours: int, alpha: float = ALPHA_0) -> complex:
    """Renormalised one-loop Π̂(q²) = Π(q²) − Π(0) of one fermion (charge in units of e).

    From iΠ^{μν} = −(−ieQ)² N_c ∫ d⁴k/(2π)⁴ tr[γ^μ i(k̸+m) γ^ν i(k̸+q̸+m)] / (denominators), with
    Π^{μν} = (q² g^{μν} − q^μ q^ν) Π: the q^μ q^ν coefficient, with Δ = m² − x(1−x)q², gives
    Π̂(q²) = (e² Q² N_c / 16π²) ∫ dx c(x) ln(m²/Δ), c(x) the numerically evaluated trace coefficient.
    Above threshold (Δ < 0) the logarithm gains +iπ (Δ carries −iε): the imaginary part is pair creation."""
    e2 = 4 * math.pi * alpha
    pref = e2 * charge * charge * colours / (16 * math.pi ** 2)
    if m == 0:
        raise ValueError("massless fermion: Π̂ is infrared-divergent at q² = 0")
    roots = []
    if q2 > 4 * m * m:
        r = math.sqrt(1 - 4 * m * m / q2)
        roots = [0.5 * (1 - r), 0.5 * (1 + r)]

    def re(x):
        d = m * m - x * (1 - x) * q2
        return _pi_coefficient(x, m) * math.log(m * m / abs(d))

    def im(x):
        d = m * m - x * (1 - x) * q2
        return _pi_coefficient(x, m) * (math.pi if d < 0 else 0.0)       # ln(m²/(Δ − iε)), Δ < 0

    real = quad(re, 0, 1, points=roots or None, limit=400, epsabs=0, epsrel=1e-10)[0]
    imag = quad(im, roots[0], roots[1], limit=200)[0] if roots else 0.0
    return pref * complex(real, imag)


def delta_alpha(q2: float, model, alpha: float = ALPHA_0, which=None) -> dict:
    """Δα(q²) = Re Π̂(q²), fermion by fermion, with the model's own charges, colours and masses.

    The photon propagator −ig/(q²(1 − Π̂)) turns e² into e²/(1 − Π̂), so α(q²) = α(0)/(1 − Δα(q²));
    Π̂ grows with |q²| (the loop screens charge less at shorter distance), and so does α. Light quarks are included at their Lagrangian masses; that is
    perturbation theory where it does not apply (below ~2 GeV quarks are bound into hadrons), and
    the result says so by disagreeing with the measured α(m_Z)."""
    out = {}
    for f in model.fermions:
        if abs(f.charge) < 1e-6 or (which is not None and f.name not in which):
            continue
        out[f.name] = vacuum_polarisation(q2, f.mass, f.charge, f.colours, alpha).real
    return out


# ------------------------------------------------------------------ W and Z self-energies: Δρ and the W mass
def _x_points(n: int = 96):
    t, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (t + 1), 0.5 * w


def vector_self_energy_0(model, vector: str, mu: float, n: int = 96) -> tuple[float, float]:
    """Transverse self-energy Π_T(0) of a weak vector boson from every fermion loop, in
    dimensional regularisation (d = 4 − 2ε, MS-bar scale ``mu``): returns (pole, finite) with
    Π_T(0) = pole/ε̄ + finite. Couplings, CKM mixing, colours and masses are the model's own.

    With L ⊃ V_μ ψ̄_i γ^μ (L_ij P_L + R_ij P_R) ψ_j, the loop with vertices (ij) and (ji) is
        iΠ^{μν} = −∫dx ∫ d^dl/(2π)^d tr[Γ^μ_ij (l̸ + m_j) Γ^ν_ji (l̸ + m_i)] / (l² − Δ)²,
    Δ = x m_i² + (1−x) m_j². The only d-dependent Dirac algebra is the contraction left by
    l^α l^β → l² g^{αβ}/d: γ^α γ^ν (L P_L + R P_R) γ_α = −(d−2) γ^ν (L P_R + R P_L), taken
    analytically (anticommuting γ5); the remaining 4×4 traces are numbers. With the standard
    integrals, −((d−2)/d) ∫ l²/(l²−Δ)² and ∫ 1/(l²−Δ)² combine into
        Π^{μν} = −(1/16π²) Σ ∫dx [1/ε̄ − ln(Δ/μ²)] (m_i m_j tr[Γ^μ Γ^ν] − Δ tr[Γ^μ γ^ν Γ̃^ν]),
    Γ̃ the projector-swapped vertex. Π_T is the g^{μν} part, read from the (1,1) component."""
    v = model.vector(vector).index
    L, R = model.ffv_L[v], model.ffv_R[v]
    masses = model.masses_fstate
    PL, PR = np.diag([1, 1, 0, 0]).astype(complex), np.diag([0, 0, 1, 1]).astype(complex)
    g1 = GAMMA[1]
    x, w = _x_points(n)
    pole = finite = 0.0
    idx = np.argwhere((np.abs(L) > 1e-14) | (np.abs(R) > 1e-14))
    for i, j in idx:
        if abs(L[j, i]) < 1e-14 and abs(R[j, i]) < 1e-14:
            continue
        Gmu = g1 @ (L[i, j] * PL + R[i, j] * PR)
        Gnu = g1 @ (L[j, i] * PL + R[j, i] * PR)
        Gnu_swap = g1 @ (L[j, i] * PR + R[j, i] * PL)
        mi, mj = masses[i], masses[j]
        M = mi * mj * np.trace(Gmu @ Gnu)
        Q = np.trace(Gmu @ Gnu_swap)
        D = x * mi * mi + (1 - x) * mj * mj
        core = M - D * Q                                    # (1,1) component, per x
        logs = np.where(D > 0, np.log(np.where(D > 0, D, 1.0) / (mu * mu)), 0.0)
        pole += -np.sum(w * core).real / (16 * math.pi ** 2)
        finite += np.sum(w * core * logs).real / (16 * math.pi ** 2)
    # Π_T = Π^{11} / g^{11} = −Π^{11}
    return -pole, -finite


def delta_rho(model, mu: float | None = None) -> dict:
    """Δρ = Π_T^W(0)/M_W² − Π_T^Z(0)/M_Z² from fermion loops: how much the W and Z masses' ratio
    departs from its tree-level value. Each self-energy has a 1/ε pole; in Δρ the poles cancel
    (a complete SU(2) doublet), which the result reports, and with it the dependence on μ."""
    mu = mu or model.vector("Z").mass
    MW, MZ = model.vector("W1").mass, model.vector("Z").mass
    pw, fw = vector_self_energy_0(model, "W1", mu)
    pz, fz = vector_self_energy_0(model, "Z", mu)
    return {"delta_rho": fw / MW ** 2 - fz / MZ ** 2, "pole": pw / MW ** 2 - pz / MZ ** 2,
            "pole_W": pw / MW ** 2, "mu": mu}


def w_mass_one_loop(model) -> dict:
    """The W mass with Δρ: from s²c² = πα/(√2 G_F M_Z²) (1 + Δr) with Δr = −(c²/s²) Δρ, the part of
    the one-loop correction that grows with the top mass (α is already α(m_Z) in the model, so the
    running of α is not counted again). The rest of Δr (vertex, box and bosonic loops) is not here."""
    from .model import ALPHA_MZ, G_FERMI, M_Z_INPUT
    MZ = M_Z_INPUT
    A = math.pi * ALPHA_MZ / (math.sqrt(2) * G_FERMI)
    dr = delta_rho(model)["delta_rho"]
    MW = model.vector("W1").mass
    for _ in range(50):                                    # Δr depends on c² = M_W²/M_Z² itself
        c2 = MW * MW / (MZ * MZ)
        Dr = -c2 / (1 - c2) * dr
        MW = math.sqrt(0.5 * MZ * MZ * (1 + math.sqrt(1 - 4 * A * (1 + Dr) / (MZ * MZ))))
    return {"M_W": MW, "M_W_tree": model.vector("W1").mass, "delta_rho": dr, "delta_r": Dr}
