"""Local spin-density exchange–correlation (LSDA).

Exchange is the exact result for the uniform electron gas (Dirac/Slater) —
derived, not fitted. Correlation is the Perdew–Wang 1992 parametrisation of
Ceperley–Alder's quantum Monte Carlo of the uniform electron gas: its
constants encode an exactly-solvable model system, never molecular data.

All arrays are float64 numpy; returns the energy density per volume
(e_xc = ρ ε_xc) and the spin potentials v_xc↑, v_xc↓.
"""

from __future__ import annotations

import numpy as np

RHO_FLOOR = 1e-14
_FZ_DEN = 2 ** (4 / 3) - 2
_FPP0 = 1.709920934161365  # f''(0)

# (A, alpha1, beta1, beta2, beta3, beta4) — Perdew & Wang, PRB 45, 13244 (1992)
_PW_UNPOL = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_PW_POL = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_PW_ALPHA = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)


def _pw_G(rs, params):
    A, a1, b1, b2, b3, b4 = params
    srs = np.sqrt(rs)
    Q = 2 * A * (b1 * srs + b2 * rs + b3 * rs * srs + b4 * rs * rs)
    dQ = 2 * A * (0.5 * b1 / srs + b2 + 1.5 * b3 * srs + 2 * b4 * rs)
    log = np.log1p(1.0 / Q)
    G = -2 * A * (1 + a1 * rs) * log
    dG = -2 * A * a1 * log + 2 * A * (1 + a1 * rs) * dQ / (Q * Q + Q)
    return G, dG


def pw92_correlation(rs, zeta):
    """ε_c(rs, ζ) per electron and its partial derivatives (∂/∂rs, ∂/∂ζ)."""
    ec0, dec0 = _pw_G(rs, _PW_UNPOL)
    ec1, dec1 = _pw_G(rs, _PW_POL)
    mac, dmac = _pw_G(rs, _PW_ALPHA)
    ac, dac = -mac, -dmac

    opz = np.clip(1 + zeta, 0.0, 2.0)
    omz = np.clip(1 - zeta, 0.0, 2.0)
    f = (opz ** (4 / 3) + omz ** (4 / 3) - 2) / _FZ_DEN
    df = (4 / 3) * (np.cbrt(opz) - np.cbrt(omz)) / _FZ_DEN
    z3 = zeta ** 3
    z4 = z3 * zeta

    ec = ec0 + ac * f / _FPP0 * (1 - z4) + (ec1 - ec0) * f * z4
    dec_drs = dec0 * (1 - f * z4) + dec1 * f * z4 + dac * f / _FPP0 * (1 - z4)
    dec_dz = 4 * z3 * f * (ec1 - ec0 - ac / _FPP0) + df * (z4 * (ec1 - ec0) + (1 - z4) * ac / _FPP0)
    return ec, dec_drs, dec_dz


def lda_xc(rho_up: np.ndarray, rho_dn: np.ndarray):
    rho_up = np.maximum(rho_up, 0.0)
    rho_dn = np.maximum(rho_dn, 0.0)
    rho = rho_up + rho_dn
    mask = rho > RHO_FLOOR
    e_xc = np.zeros_like(rho)
    v_up = np.zeros_like(rho)
    v_dn = np.zeros_like(rho)
    if not mask.any():
        return e_xc, v_up, v_dn

    ru, rd, r = rho_up[mask], rho_dn[mask], rho[mask]

    # Exchange: E_x = -(3/4)(6/π)^{1/3} Σσ ρσ^{4/3}
    cx = (6 / np.pi) ** (1 / 3)
    ex = -0.75 * cx * (ru ** (4 / 3) + rd ** (4 / 3))
    vxu = -cx * np.cbrt(ru)
    vxd = -cx * np.cbrt(rd)

    # Correlation
    rs = (3 / (4 * np.pi * r)) ** (1 / 3)
    zeta = np.clip((ru - rd) / r, -1.0, 1.0)
    ec, dec_drs, dec_dz = pw92_correlation(rs, zeta)
    common = ec - rs / 3 * dec_drs
    vcu = common - (zeta - 1) * dec_dz
    vcd = common - (zeta + 1) * dec_dz

    e_xc[mask] = ex + r * ec
    v_up[mask] = vxu + vcu
    v_dn[mask] = vxd + vcd
    return e_xc, v_up, v_dn


# ------------------------------------------------------------------ PBE (generalised gradient)
# Perdew, Burke & Ernzerhof, PRL 77, 3865 (1996). Every constant is fixed by an exact condition,
# none by fitting: μ from the gradient expansion of correlation cancelling that of exchange (μ =
# β π²/3), κ from the Lieb–Oxford bound, β from the high-density limit of the gradient expansion
# of correlation, γ = (1 − ln 2)/π² from the high-density limit of the correlation energy.
PBE_KAPPA = 0.804
PBE_BETA = 0.06672455060314922
PBE_MU = PBE_BETA * np.pi ** 2 / 3
PBE_GAMMA = (1 - np.log(2.0)) / np.pi ** 2
_CX = 0.75 * (3 / np.pi) ** (1 / 3)          # ε_x^unif = −_CX n^{1/3}
_KF = (3 * np.pi ** 2) ** (1 / 3)            # k_F = _KF n^{1/3}


def _pbe_exchange(n, g2):
    """Unpolarised PBE exchange: energy per volume and its partial derivatives ∂/∂n, ∂/∂|∇n|²."""
    n13 = np.cbrt(n)
    ex_unif = -_CX * n * n13                                  # per volume
    inv = 1.0 / (4 * _KF ** 2 * n ** (8 / 3))                 # s² = g2 · inv
    s2 = g2 * inv
    den = 1 + PBE_MU * s2 / PBE_KAPPA
    Fx = 1 + PBE_KAPPA - PBE_KAPPA / den
    dFx = PBE_MU / (den * den)                                # dFx/ds²
    e = ex_unif * Fx
    de_dn = (4 / 3) * ex_unif / n * Fx + ex_unif * dFx * (-8 / 3) * s2 / n
    de_dg2 = ex_unif * dFx * inv
    return e, de_dn, de_dg2


def _pbe_correlation(n, zeta, g2):
    """PBE correlation: n(ε_c^LDA + H). Energy per volume and ∂/∂n, ∂/∂ζ, ∂/∂|∇n|²."""
    rs = (3 / (4 * np.pi * n)) ** (1 / 3)
    ec, dec_drs, dec_dz = pw92_correlation(rs, zeta)
    opz = np.clip(1 + zeta, 1e-12, 2.0)
    omz = np.clip(1 - zeta, 1e-12, 2.0)
    phi = 0.5 * (opz ** (2 / 3) + omz ** (2 / 3))
    dphi = (np.cbrt(opz) ** -1 - np.cbrt(omz) ** -1) / 3
    phi3 = phi ** 3
    # t² = |∇n|² / (2 φ k_s n)², k_s² = 4 k_F / π
    ct = np.pi / (16 * _KF * n ** (7 / 3))                    # y = t² = g2 · ct / φ²
    y = np.minimum(g2 * ct / (phi * phi), 1e30)
    bg = PBE_BETA / PBE_GAMMA
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        E = np.exp(np.minimum(-ec / (PBE_GAMMA * phi3), 700.0))
        A = np.minimum(bg / np.maximum(E - 1, 1e-300), 1e30)
        dA_dec = np.where(A < 1e30, A * A * E / (PBE_BETA * phi3), 0.0)
        dA_dphi = np.where(A < 1e30, -A * A * E * 3 * ec / (PBE_BETA * phi3 * phi), 0.0)
    u = A * y
    num = y * (1 + u)
    den = 1 + u + u * u
    Q = num / den
    dQ_dy = ((1 + 2 * u) * den - num * (A + 2 * A * u)) / (den * den)
    dQ_dA = (y * y * den - num * (y + 2 * u * y)) / (den * den)
    L = np.log1p(bg * Q)
    H = PBE_GAMMA * phi3 * L
    dH_dQ = PBE_BETA * phi3 / (1 + bg * Q)
    # H through its three inputs: ε_c, φ and t²
    dH_dec = dH_dQ * dQ_dA * dA_dec
    dH_dphi = 3 * PBE_GAMMA * phi * phi * L + dH_dQ * (dQ_dA * dA_dphi + dQ_dy * (-2 * y / phi))
    dH_dy = dH_dQ * dQ_dy
    e = n * (ec + H)
    dec_dn = dec_drs * (-rs / (3 * n))
    de_dn = ec + H + n * (dec_dn + dH_dec * dec_dn + dH_dy * (-7 / 3) * y / n)
    de_dz = n * (dec_dz + dH_dec * dec_dz + dH_dphi * dphi)
    de_dg2 = n * dH_dy * ct / (phi * phi)
    return e, de_dn, de_dz, de_dg2


def pbe_xc(rho_up, rho_dn, s_uu, s_ud, s_dd):
    """Spin-polarised PBE. ``s_uu`` = |∇ρ↑|², ``s_ud`` = ∇ρ↑·∇ρ↓, ``s_dd`` = |∇ρ↓|².

    Returns (e_xc per volume, ∂e/∂ρ↑, ∂e/∂ρ↓, ∂e/∂s_uu, ∂e/∂s_ud, ∂e/∂s_dd). The Kohn–Sham potential
    is v_σ = ∂e/∂ρσ − ∇·(2 ∂e/∂s_σσ ∇ρσ + ∂e/∂s_ud ∇ρσ'), which the caller forms with its own
    gradient; with all gradients zero this is lda_xc exactly."""
    ru = np.maximum(rho_up, 0.0)
    rd = np.maximum(rho_dn, 0.0)
    n = ru + rd
    out = [np.zeros_like(n) for _ in range(6)]
    m = n > RHO_FLOOR
    if not m.any():
        return tuple(out)
    ru, rd, n = ru[m], rd[m], n[m]
    suu, sud, sdd = s_uu[m], s_ud[m], s_dd[m]
    # exchange by spin scaling: E_x[ρ↑, ρ↓] = ½ E_x[2ρ↑] + ½ E_x[2ρ↓]
    e = np.zeros_like(n)
    vu = np.zeros_like(n)
    vd = np.zeros_like(n)
    wuu = np.zeros_like(n)
    wdd = np.zeros_like(n)
    for r, s, v, w in ((ru, suu, vu, wuu), (rd, sdd, vd, wdd)):
        ok = r > RHO_FLOOR / 2
        ex, dn, dg = _pbe_exchange(2 * r[ok], 4 * s[ok])
        e[ok] += 0.5 * ex
        v[ok] += dn
        w[ok] += 2 * dg
    zeta = np.clip((ru - rd) / n, -1.0, 1.0)
    g2 = np.maximum(suu + 2 * sud + sdd, 0.0)
    ec, dn, dz, dg = _pbe_correlation(n, zeta, g2)
    e += ec
    vu += dn + dz * (1 - zeta) / n
    vd += dn - dz * (1 + zeta) / n
    wuu += dg
    wdd += dg
    for o, x in zip(out, (e, vu, vd, wuu, 2 * dg, wdd)):
        o[m] = x
    return tuple(out)
