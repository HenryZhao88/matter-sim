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
