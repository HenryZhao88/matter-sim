"""Parton shower: quarks and gluons radiating more gluons, from the collinear limit of QCD.

A fast quark or gluon does not stay alone: the QCD vertices make it radiate gluons, which
split again, down to scales of about 1 GeV where confinement takes over. In the collinear
limit the rate of each splitting a → b c is

    dP = (α_s(p_T²) / 2π) · (dp_T² / p_T²) · P_{a→bc}(z) dz

with the Altarelli–Parisi splitting functions P, whose colour factors come from the
gauge group:  C_F (quark), C_A (gluon), T_R (gluon → quark pair). Those three numbers are
*computed* here from the SU(3) generators and structure constants in the model, and the
running of α_s uses β₀ = (11 C_A − 4 T_R n_f) / 3, also from the group.

This is the leading-logarithm approximation to the full theory, and the kinematics are
simplified (each jet's final partons are rescaled to conserve its original momentum).
The shower stops at the confinement scale; what happens below it is not computed.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .model import ALPHA_S, M_Z_INPUT, gell_mann, structure_constants

T_CUT = 1.0         # GeV: shower stops at p_T ≈ 1 GeV, where confinement takes over
QUARKS = ("u", "d", "s", "c", "b")


@functools.lru_cache(maxsize=1)
def group_constants() -> dict:
    """Casimirs and normalisation of SU(3), computed from the generators (not typed in)."""
    T = [l / 2 for l in gell_mann()]
    CF = float(np.real(sum(t @ t for t in T))[0, 0])                   # Σ_a T^a T^a = C_F · 1
    f = structure_constants(T)
    CA = float(np.einsum("acd,bcd->ab", f, f)[0, 0])                    # f^{acd} f^{bcd} = C_A δ^{ab}
    TR = float(np.real(np.trace(T[0] @ T[0])))                          # Tr(T^a T^b) = T_R δ^{ab}
    return {"CF": CF, "CA": CA, "TR": TR}


def alpha_s(q: float, nf: int = 5) -> float:
    """One-loop running coupling, β₀ from the group constants."""
    g = group_constants()
    b0 = (11 * g["CA"] - 4 * g["TR"] * nf) / (12 * math.pi)
    return ALPHA_S / (1 + ALPHA_S * b0 * math.log(q * q / (M_Z_INPUT * M_Z_INPUT)))


def splitting(kind: str, z: float) -> float:
    g = group_constants()
    if kind == "q>qg":
        return g["CF"] * (1 + z * z) / (1 - z)
    if kind == "g>gg":
        return g["CA"] * (z / (1 - z) + (1 - z) / z + z * (1 - z))
    if kind == "g>qq":
        return g["TR"] * (z * z + (1 - z) * (1 - z))
    raise KeyError(kind)


def _evolve(name: str, E: float, t_max: float, rng) -> tuple | None:
    """Next branching of a parton with energy E below scale t_max (= p_T²): the veto algorithm.

    Overestimate every kernel by c_k · g(z) with g(z) = 1/z + 1/(1−z), which bounds all three
    splitting functions (q→qg ≤ 2C_F g, g→gg ≤ 1.25 C_A g, g→qq̄ ≤ n_f T_R g/4), sample the
    overestimate exactly, then accept with probability true/overestimate.
    """
    g = group_constants()
    amax = alpha_s(T_CUT) * 1.02
    z0 = T_CUT / E
    if z0 >= 0.5:
        return None
    Iz = 2 * math.log((1 - z0) / z0)                     # ∫ g(z) dz over [z0, 1 − z0]
    if name == "g":
        kinds = [("g>gg", 1.25 * g["CA"]), ("g>qq", g["TR"] * len(QUARKS) / 4)]
    else:
        kinds = [("q>qg", 2 * g["CF"])]
    csum = sum(c for _, c in kinds)
    t = t_max
    while True:
        t *= rng.random() ** (2 * math.pi / (amax * csum * Iz))
        if t < T_CUT * T_CUT:
            return None
        kind, c = kinds[0] if len(kinds) == 1 or rng.random() < kinds[0][1] / csum else kinds[1]
        u = z0 * ((1 - z0) / z0) ** rng.random()           # ∝ 1/u on [z0, 1−z0]
        z = u if rng.random() < 0.5 else 1 - u             # ∝ g(z)
        pt = math.sqrt(t)
        if z * (1 - z) * E < pt:                           # outside the phase space at this p_T
            continue
        gz = 1 / z + 1 / (1 - z)
        if rng.random() * amax * c * gz <= alpha_s(pt) * splitting(kind, z):
            if kind == "q>qg":
                return t, z, (name, "g")
            if kind == "g>gg":
                return t, z, ("g", "g")
            q = QUARKS[int(rng.integers(len(QUARKS)))]
            return t, z, (q, q + "~")


def shower(name: str, p: np.ndarray, rng, t_start: float | None = None, depth: int = 0) -> list[tuple[str, np.ndarray]]:
    """Final partons (name, massless 4-momentum) from showering one parton."""
    E = float(p[0])
    t0 = t_start if t_start is not None else (E * E)
    out = []
    stack = [(name, np.asarray(p, float), t0)]
    while stack:
        nm, mom, tmax = stack.pop()
        if len(out) + len(stack) > 60 or mom[0] < 2 * T_CUT:
            out.append((nm, mom))
            continue
        b = _evolve(nm, float(mom[0]), tmax, rng)
        if b is None:
            out.append((nm, mom))
            continue
        t, z, (n1, n2) = b
        pt = math.sqrt(t)
        E1, E2 = z * mom[0], (1 - z) * mom[0]
        axis = mom[1:] / max(np.linalg.norm(mom[1:]), 1e-12)
        perp = np.cross(axis, [0, 0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0, 0])
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(axis, perp)
        phi = 2 * math.pi * rng.random()
        kt = pt * (math.cos(phi) * perp + math.sin(phi) * perp2)
        p1 = axis * math.sqrt(max(E1 * E1 - pt * pt, 0)) + kt
        p2 = axis * math.sqrt(max(E2 * E2 - pt * pt, 0)) - kt
        stack.append((n1, np.concatenate([[np.linalg.norm(p1)], p1]), t))
        stack.append((n2, np.concatenate([[np.linalg.norm(p2)], p2]), t))
    # restore the parent's energy and momentum (collinear kinematics are approximate)
    tot = sum(m for _, m in out)
    if tot[0] > 0:
        scale = p[0] / tot[0]
        out = [(n, m * scale) for n, m in out]
        drift = (p[1:] - sum(m[1:] for _, m in out)) / len(out)
        out = [(n, np.concatenate([[np.linalg.norm(m[1:] + drift)], m[1:] + drift])) for n, m in out]
    return out
