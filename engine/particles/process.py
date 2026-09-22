"""Particles by name, and the observables built from amplitudes.

Cross sections (2 → 2) and decay widths (1 → 2, 1 → 3) come straight from |M|² summed over
spins and colours; nothing about which processes exist is written here.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from . import model as sm
from .amplitudes import Amplitude, Leg

GEV2_TO_PB = 0.3893793721e9    # (ħc)² in GeV² pb
HBAR_GEV_S = 6.582119569e-25    # ħ in GeV s

LEPTONS = ("e", "mu", "tau")
NEUTRINOS = ("nu_e", "nu_mu", "nu_tau")
QUARKS = ("u", "d", "c", "s", "t", "b")


@dataclass(frozen=True)
class Species:
    name: str          # e.g. "e-", "u~", "W+", "g"
    kind: str          # "f", "fbar", "v", "s"
    base: str          # model name ("e", "u", "Z", ...)
    mass: float
    charge: float
    colour: int        # 1, 3 (quark), 8 (gluon)
    spin_states: int
    stable: bool       # stable, or confined (never seen alone)
    confined: bool = False


@functools.lru_cache(maxsize=1)
def model() -> sm.Model:
    return sm.build()


def _fermion_names(f: sm.Fermion) -> tuple[str, str]:
    if f.name in LEPTONS:
        return f"{f.name}-", f"{f.name}+"
    return f.name, f"{f.name}~"


@functools.lru_cache(maxsize=1)
def registry() -> dict[str, tuple[Species, Leg]]:
    """Every particle, with a template leg (direction filled in when used)."""
    m = model()
    reg: dict[str, tuple[Species, Leg]] = {}
    for f in m.fermions:
        states = list(range(f.index, f.index + f.colours))
        pname, aname = _fermion_names(f)
        conf = f.colours == 3
        stable = f.name in ("e",) or f.name in NEUTRINOS or conf
        reg[pname] = (Species(pname, "f", f.name, f.mass, f.charge, f.colours, 2, stable, conf),
                      Leg("f", states, True, f.mass, pname))
        reg[aname] = (Species(aname, "fbar", f.name, f.mass, -f.charge, f.colours, 2, stable, conf),
                      Leg("fbar", states, True, f.mass, aname))
    nV = len(m.vectors)

    def unit(i):
        e = np.zeros(nV, complex)
        e[i] = 1
        return e

    reg["photon"] = (Species("photon", "v", "photon", 0.0, 0.0, 1, 2, True),
                     Leg("v", [unit(m.vector("photon").index)], True, 0.0, "photon"))
    z = m.vector("Z")
    reg["Z"] = (Species("Z", "v", "Z", z.mass, 0.0, 1, 3, False), Leg("v", [unit(z.index)], True, z.mass, "Z"))
    reg["g"] = (Species("g", "v", "g", 0.0, 0.0, 8, 2, True, True),
                Leg("v", [unit(a) for a in range(8)], True, 0.0, "g"))
    i1, i2 = m.w_pair
    mw = m.vectors[i1].mass
    plus, minus = _w_combinations(m, i1, i2)
    reg["W+"] = (Species("W+", "v", "W", mw, 1.0, 1, 3, False), Leg("v", [plus], True, mw, "W+"))
    reg["W-"] = (Species("W-", "v", "W", mw, -1.0, 1, 3, False), Leg("v", [minus], True, mw, "W-"))
    reg["H"] = (Species("H", "s", "H", m.higgs_mass, 0.0, 1, 1, False), Leg("s", [None], True, m.higgs_mass, "H"))
    return reg


def _w_combinations(m: sm.Model, i1: int, i2: int):
    """Which complex mix of the two degenerate fields is the W⁺? Ask the Lagrangian:
    it is the one a u quark and a d̄ antiquark can turn into."""
    nV = len(m.vectors)
    a = np.zeros(nV, complex)
    a[i1], a[i2] = 1 / math.sqrt(2), -1j / math.sqrt(2)
    b = a.conj()
    u, d = m.fermion("u"), m.fermion("d")
    # coupling ū γ d W: amplitude for d → u + W⁻-like emission ∝ Σ_v L[v,u,d] mix_v
    ca = abs(np.sum(m.ffv_L[:, u.index, d.index] * a))
    cb = abs(np.sum(m.ffv_L[:, u.index, d.index] * b))
    # absorbing an incoming W⁺ turns d into u: the incoming W⁺ mix contracts with L[v,u,d]
    return (a, b) if ca > cb else (b, a)


def leg(name: str, incoming: bool) -> Leg:
    sp, template = registry()[name]
    return Leg(template.kind, template.states, incoming, template.mass, name)


def species(name: str) -> Species:
    return registry()[name][0]


def antiparticle(name: str) -> str:
    if name in ("photon", "Z", "g", "H"):
        return name
    if name == "W+":
        return "W-"
    if name == "W-":
        return "W+"
    if name.endswith("-"):
        return name[:-1] + "+"
    if name.endswith("+"):
        return name[:-1] + "-"
    return name[:-1] if name.endswith("~") else name + "~"


# ------------------------------------------------------------------ kinematics
def two_body(sqrt_s: float, m3: float, m4: float, cos_t: float, phi: float = 0.0):
    """Beams along ±z (massless or massive) and a 2-body final state in the CM frame."""
    E = sqrt_s / 2
    pf = math.sqrt(max((sqrt_s ** 2 - (m3 + m4) ** 2) * (sqrt_s ** 2 - (m3 - m4) ** 2), 0)) / (2 * sqrt_s)
    sin_t = math.sqrt(max(1 - cos_t * cos_t, 0))
    n = np.array([sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t])
    p3 = np.concatenate([[math.sqrt(pf * pf + m3 * m3)], pf * n])
    p4 = np.concatenate([[math.sqrt(pf * pf + m4 * m4)], -pf * n])
    return p3, p4, pf


def beams(sqrt_s: float, m1: float, m2: float):
    s = sqrt_s ** 2
    pi = math.sqrt(max((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2), 0)) / (2 * sqrt_s)
    p1 = np.array([math.sqrt(pi * pi + m1 * m1), 0, 0, pi])
    p2 = np.array([math.sqrt(pi * pi + m2 * m2), 0, 0, -pi])
    return p1, p2, pi


def _avg_factor(names_in) -> float:
    f = 1.0
    for n in names_in:
        sp = species(n)
        f *= sp.spin_states * sp.colour
    return f


def _identical_factor(names_out) -> float:
    f = 1.0
    for n in set(names_out):
        f *= math.factorial(list(names_out).count(n))
    return f


def cross_section(a: str, b: str, c: str, d: str, sqrt_s: float, widths=None, n_cos: int = 64,
                  cos_max: float = 1.0, max_configs: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """σ(a b → c d) in pb, plus dσ/dcosθ on Gauss–Legendre nodes (θ of particle c)."""
    names_in, names_out = (a, b), (c, d)
    ma, mb, mc, md = (species(x).mass for x in (a, b, c, d))
    if sqrt_s <= mc + md or sqrt_s <= ma + mb:
        return 0.0, np.zeros(0), np.zeros(0)
    amp = Amplitude(model(), [leg(a, True), leg(b, True), leg(c, False), leg(d, False)], widths)
    amp.max_configs = max_configs
    p1, p2, pi = beams(sqrt_s, ma, mb)
    x, w = np.polynomial.legendre.leggauss(n_cos)
    x, w = x * cos_max, w * cos_max
    dsig = np.zeros(n_cos)
    s = sqrt_s ** 2
    for k, ct in enumerate(x):
        p3, p4, pf = two_body(sqrt_s, mc, md, float(ct))
        M = amp.evaluate([p1, p2, p3, p4])
        m2 = float(np.sum(np.abs(M) ** 2)) * amp.config_scale / _avg_factor(names_in)
        # dσ/dcosθ = |M|² p_f / (32 π s p_i)
        dsig[k] = m2 * pf / (32 * math.pi * s * pi) / _identical_factor(names_out)
    sigma = float(np.sum(w * dsig)) * GEV2_TO_PB
    return sigma, x, dsig * GEV2_TO_PB


def width_2body(parent: str, c: str, d: str, widths=None) -> float:
    """Γ(parent → c d) in GeV."""
    M = species(parent).mass
    mc, md = species(c).mass, species(d).mass
    if M <= mc + md:
        return 0.0
    amp = Amplitude(model(), [leg(parent, True), leg(c, False), leg(d, False)], widths)
    P = np.array([M, 0, 0, 0.0])
    p3, p4, pf = two_body(M, mc, md, 0.3, 0.7)          # isotropic after spin sum
    m2 = float(np.sum(np.abs(amp.evaluate([P, p3, p4])) ** 2)) / _avg_factor((parent,))
    return m2 * pf / (8 * math.pi * M * M) / _identical_factor((c, d))
