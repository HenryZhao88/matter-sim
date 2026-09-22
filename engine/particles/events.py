"""Collision events: beams in, particles out, decays chained until stable or confined.

1. Every two-body final state the beams could produce is tried; each gets its cross section
   from the amplitudes (zero if the Lagrangian forbids it).
2. An event picks a final state with probability ∝ σ and an angle from dσ/dcosθ.
3. Unstable particles decay by their computed branching ratios after a flight distance
   drawn from their computed lifetime, recursively.
4. Quarks and gluons stop at the confinement boundary: in nature they become jets of
   hadrons, a step no known method computes from first principles in real time.

A coloured particle is treated as confined unless it decays faster than the strong
interaction acts, i.e. unless its width exceeds Λ_QCD, which itself comes from the
one-loop running of α_s. That is why the top quark decays as a free quark and the
bottom quark does not.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

from .decays import UNSTABLE, _momenta_from_dalitz, _random_rotation, channels, total_width
from .process import GEV2_TO_PB, cross_section, registry, species, two_body

ACCEPTANCE = 0.95        # |cos θ| < 0.95: forward scattering at zero angle is never observed
E_MIN_MASSLESS = 1.0     # GeV: photons and gluons softer than this are not counted (they are
                         # emitted with divergent probability as E → 0, as QED predicts)
HBARC_M = 1.973269804e-16  # ħc in GeV·m


def lambda_qcd() -> float:
    """One-loop Λ_QCD (five flavours) from α_s(m_Z)."""
    from .model import ALPHA_S, M_Z_INPUT
    beta0 = 11 - 2 * 5 / 3
    return M_Z_INPUT * math.exp(-2 * math.pi / (beta0 * ALPHA_S))


def is_confined(name: str) -> bool:
    sp = species(name)
    if sp.colour == 1:
        return False
    if name in UNSTABLE:
        return total_width(name) < lambda_qcd()
    return True


@dataclass
class Particle:
    name: str
    p: list[float]                 # 4-momentum (E, px, py, pz), GeV
    origin: list[float]            # production point, metres
    parent: int | None
    status: str                    # "final", "decayed", "confined", "invisible"
    decay_point: list[float] | None = None
    children: list[int] = field(default_factory=list)


def _boost(p, beta):
    b2 = float(beta @ beta)
    if b2 < 1e-18:
        return p
    g = 1 / math.sqrt(1 - b2)
    bp = float(beta @ p[1:])
    E = g * (p[0] + bp)
    vec = p[1:] + ((g - 1) * bp / b2 + g * p[0]) * beta
    return np.concatenate([[E], vec])


_TABLES: dict[tuple[str, str, float], tuple] = {}   # (beam, beam, √s) → outcome table


def outcomes(a: str, b: str, sqrt_s: float) -> tuple:
    """((c, d, σ pb, cos nodes, dσ/dcos), ...) for every open final state, largest first."""
    key = (a, b, float(sqrt_s))
    if key not in _TABLES:
        _TABLES[key] = _compute_outcomes(a, b, float(sqrt_s))
    return _TABLES[key]


def _compute_outcomes(a: str, b: str, sqrt_s: float) -> tuple:
    names = list(registry())
    qa, qb = species(a).charge, species(b).charge
    out, seen = [], set()
    for i, c in enumerate(names):
        for d in names[i:]:
            key = tuple(sorted((c, d)))
            if key in seen:
                continue
            seen.add(key)
            if abs(species(c).charge + species(d).charge - qa - qb) > 1e-9:
                continue
            if species(c).mass + species(d).mass >= sqrt_s:
                continue
            if not _visible_energies_ok(c, d, sqrt_s):
                continue
            if not _nonzero(a, b, c, d, sqrt_s):
                continue
            coloured = sum(species(x).colour > 1 for x in (a, b, c, d))
            sig, x, ds = cross_section(a, b, c, d, sqrt_s, widths=_propagator_widths(), n_cos=24,
                                       cos_max=ACCEPTANCE, max_configs=2048 if coloured >= 3 else None)
            if sig > 1e-9:
                out.append((c, d, sig, tuple(x), tuple(ds)))
    out.sort(key=lambda r: -r[2])
    return tuple(out)


def _visible_energies_ok(c: str, d: str, sqrt_s: float) -> bool:
    mc, md = species(c).mass, species(d).mass
    Ec = (sqrt_s ** 2 + mc * mc - md * md) / (2 * sqrt_s)
    Ed = sqrt_s - Ec
    return all(E >= E_MIN_MASSLESS for E, m in ((Ec, mc), (Ed, md)) if m == 0)


def _nonzero(a, b, c, d, sqrt_s) -> bool:
    """Cheap test at one generic angle: forbidden final states give exactly zero."""
    from .amplitudes import Amplitude
    from .process import beams, leg, model
    amp = Amplitude(model(), [leg(a, True), leg(b, True), leg(c, False), leg(d, False)])
    amp.max_configs = 256
    p1, p2, _ = beams(sqrt_s, species(a).mass, species(b).mass)
    p3, p4, _ = two_body(sqrt_s, species(c).mass, species(d).mass, 0.3718, 1.234)
    return bool(np.any(np.abs(amp.evaluate([p1, p2, p3, p4])) > 1e-14))


@functools.lru_cache(maxsize=1)
def _propagator_widths() -> dict:
    return {"Z": total_width("Z"), "W": total_width("W+"), "t": total_width("t"), "H": total_width("H")}


def generate(a: str, b: str, sqrt_s: float, rng: np.random.Generator) -> dict:
    table = outcomes(a, b, sqrt_s)
    if not table:
        return {"beams": [a, b], "sqrt_s": sqrt_s, "particles": [], "channel": None, "sigma_total_pb": 0.0}
    sig = np.array([r[2] for r in table])
    k = int(rng.choice(len(table), p=sig / sig.sum()))
    c, d, _, xs, ds = table[k]
    # sample cos θ from the tabulated dσ/dcosθ
    xs, ds = np.array(xs), np.clip(np.array(ds), 0, None)
    order = np.argsort(xs)
    xs, ds = xs[order], ds[order]
    cdf = np.cumsum(ds)
    cdf /= cdf[-1]
    ct = float(np.interp(rng.random(), cdf, xs))
    phi = 2 * math.pi * rng.random()
    p3, p4, _ = two_body(sqrt_s, species(c).mass, species(d).mass, ct, phi)
    parts: list[Particle] = []
    for name, p in ((c, p3), (d, p4)):
        _add(parts, name, p, np.zeros(3), None, rng)
    return {"beams": [a, b], "sqrt_s": sqrt_s, "channel": [c, d],
            "sigma_pb": table[k][2], "sigma_total_pb": float(sig.sum()),
            "particles": [vars(p) for p in parts], "acceptance": ACCEPTANCE}


def _add(parts, name, p, origin, parent, rng, depth=0):
    idx = len(parts)
    sp = species(name)
    status = "final"
    if sp.colour > 1 and is_confined(name):
        status = "confined"
    elif name.startswith("nu"):
        status = "invisible"
    parts.append(Particle(name, [float(x) for x in p], [float(x) for x in origin], parent, status))
    if parent is not None:
        parts[parent].children.append(idx)
    if name not in UNSTABLE or depth > 6 or status == "confined":
        return
    # flight distance from the computed lifetime: L = βγ cτ, exponential
    width = total_width(name)
    ctau = HBARC_M / width
    pv = np.array(p[1:])
    pmag = float(np.linalg.norm(pv))
    bg = pmag / sp.mass
    L = -math.log(max(rng.random(), 1e-300)) * bg * ctau
    direction = pv / pmag if pmag > 0 else np.array([0, 0, 1.0])
    dp = origin + direction * min(L, 1e4)
    parts[idx].status = "decayed"
    parts[idx].decay_point = [float(x) for x in dp]
    ch = channels(name)
    w = np.array([x[1] for x in ch])
    prods = ch[int(rng.choice(len(ch), p=w / w.sum()))][0]
    beta = pv / p[0]
    M = sp.mass
    ms = [species(x).mass for x in prods]
    if len(prods) == 2:
        ct = 2 * rng.random() - 1
        ph = 2 * math.pi * rng.random()
        q1, q2, _ = two_body(M, ms[0], ms[1], ct, ph)
        moms = [q1, q2]
    else:
        moms = None
        while moms is None:
            s12 = (ms[0] + ms[1]) ** 2 + ((M - ms[2]) ** 2 - (ms[0] + ms[1]) ** 2) * rng.random()
            s23 = (ms[1] + ms[2]) ** 2 + ((M - ms[0]) ** 2 - (ms[1] + ms[2]) ** 2) * rng.random()
            moms = _momenta_from_dalitz(M, *ms, s12, s23, rng)
    for pname, q in zip(prods, moms):
        _add(parts, pname, _boost(np.array(q), beta), dp, idx, rng, depth + 1)


def scan(a: str, b: str, energies) -> list[dict]:
    """Total visible cross section vs √s, and its biggest components."""
    rows = []
    for E in energies:
        table = outcomes(a, b, float(E))
        rows.append({"sqrt_s": float(E), "total_pb": float(sum(r[2] for r in table)),
                     "channels": [{"final": [r[0], r[1]], "pb": r[2]} for r in table[:6]]})
    return rows
