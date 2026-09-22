"""Decays of unstable particles, all computed from the same amplitudes.

For each unstable particle every kinematically open two-body final state is tried; the
ones the vertices allow get a width, the rest come out exactly zero. Particles with no
open two-body channel (the muon and tau) decay through three-body final states, integrated
over the Dalitz plot.

The confinement boundary: quarks and gluons never appear alone. A decay into quarks
needs at least the mass of the lightest hadron, the pion. That one number (0.1396 GeV) is a
measured input here; the lattice-QCD rung of this project computes hadron masses from
quarks and gluons directly.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from .amplitudes import Amplitude
from .process import (HBAR_GEV_S, _avg_factor, _identical_factor, leg, model, registry, species,
                      width_2body)

PION_MASS = 0.13957      # lightest hadron (measured); gates quark final states
UNSTABLE = ("Z", "W+", "W-", "H", "t", "t~", "mu-", "mu+", "tau-", "tau+")


def _allowed_by_confinement(parent: str, products: tuple[str, ...]) -> bool:
    quarks = [p for p in products if species(p).colour > 1]
    if not quarks:
        return True
    rest = sum(species(p).mass for p in products if species(p).colour == 1)
    return species(parent).mass - rest >= PION_MASS * (1 if len(quarks) >= 2 else 1)


def _charge_ok(parent: str, products) -> bool:
    return abs(species(parent).charge - sum(species(p).charge for p in products)) < 1e-9


@functools.lru_cache(maxsize=None)
def channels(parent: str) -> tuple[tuple[tuple[str, ...], float], ...]:
    """((products, width GeV), ...) sorted by width, for every open channel."""
    names = [n for n in registry() if n != parent]
    M = species(parent).mass
    out = []
    seen = set()
    for i, a in enumerate(names):
        for b in names[i:]:
            key = tuple(sorted((a, b)))
            if key in seen or species(a).mass + species(b).mass >= M:
                continue
            seen.add(key)
            if not _charge_ok(parent, key) or not _allowed_by_confinement(parent, key):
                continue
            w = width_2body(parent, a, b)
            if w > 1e-12 * M:
                out.append((key, w))
    if not out:
        out = _three_body_channels(parent)
    out.sort(key=lambda x: -x[1])
    return tuple(out)


def _three_body_channels(parent: str):
    M = species(parent).mass
    light = [n for n in registry() if n != parent and species(n).mass < M]
    out = []
    seen = set()
    for i, a in enumerate(light):
        for j, b in enumerate(light[i:], i):
            for c in light[j:]:
                key = tuple(sorted((a, b, c)))
                if key in seen:
                    continue
                seen.add(key)
                if sum(species(x).mass for x in key) >= M:
                    continue
                if not _charge_ok(parent, key) or not _allowed_by_confinement(parent, key):
                    continue
                w = width_3body(parent, key, n_points=400)
                if w > 1e-30:
                    out.append((key, width_3body(parent, key, n_points=3000)))
    return out


def dalitz_points(M, m1, m2, m3, n, rng):
    """Uniform points in the Dalitz plot and the three momenta in the parent's rest frame."""
    lo12, hi12 = (m1 + m2) ** 2, (M - m3) ** 2
    lo23, hi23 = (m2 + m3) ** 2, (M - m1) ** 2
    area_box = (hi12 - lo12) * (hi23 - lo23)
    pts = []
    tries = 0
    while len(pts) < n and tries < 50 * n:
        tries += 1
        s12 = lo12 + (hi12 - lo12) * rng.random()
        s23 = lo23 + (hi23 - lo23) * rng.random()
        mom = _momenta_from_dalitz(M, m1, m2, m3, s12, s23, rng)
        if mom is not None:
            pts.append(mom)
    return pts, area_box * len(pts) / max(tries, 1)


def _momenta_from_dalitz(M, m1, m2, m3, s12, s23, rng):
    # energies in the parent frame
    E3 = (M * M + m3 * m3 - s12) / (2 * M)
    E1 = (M * M + m1 * m1 - s23) / (2 * M)
    E2 = M - E1 - E3
    if E1 < m1 or E2 < m2 or E3 < m3:
        return None
    p1, p2, p3 = (math.sqrt(max(E * E - m * m, 0)) for E, m in ((E1, m1), (E2, m2), (E3, m3)))
    if p1 == 0 or p3 == 0:
        return None
    cos13 = (p2 * p2 - p1 * p1 - p3 * p3) / (2 * p1 * p3)
    if abs(cos13) > 1:
        return None
    sin13 = math.sqrt(1 - cos13 * cos13)
    v1 = np.array([0, 0, 1.0]) * p1
    v3 = np.array([sin13, 0, cos13]) * p3
    v2 = -(v1 + v3)
    R = _random_rotation(rng)
    return [np.concatenate([[E], R @ v]) for E, v in ((E1, v1), (E2, v2), (E3, v3))]


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def width_3body(parent: str, products, n_points: int = 3000, seed: int = 1) -> float:
    M = species(parent).mass
    ms = [species(p).mass for p in products]
    amp = Amplitude(model(), [leg(parent, True)] + [leg(p, False) for p in products], _widths_for_propagators())
    rng = np.random.default_rng(seed)
    pts, area = dalitz_points(M, *ms, n_points, rng)
    if not pts:
        return 0.0
    P = np.array([M, 0, 0, 0.0])
    m2 = [float(np.sum(np.abs(amp.evaluate([P] + mom)) ** 2)) for mom in pts]
    avg = float(np.mean(m2)) / _avg_factor((parent,))
    # dΓ = |M|² / (256 π³ M³) ds12 ds23
    return avg * area / (256 * math.pi ** 3 * M ** 3) / _identical_factor(products)


@functools.lru_cache(maxsize=1)
def _widths_for_propagators() -> dict:
    # Widths of the W and Z enter propagators; at tree level from their own two-body decays.
    return {"W": total_width("W+", use_cache=False), "Z": total_width("Z", use_cache=False)}


def total_width(parent: str, use_cache: bool = True) -> float:
    if not use_cache:
        names = [n for n in registry() if n != parent]
        M = species(parent).mass
        tot = 0.0
        seen = set()
        for i, a in enumerate(names):
            for b in names[i:]:
                key = tuple(sorted((a, b)))
                if key in seen or species(a).mass + species(b).mass >= M:
                    continue
                seen.add(key)
                if _charge_ok(parent, key) and _allowed_by_confinement(parent, key):
                    tot += width_2body(parent, a, b)
        return tot
    return sum(w for _, w in channels(parent))


def lifetime_seconds(parent: str) -> float:
    return HBAR_GEV_S / total_width(parent)


def branching_ratios(parent: str) -> list[tuple[tuple[str, ...], float]]:
    ch = channels(parent)
    tot = sum(w for _, w in ch)
    return [(p, w / tot) for p, w in ch]
