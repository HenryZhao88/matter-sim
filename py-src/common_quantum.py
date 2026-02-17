from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

A0 = 1.0
HBAR = 1.0
M_E = 1.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.cos(theta)
    z = r * math.sin(theta) * math.sin(phi)
    return x, y, z


def associated_laguerre(k: int, alpha: int, x: float) -> float:
    if k == 0:
        return 1.0
    if k == 1:
        return 1.0 + alpha - x
    lm2 = 1.0
    lm1 = 1.0 + alpha - x
    l = lm1
    for j in range(2, k + 1):
        l = ((2 * j - 1 + alpha - x) * lm1 - (j - 1 + alpha) * lm2) / j
        lm2, lm1 = lm1, l
    return l


def associated_legendre(l: int, m: int, x: float) -> float:
    m_abs = abs(m)

    pmm = 1.0
    if m_abs > 0:
        somx2 = math.sqrt(max(0.0, (1.0 - x) * (1.0 + x)))
        fact = 1.0
        for _ in range(1, m_abs + 1):
            pmm *= -fact * somx2
            fact += 2.0

    if l == m_abs:
        plm = pmm
    else:
        pm1m = x * (2 * m_abs + 1) * pmm
        if l == m_abs + 1:
            plm = pm1m
        else:
            pll = pm1m
            for ll in range(m_abs + 2, l + 1):
                pll = ((2 * ll - 1) * x * pm1m - (ll + m_abs - 1) * pmm) / (ll - m_abs)
                pmm, pm1m = pm1m, pll
            plm = pm1m

    if m < 0:
        sign = -1.0 if (m_abs % 2 == 1) else 1.0
        ratio = math.factorial(l - m_abs) / math.factorial(l + m_abs)
        return sign * ratio * plm
    return plm


@lru_cache(maxsize=128)
def _build_r_cdf(n: int, l: int, samples: int = 4096) -> Tuple[float, List[float]]:
    r_max = 10.0 * n * n * A0
    dr = r_max / (samples - 1)
    cdf: List[float] = [0.0] * samples
    total = 0.0

    for i in range(samples):
        r = i * dr
        rho = 2.0 * r / (n * A0)
        k = n - l - 1
        alpha = 2 * l + 1
        lag = associated_laguerre(k, alpha, rho)
        norm = (2.0 / (n * A0)) ** 3 * math.gamma(n - l) / (2.0 * n * math.gamma(n + l + 1))
        radial = math.sqrt(norm) * math.exp(-rho / 2.0) * (rho**l) * lag
        pdf = r * r * radial * radial
        total += pdf
        cdf[i] = total

    if total <= 0.0:
        return r_max, [i / (samples - 1) for i in range(samples)]

    cdf = [v / total for v in cdf]
    return r_max, cdf


def sample_r(n: int, l: int) -> float:
    r_max, cdf = _build_r_cdf(n, l)
    u = random.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo * (r_max / (len(cdf) - 1))


@lru_cache(maxsize=128)
def _build_theta_cdf(l: int, m: int, samples: int = 2048) -> List[float]:
    dtheta = math.pi / (samples - 1)
    cdf: List[float] = [0.0] * samples
    total = 0.0

    for i in range(samples):
        theta = i * dtheta
        x = math.cos(theta)
        plm = associated_legendre(l, m, x)
        pdf = max(0.0, math.sin(theta) * plm * plm)
        total += pdf
        cdf[i] = total

    if total <= 0.0:
        return [i / (samples - 1) for i in range(samples)]
    return [v / total for v in cdf]


def sample_theta(l: int, m: int) -> float:
    cdf = _build_theta_cdf(l, m)
    u = random.random()
    lo, hi = 0, len(cdf) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    return lo * (math.pi / (len(cdf) - 1))


def sample_phi() -> float:
    return 2.0 * math.pi * random.random()


def probability_flow(px: float, py: float, pz: float, m: int) -> Tuple[float, float, float]:
    r = math.sqrt(px * px + py * py + pz * pz)
    if r < 1e-6:
        return 0.0, 0.0, 0.0

    theta = math.acos(clamp(py / r, -1.0, 1.0))
    phi = math.atan2(pz, px)
    sin_t = max(1e-4, abs(math.sin(theta)))
    v_mag = HBAR * m / (M_E * r * sin_t)
    vx = -v_mag * math.sin(phi)
    vy = 0.0
    vz = v_mag * math.cos(phi)
    return vx, vy, vz


def heatmap_fire(value: float) -> Tuple[float, float, float, float]:
    value = clamp(value, 0.0, 1.0)
    colors = [
        (0.0, 0.0, 0.0, 1.0),
        (0.5, 0.0, 0.99, 1.0),
        (0.8, 0.0, 0.0, 1.0),
        (1.0, 0.5, 0.0, 1.0),
        (1.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0, 1.0),
    ]
    scaled = value * (len(colors) - 1)
    i = int(scaled)
    j = min(i + 1, len(colors) - 1)
    t = scaled - i
    return tuple(colors[i][k] + t * (colors[j][k] - colors[i][k]) for k in range(4))


def inferno_color(r: float, theta: float, phi: float, n: int, l: int, m: int, scale: float = 1.5) -> Tuple[float, float, float, float]:
    rho = 2.0 * r / (n * A0)
    k = n - l - 1
    alpha = 2 * l + 1
    lag = associated_laguerre(k, alpha, rho)

    norm = (2.0 / (n * A0)) ** 3 * math.gamma(n - l) / (2.0 * n * math.gamma(n + l + 1))
    radial = (math.sqrt(norm) * math.exp(-rho / 2.0) * (rho**l) * lag) ** 2

    x = math.cos(theta)
    ang = associated_legendre(l, m, x)
    angular = ang * ang

    intensity = radial * angular
    return heatmap_fire(intensity * scale * (5**n))


@dataclass
class Camera:
    radius: float = 50.0
    azimuth: float = 0.0
    elevation: float = math.pi / 2
    orbit_speed: float = 0.01
    zoom_speed: float = 10.0
    dragging: bool = False
    last_x: float = 0.0
    last_y: float = 0.0

    def position(self) -> Tuple[float, float, float]:
        e = clamp(self.elevation, 0.01, math.pi - 0.01)
        x = self.radius * math.sin(e) * math.cos(self.azimuth)
        y = self.radius * math.cos(e)
        z = self.radius * math.sin(e) * math.sin(self.azimuth)
        return x, y, z
