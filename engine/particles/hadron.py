"""Proton–proton collisions: partons from the measured proton, collisions from the Lagrangian.

    σ(pp → X) = Σ_ij ∫ dx₁ dx₂ f_i(x₁) f_j(x₂) σ̂_ij(x₁x₂ s)

The parton densities f are measured (see pdf.py). The partonic cross sections σ̂ come
from this project's Standard Model amplitudes, tabulated against partonic energy:

* every final state on a coarse grid of √ŝ (smooth, mostly strong-force channels), and
* colour-neutral final states of quark–antiquark pairs on a fine grid that resolves the
  W and Z resonances.

Only *hard* collisions are generated: partonic energy above SQRT_SHAT_MIN and
|cos θ*| < 0.95. Most of the ~100 mb proton cross section is soft, non-perturbative
scattering that no first-principles method computes; it is left out and said so.
Pairs related by CPT or by swapping the beams reuse one table (exact symmetries).
Bottom quarks inside the proton (about 1% of hard collisions) are not yet included.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np

from . import events as ev
from .decays import total_width
from .pdf import proton
from .process import antiparticle, species, two_body

SQRT_SHAT_MIN = 50.0
PARTONS = ["g", "u", "d", "s", "c", "u~", "d~", "s~", "c~"]
COARSE = [50, 80, 130, 220, 380, 650, 1100, 2000, 3600, 6500]
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "collider" / "hadron"


def canonical(a: str, b: str) -> tuple[tuple[str, str], bool, bool]:
    """(key, swapped, conjugated) with key a representative of {ab, ba, āb̄, b̄ā}."""
    options = []
    for conj in (False, True):
        x, y = (antiparticle(a), antiparticle(b)) if conj else (a, b)
        for swap in (False, True):
            key = (y, x) if swap else (x, y)
            options.append((key, swap, conj))
    return min(options, key=lambda o: o[0])


def _fine_grid() -> list[float]:
    pts = set(np.round(np.geomspace(SQRT_SHAT_MIN, 6500, 40), 3).tolist())
    for name in ("Z", "W+"):
        M, G = species(name).mass, total_width(name)
        for k in (-6, -3, -1.5, -0.75, -0.3, 0, 0.3, 0.75, 1.5, 3, 6):
            if M + k * G > SQRT_SHAT_MIN:
                pts.add(round(M + k * G, 4))
    return sorted(pts)


def _is_quark_antiquark(key) -> bool:
    a, b = key
    return a != "g" and b != "g" and (a.endswith("~") != b.endswith("~"))


def _colour_neutral(final) -> bool:
    return all(species(x).colour == 1 for x in final)


def is_rare(final) -> bool:
    """What a trigger keeps: anything besides quarks and gluons (top quarks count: they decay)."""
    return any(species(x).colour == 1 or x in ("t", "t~") for x in final)


class HadronCollider:
    def __init__(self, sqrt_s: float = 13600.0, notify=None) -> None:
        self.sqrt_s = sqrt_s
        self.notify = notify or (lambda m: None)
        self.pdf = proton(None)
        self.coarse: dict = {}      # key → [(√ŝ, table)]
        self.fine: dict = {}        # key → [(√ŝ, table of colour-neutral channels)]
        CACHE.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ tables
    def keys(self) -> list[tuple[str, str]]:
        return sorted({canonical(a, b)[0] for a in PARTONS for b in PARTONS})

    def _table(self, key, E, fine: bool):
        path = self._path(key, E, fine)
        if path.exists():
            return pickle.loads(path.read_bytes())
        if fine:
            tab = self._colour_neutral_outcomes(key[0], key[1], E)
        else:
            tab = ev._compute_outcomes(key[0], key[1], E, n_cos=12, max_configs=768)
        path.write_bytes(pickle.dumps(tab))
        return tab

    def _colour_neutral_outcomes(self, a, b, E):
        from .process import registry
        names = [n for n in registry() if species(n).colour == 1]
        out, seen = [], set()
        for i, c in enumerate(names):
            for d in names[i:]:
                key = tuple(sorted((c, d)))
                if key in seen or species(c).mass + species(d).mass >= E:
                    continue
                seen.add(key)
                if not ev.conserves((a, b), key) or not ev._visible_energies_ok(c, d, E):
                    continue
                if not ev._nonzero(a, b, c, d, E):
                    continue
                sig, x, ds = ev.cross_section(a, b, c, d, E, widths=ev._propagator_widths(), n_cos=12,
                                              cos_max=ev.ACCEPTANCE)
                if sig > 1e-9:
                    out.append((c, d, sig, tuple(x), tuple(ds)))
        out.sort(key=lambda r: -r[2])
        return tuple(out)

    def jobs(self) -> list:
        keys = self.keys()
        fine_keys = [k for k in keys if _is_quark_antiquark(k)]
        coarse_E = [E for E in COARSE if E < self.sqrt_s]
        fine_E = [E for E in _fine_grid() if E < self.sqrt_s]
        return [(k, E, False) for k in keys for E in coarse_E] + [(k, E, True) for k in fine_keys for E in fine_E]

    def build(self, progress=None, workers: int | None = None) -> None:
        """Compute (or load) every table. Slow once (parallel over cores), then cached."""
        import concurrent.futures as cf
        import os
        jobs = self.jobs()
        todo = [j for j in jobs if not self._path(*j).exists()]
        done = len(jobs) - len(todo)
        if todo:
            n = workers or max(1, (os.cpu_count() or 2) - 2)
            with cf.ProcessPoolExecutor(max_workers=n) as pool:
                for _ in pool.map(_build_job, [(self.sqrt_s,) + j for j in todo], chunksize=1):
                    done += 1
                    if progress:
                        progress(done, len(jobs))
        for k, E, fine in jobs:
            (self.fine if fine else self.coarse).setdefault(k, []).append((E, self._table(k, E, fine)))
        for d in (self.fine, self.coarse):
            for k in d:
                d[k].sort(key=lambda r: r[0])
        self._prepare_sampling()

    def _path(self, key, E, fine):
        tag = "fine" if fine else "coarse"
        return CACHE / f"{tag}_{key[0]}_{key[1]}_{E:.4f}.pkl".replace("~", "bar")

    # ------------------------------------------------------------ partonic σ̂ at any ŝ
    def _interp(self, rows, E):
        """Channel → (σ̂, shape) at energy E by log–log interpolation between grid points."""
        Es = [r[0] for r in rows]
        if E <= Es[0]:
            lo = hi = 0
        elif E >= Es[-1]:
            lo = hi = len(Es) - 1
        else:
            hi = int(np.searchsorted(Es, E))
            lo = hi - 1
        def as_map(tab):
            return {tuple(sorted((r[0], r[1]))): r for r in tab}
        A, B = as_map(rows[lo][1]), as_map(rows[hi][1])
        t = 0.0 if hi == lo else (math.log(E) - math.log(Es[lo])) / (math.log(Es[hi]) - math.log(Es[lo]))
        out = {}
        for ch in set(A) | set(B):
            sa = A[ch][2] if ch in A else 0.0
            sb = B[ch][2] if ch in B else 0.0
            if sa > 0 and sb > 0:
                s = math.exp((1 - t) * math.log(sa) + t * math.log(sb))
            else:
                s = (1 - t) * sa + t * sb
            src = (A.get(ch) if t < 0.5 else B.get(ch)) or A.get(ch) or B.get(ch)
            out[ch] = (s, src)
        return out

    def partonic(self, key, E) -> dict:
        ch = self._interp(self.coarse[key], E)
        if key in self.fine:
            ch = {c: v for c, v in ch.items() if not _colour_neutral(c)}
            ch.update(self._interp(self.fine[key], E))
        return ch

    # ------------------------------------------------------------ luminosity and sampling
    def _prepare_sampling(self) -> None:
        s = self.sqrt_s ** 2
        grid = sorted(set(COARSE) | set(_fine_grid()))
        grid = [E for E in grid if SQRT_SHAT_MIN <= E < 0.95 * self.sqrt_s]
        # cell edges in √ŝ
        edges = [grid[0]] + [math.sqrt(grid[i] * grid[i + 1]) for i in range(len(grid) - 1)] + [grid[-1]]
        yg, yw = np.polynomial.legendre.leggauss(24)
        cells = []
        for k, E in enumerate(grid):
            tau = E * E / s
            ymax = -0.5 * math.log(tau)
            dlnshat = 2 * (math.log(edges[k + 1]) - math.log(edges[k])) if edges[k + 1] > edges[k] else 0
            ys = yg * ymax
            for a in PARTONS:
                for b in PARTONS:
                    key, _, _ = canonical(a, b)
                    sig = sum(v[0] for v in self.partonic(key, E).values())
                    if sig <= 0:
                        continue
                    lum = 0.0
                    for y, w in zip(ys, yw):
                        x1, x2 = math.sqrt(tau) * math.exp(y), math.sqrt(tau) * math.exp(-y)
                        if x1 >= 1 or x2 >= 1:
                            continue
                        lum += w * ymax * x1 * self.pdf.f(a, x1, E) * x2 * self.pdf.f(b, x2, E)
                    # dσ = Σ f f σ̂ dx₁dx₂ = Σ (x₁f)(x₂f) σ̂ d(ln ŝ) dy
                    W = lum * sig * dlnshat
                    if W > 0:
                        cells.append((W, E, a, b, edges[k], edges[k + 1]))
        self.cells = cells
        self.weights = np.array([c[0] for c in cells])
        self.sigma_hard_pb = float(self.weights.sum())
        rare = []
        for W, E, a, b, lo, hi in cells:
            ch = self.partonic(canonical(a, b)[0], E)
            tot = sum(v[0] for v in ch.values())
            rare.append(W * sum(v[0] for c, v in ch.items() if is_rare(c)) / tot if tot > 0 else 0.0)
        self.rare_weights = np.array(rare)
        self.sigma_rare_pb = float(self.rare_weights.sum())

    def summary(self, top: int = 10) -> list[dict]:
        """Hard cross section split by what the collision makes (rough, from the grid)."""
        agg: dict[tuple, float] = {}
        for W, E, a, b, lo, hi in self.cells:
            key, swap, conj = canonical(a, b)
            chans = self.partonic(key, E)
            tot = sum(v[0] for v in chans.values())
            for ch, (sg, _) in chans.items():
                name = tuple(sorted(antiparticle(x) for x in ch)) if conj else ch
                agg[name] = agg.get(name, 0.0) + W * sg / tot
        rows = sorted(agg.items(), key=lambda kv: -kv[1])[:top]
        return [{"final": list(k), "pb": v, "share": v / self.sigma_hard_pb} for k, v in rows]

    def generate(self, rng: np.random.Generator, trigger: bool = False) -> dict:
        """One hard collision. With ``trigger``, only collisions that make something besides
        quarks and gluons, drawn from the exact conditional distribution (like a detector trigger)."""
        w_cells = self.rare_weights if trigger else self.weights
        k = int(rng.choice(len(self.cells), p=w_cells / w_cells.sum()))
        _, _, a, b, lo, hi = self.cells[k]
        E = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * rng.random()) if hi > lo else lo
        s = self.sqrt_s ** 2
        tau = E * E / s
        ymax = -0.5 * math.log(tau)
        # rapidity y from the parton luminosity x₁f(x₁)·x₂f(x₂) at this ŝ, by rejection sampling
        def lum(y):
            x1, x2 = math.sqrt(tau) * math.exp(y), math.sqrt(tau) * math.exp(-y)
            if x1 >= 1 or x2 >= 1:
                return 0.0, x1, x2
            return x1 * self.pdf.f(a, x1, E) * x2 * self.pdf.f(b, x2, E), x1, x2
        peak = max(lum(y)[0] for y in np.linspace(-ymax, ymax, 64)) * 1.2
        for _ in range(10000):
            y = (2 * rng.random() - 1) * ymax
            w, x1, x2 = lum(y)
            if rng.random() * peak <= w:
                break
        key, swap, conj = canonical(a, b)
        chans = self.partonic(key, E)
        names = [c for c in chans if is_rare(c)] if trigger else list(chans)
        w = np.array([chans[c][0] for c in names])
        pick = names[int(rng.choice(len(names), p=w / w.sum()))]
        row = chans[pick][1]
        c, d = row[0], row[1]
        # sample the angle from the tabulated shape
        xs, ds = np.array(row[3]), np.clip(np.array(row[4]), 0, None)
        order = np.argsort(xs)
        cdf = np.cumsum(ds[order])
        ct = float(np.interp(rng.random() * cdf[-1], cdf, xs[order]))
        if conj:
            c, d = antiparticle(c), antiparticle(d)
        if swap:
            ct = -ct
        phi = 2 * math.pi * rng.random()
        p3, p4, _ = two_body(E, species(c).mass, species(d).mass, ct, phi)
        beta = np.array([0.0, 0.0, math.tanh(y)])
        parts: list = []
        for name, p in ((c, p3), (d, p4)):
            ev._add(parts, name, ev._boost(np.array(p), beta), np.zeros(3), None, rng)
        remnants = []
        for sign, x in ((+1, x1), (-1, x2)):
            Er = (1 - x) * self.sqrt_s / 2
            remnants.append({"p": [Er, 0.0, 0.0, sign * Er], "x": x})
        return {"beams": ["p", "p"], "sqrt_s": self.sqrt_s, "channel": [c, d], "partons": [a, b], "trigger": trigger,
                "x": [x1, x2], "sqrt_shat": E, "sigma_total_pb": self.sigma_hard_pb,
                "particles": [vars(p) for p in parts], "remnants": remnants}


def _build_job(args) -> bool:
    """Worker-process entry: compute one table and write it to the cache."""
    sqrt_s, key, E, fine = args
    from .decays import warm_cache
    warm_cache(CACHE.parent / "decays.pkl")
    h = HadronCollider.__new__(HadronCollider)
    h.sqrt_s = sqrt_s
    h._table(key, E, fine)
    return True
