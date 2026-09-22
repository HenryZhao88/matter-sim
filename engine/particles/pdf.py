"""Parton distributions: what a proton is made of, as measured.

A proton is three valence quarks in a sea of gluons and quark–antiquark pairs; f_i(x, Q)
gives the density of parton i carrying momentum fraction x when probed at scale Q. This
structure is non-perturbative QCD. Lattice QCD can compute pieces of it but not yet the
whole function, so every collider experiment uses distributions fitted to measurements —
and so does this project. It is labelled as measured input, like the particle masses.

The set used is CT14 LO (Dulat et al., arXiv:1506.07443), fetched once from CERN's public
LHAPDF archive into .cache/pdf. Everything after "which partons collide" is computed.
"""

from __future__ import annotations

import functools
import io
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

SET_NAME = "CT14lo"
URL = f"https://lhapdfsets.web.cern.ch/current/{SET_NAME}.tar.gz"
CACHE = Path(__file__).resolve().parents[2] / ".cache" / "pdf"

PDG = {"g": 21, "d": 1, "u": 2, "s": 3, "c": 4, "b": 5}


def ensure_downloaded(notify=None) -> Path:
    path = CACHE / SET_NAME / f"{SET_NAME}_0000.dat"
    if path.exists():
        return path
    if notify:
        notify("Fetching the measured quark and gluon content of the proton (CT14 LO, 0.5 MB, once)…")
    CACHE.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(URL, timeout=60).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(CACHE, filter="data")
    return path


class PDF:
    """LHAPDF 'lhagrid1' reader with bilinear interpolation in (log x, log Q²)."""

    def __init__(self, path: Path) -> None:
        text = path.read_text()
        blocks = text.split("---")[1:]
        self.grids = []
        for blk in blocks:
            lines = [l for l in blk.strip().splitlines() if l.strip()]
            if len(lines) < 4:
                continue
            x = np.array(lines[0].split(), float)
            q = np.array(lines[1].split(), float)
            flav = [int(v) for v in lines[2].split()]
            vals = np.array([l.split() for l in lines[3:]], float).reshape(len(x), len(q), len(flav))
            self.grids.append((x, q, {f: k for k, f in enumerate(flav)}, vals))

    def xf(self, pid: int, x: float, Q: float) -> float:
        """x·f(x, Q) for PDG id ``pid`` (21 = gluon; negative = antiquark)."""
        x = min(max(x, 1e-9), 1.0)
        for xs, qs, fmap, vals in self.grids:
            if qs[0] <= Q <= qs[-1] or (Q < qs[0] and xs is self.grids[0][0]):
                break
        else:
            xs, qs, fmap, vals = self.grids[-1]
        if pid not in fmap:
            return 0.0
        Q = min(max(Q, qs[0]), qs[-1])
        lx, lq = np.log(xs), np.log(qs * qs)
        i = int(np.clip(np.searchsorted(lx, np.log(x)) - 1, 0, len(lx) - 2))
        j = int(np.clip(np.searchsorted(lq, np.log(Q * Q)) - 1, 0, len(lq) - 2))
        tx = (np.log(x) - lx[i]) / (lx[i + 1] - lx[i])
        tq = (np.log(Q * Q) - lq[j]) / (lq[j + 1] - lq[j])
        f = vals[:, :, fmap[pid]]
        v = ((1 - tx) * (1 - tq) * f[i, j] + tx * (1 - tq) * f[i + 1, j]
             + (1 - tx) * tq * f[i, j + 1] + tx * tq * f[i + 1, j + 1])
        return max(float(v), 0.0)

    def f(self, parton: str, x: float, Q: float) -> float:
        base = parton.rstrip("~")
        pid = PDG[base] * (-1 if parton.endswith("~") else 1)
        return self.xf(pid, x, Q) / x


@functools.lru_cache(maxsize=1)
def proton(notify=None) -> PDF:
    return PDF(ensure_downloaded(notify))
