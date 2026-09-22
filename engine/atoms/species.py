"""How each element enters the 3D engine.

H and He have no core electrons and are simulated all-electron with exact
point nuclei. Li and heavier use pseudopotentials generated on first use from
this engine's own all-electron atom solver (a few seconds each), then cached
on disk so later launches are instant.
"""

from __future__ import annotations

import functools
import pickle
from pathlib import Path

from .pseudo import Pseudopotential, generate

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pseudo"
CACHE_VERSION = 2


def is_pseudized(Z: int) -> bool:
    return Z >= 3


@functools.lru_cache(maxsize=None)
def pseudopotential(Z: int) -> Pseudopotential:
    path = CACHE_DIR / f"v{CACHE_VERSION}_Z{Z}.pkl"
    if path.exists():
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except Exception:
            path.unlink(missing_ok=True)
    pp = generate(Z)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(pp, f)
    return pp


def valence_charge(Z: int) -> float:
    return pseudopotential(Z).Z_val if is_pseudized(Z) else float(Z)
