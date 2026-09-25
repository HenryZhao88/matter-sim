"""How each element enters the 3D engine.

H and He have no core electrons and are simulated all-electron with exact
point nuclei. Li and heavier use pseudopotentials generated on first use from
this engine's own all-electron atom solver (a few seconds each), then cached
on disk so later launches are instant.
"""

from __future__ import annotations

import functools
import json
import pickle

import numpy as np
from pathlib import Path

from .pseudo import GHOST_TOL, Pseudopotential, generate, ghost_check, verify, worst_transfer

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pseudo"
CACHE_VERSION = 5
TRANSFER_TOL_EV = 0.15      # largest allowed AE–PS difference in excitation energies


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
    _record_checks(pp)
    return pp


def _checks_path(Z: int) -> Path:
    return CACHE_DIR / f"v{CACHE_VERSION}_Z{Z}_checks.json"


def _record_checks(pp: Pseudopotential) -> dict:
    """Ghost states and transferability of a freshly generated pseudopotential, kept on disk."""
    ghosts = ghost_check(pp)
    worst = worst_transfer(verify(pp)) * 27.211386
    rec = {"ghost_free": bool(all(abs(a - b) < GHOST_TOL for a, b in ghosts.values())),
           "transfer_eV": float(worst) if np.isfinite(worst) else None, "l_local": int(pp.l_local), "Z_val": float(pp.Z_val)}
    rec["ok"] = bool(rec["ghost_free"] and np.isfinite(worst) and worst < TRANSFER_TOL_EV)
    _checks_path(pp.Z).write_text(json.dumps(rec))
    return rec


def checks() -> dict:
    out = {}
    for f in CACHE_DIR.glob(f"v{CACHE_VERSION}_Z*_checks.json"):
        try:
            out[f.name.split("_")[1][1:]] = json.loads(f.read_text())
        except (OSError, ValueError):
            pass
    return out


def passes_checks(Z: int) -> bool:
    """Li–Ar were validated against molecules; beyond that, an element is offered once its
    generated pseudopotential has passed the ghost and transferability checks."""
    if Z <= 18:
        return True
    rec = checks().get(str(Z))
    return bool(rec and rec["ok"])


def valence_charge(Z: int) -> float:
    return pseudopotential(Z).Z_val if is_pseudized(Z) else float(Z)
