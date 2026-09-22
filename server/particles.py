"""Collider worker: runs the Standard Model rung on its own thread.

Commands (JSON, "type" prefixed with "collider."):
  collider.hello                         → beams, energies, particle metadata
  collider.select   {beams, sqrt_s}      → what these beams can make at this energy
  collider.collide  {beams, sqrt_s, n}   → n generated events
  collider.scan     {beams}              → σ vs √s, streamed row by row
  collider.particle {name}               → mass, lifetime, decay channels
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from .protocol import sanitize

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "collider"

BEAMS = [
    {"id": "ee", "pair": ["e-", "e+"], "label": "Electron + positron",
     "note": "Clean collisions: nothing inside the beams but the particles themselves."},
    {"id": "mumu", "pair": ["mu-", "mu+"], "label": "Muon + antimuon",
     "note": "Heavier electrons. They talk to the Higgs 200 times more strongly."},
    {"id": "aa", "pair": ["photon", "photon"], "label": "Photon + photon",
     "note": "Light on light. Photons cannot touch each other directly; watch what they make anyway."},
    {"id": "uu", "pair": ["u", "u~"], "label": "Up quark + anti-up",
     "note": "Quarks collide through all three forces. Slow to compute the first time."},
    {"id": "gg", "pair": ["g", "g"], "label": "Gluon + gluon",
     "note": "Carriers of the strong force colliding. Minutes to compute the first time."},
]
ENERGIES = [10.0, 50.0, 91.19, 125.0, 160.0, 200.0, 250.0, 350.0, 500.0, 1000.0]
SCAN = [5, 10, 20, 35, 50, 65, 80, 86, 88, 89.5, 90.5, 91.19, 92, 93.5, 96, 100, 120, 150, 161,
        170, 185, 200, 250, 300, 350, 400, 500, 700, 1000]

DISPLAY = {  # symbol, family (display only)
    "e-": ("e⁻", "lepton"), "e+": ("e⁺", "lepton"), "mu-": ("μ⁻", "lepton"), "mu+": ("μ⁺", "lepton"),
    "tau-": ("τ⁻", "lepton"), "tau+": ("τ⁺", "lepton"),
    "nu_e": ("νₑ", "neutrino"), "nu_e~": ("ν̄ₑ", "neutrino"), "nu_mu": ("ν_μ", "neutrino"),
    "nu_mu~": ("ν̄_μ", "neutrino"), "nu_tau": ("ν_τ", "neutrino"), "nu_tau~": ("ν̄_τ", "neutrino"),
    "photon": ("γ", "photon"), "g": ("g", "gluon"), "Z": ("Z", "boson"), "W+": ("W⁺", "boson"),
    "W-": ("W⁻", "boson"), "H": ("H", "higgs"),
}
for q in ("u", "d", "c", "s", "t", "b"):
    DISPLAY[q] = (q, "quark")
    DISPLAY[q + "~"] = (q + "̄", "quark")


class ColliderWorker:
    def __init__(self, publish_json: Callable[[dict], None]) -> None:
        self._pub = publish_json
        self._q: queue.Queue = queue.Queue()
        self._rng = np.random.default_rng()
        self._thread = threading.Thread(target=self._loop, name="collider", daemon=True)
        self._stop = threading.Event()
        self._warm = False

    def start(self) -> None:
        self._thread.start()
        self._q.put({"type": "collider.warmup"})   # derive decays in the background at launch

    def stop(self) -> None:
        self._stop.set()

    def submit(self, cmd: dict) -> None:
        self._q.put(cmd)

    # ---------------------------------------------------------------- loop
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                cmd = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._dispatch(cmd)
            except Exception as exc:
                traceback.print_exc()
                self._pub({"type": "error", "message": f"Collider: {type(exc).__name__}: {exc}"})

    def _warmup(self) -> None:
        if self._warm:
            return
        from engine.particles.decays import warm_cache
        first = not (CACHE / "decays.pkl").exists()
        self._say("Deriving the Standard Model from its Lagrangian and computing every decay"
                  + (" (a minute or two, once)…" if first else "…"))
        warm_cache(CACHE / "decays.pkl")
        self._warm = True

    def _say(self, msg: str) -> None:
        self._pub({"type": "collider.progress", "message": msg})

    def _dispatch(self, cmd: dict) -> None:
        t = cmd["type"]
        if t == "collider.warmup":
            self._warmup()
            self._pub({"type": "collider.progress", "message": "Standard Model ready."})
        elif t == "collider.hello":
            self._pub(self._hello())
        elif t == "collider.select":
            self._warmup()
            a, b = cmd["beams"]
            E = float(cmd["sqrt_s"])
            rows = self._outcomes(a, b, E)
            tot = sum(r[2] for r in rows)
            self._pub(sanitize({
                "type": "collider.outcomes", "beams": [a, b], "sqrt_s": E, "total_pb": tot,
                "rows": [{"final": [r[0], r[1]], "pb": r[2], "share": r[2] / tot if tot else 0} for r in rows],
            }))
        elif t == "collider.collide":
            self._warmup()
            from engine.particles.events import generate
            a, b = cmd["beams"]
            E = float(cmd["sqrt_s"])
            self._outcomes(a, b, E)
            n = max(1, min(int(cmd.get("n", 1)), 200))
            events = [generate(a, b, E, self._rng) for _ in range(n)]
            self._pub(sanitize({"type": "collider.events", "beams": [a, b], "sqrt_s": E, "events": events}))
        elif t == "collider.scan":
            self._warmup()
            a, b = cmd["beams"]
            for E in SCAN:
                rows = self._outcomes(a, b, float(E), quiet=True)
                top = rows[:4]
                self._pub(sanitize({"type": "collider.scan_row", "beams": [a, b], "sqrt_s": float(E),
                                    "total_pb": sum(r[2] for r in rows),
                                    "top": [{"final": [r[0], r[1]], "pb": r[2]} for r in top]}))
            self._pub({"type": "collider.scan_done", "beams": [a, b]})
        elif t == "collider.particle":
            self._warmup()
            self._pub(sanitize(self._particle(cmd["name"])))
        else:
            raise ValueError(f"unknown collider command {t}")

    # ---------------------------------------------------------------- pieces
    def _outcomes(self, a: str, b: str, E: float, quiet: bool = False):
        from engine.particles import events
        key = f"{a}_{b}_{E:.3f}".replace("~", "bar").replace("+", "p").replace("-", "m")
        path = CACHE / f"{key}.pkl"
        if path.exists():
            rows = pickle.loads(path.read_bytes())
            events._TABLES[(a, b, E)] = rows
            return rows
        if not quiet:
            self._say(f"Trying every final state these beams could make at {E:g} GeV…")
        t0 = time.perf_counter()
        rows = events.outcomes(a, b, E)
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(rows))
        if not quiet:
            self._say(f"Done in {time.perf_counter() - t0:.0f} s.")
        return rows

    def _hello(self) -> dict:
        from engine.particles.process import registry
        parts = {}
        for name, (sp, _) in registry().items():
            sym, fam = DISPLAY.get(name, (name, "other"))
            parts[name] = {"symbol": sym, "family": fam, "mass": sp.mass, "charge": sp.charge,
                           "colour": sp.colour}
        return sanitize({"type": "collider.hello", "beams": BEAMS, "energies": ENERGIES, "particles": parts})

    def _particle(self, name: str) -> dict:
        from engine.particles.decays import UNSTABLE, branching_ratios, lifetime_seconds, total_width
        from engine.particles.events import is_confined
        from engine.particles.process import species
        sp = species(name)
        info = {"type": "collider.particle", "name": name, "mass": sp.mass, "charge": sp.charge,
                "confined": bool(sp.colour > 1 and is_confined(name))}
        if name in UNSTABLE:
            info |= {"width": total_width(name), "lifetime_s": lifetime_seconds(name),
                     "decays": [{"products": list(p), "br": b} for p, b in branching_ratios(name)[:8]]}
        return info
