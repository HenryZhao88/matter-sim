"""Lattice worker: the 1D real-time universe and lattice QCD, on their own thread.

Commands:
  lattice.run   {scenario, N, mass, strength, t_max, frames}  → streams lattice.frame, then lattice.done
  qcd.run       {beta, L, T, sweeps}                         → streams qcd.progress, then qcd.result
A new command cancels whatever is running.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable

import numpy as np

from .protocol import sanitize


class LatticeWorker:
    def __init__(self, publish_json: Callable[[dict], None]) -> None:
        self._pub = publish_json
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lattice", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, cmd: dict) -> None:
        self._q.put(cmd)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                cmd = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            while not self._q.empty():          # only the latest request matters
                cmd = self._q.get_nowait()
            try:
                if cmd["type"] == "lattice.run":
                    self._run_schwinger(cmd)
                elif cmd["type"] == "qcd.run":
                    self._run_qcd(cmd)
                elif cmd["type"] == "lattice.hello":
                    from engine.lattice.schwinger import SCENARIOS
                    self._pub({"type": "lattice.hello", "scenarios": SCENARIOS})
            except Exception as exc:
                traceback.print_exc()
                self._pub({"type": "error", "message": f"Lattice: {type(exc).__name__}: {exc}"})

    def _cancelled(self) -> bool:
        return not self._q.empty() or self._stop.is_set()

    # ---------------------------------------------------------------- Schwinger model
    def _run_schwinger(self, cmd: dict) -> None:
        from engine.lattice.schwinger import run_scenario, vacuum_profile
        kind = cmd.get("scenario", "pair_creation")
        N = int(cmd.get("N", 18))
        mass = float(cmd.get("mass", 0.4))
        strength = float(cmd.get("strength", 1.0))
        frames = int(cmd.get("frames", 81))
        t_max = float(cmd.get("t_max", 10.0))
        a = float(cmd.get("a", 0.5))
        t0 = time.perf_counter()
        vac = vacuum_profile(N, mass, a)
        model, it = run_scenario(kind, N=N, mass=mass, t_max=t_max, frames=frames, strength=strength, a=a)
        self._pub(sanitize({"type": "lattice.start", "scenario": kind, "N": N, "mass": mass, "dim": model.dim,
                            "frames": frames, "t_max": t_max, "vacuum_particles": vac["particles"]}))
        e0 = None
        for t, obs in it:
            if self._cancelled():
                self._pub({"type": "lattice.done", "cancelled": True})
                return
            e0 = obs["energy"] if e0 is None else e0
            self._pub(sanitize({
                "type": "lattice.frame", "t": t,
                "number": (np.asarray(obs["number"]) - np.asarray(vac["number"])).tolist(),
                "charge": (np.asarray(obs["charge"]) - np.asarray(vac["charge"])).tolist(),
                "field": np.asarray(obs["field"]).tolist(),
                "particles": obs["particles"] - vac["particles"],
                "energy_drift": obs["energy"] - e0,
            }))
        self._pub({"type": "lattice.done", "seconds": time.perf_counter() - t0})

    # ---------------------------------------------------------------- lattice QCD
    def _run_qcd(self, cmd: dict) -> None:
        from engine.lattice.qcd import polyakov_scan, run_measurement
        mode = cmd.get("mode", "confinement")
        L = int(cmd.get("L", 8))
        if mode == "confinement":
            for msg in run_measurement(float(cmd.get("beta", 5.7)), L, L, int(cmd.get("sweeps", 90)),
                                       cancelled=self._cancelled):
                self._pub(sanitize(msg))
            return
        betas = cmd.get("betas", [5.45, 5.55, 5.62, 5.67, 5.71, 5.75, 5.8, 5.9, 6.05, 6.2])
        self._pub({"type": "qcd.scan_start", "betas": betas, "L": L, "Nt": 4})
        for beta in betas:
            if self._cancelled():
                self._pub({"type": "qcd.done", "cancelled": True})
                return
            (b, p, e), = polyakov_scan([beta], L=L, Nt=4, sweeps=int(cmd.get("sweeps", 120)))
            self._pub(sanitize({"type": "qcd.scan_row", "beta": b, "polyakov": p, "error": e}))
        self._pub({"type": "qcd.done"})
