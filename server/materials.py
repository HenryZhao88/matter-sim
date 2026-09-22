"""Materials worker: many-atom molecular dynamics with the learned potential, and the
continuum block built from what it measures.

Commands:
  matter.hello                          → matter.hello {ready, chain, block, reference}
  matter.md    {T, P_GPa, cells, start} → streams matter.frame, then matter.done
  matter.stop                           → cancels the running dynamics
  matter.block {T, P_GPa, stress_MPa}   → matter.block (the 1 cm³ block under those conditions)
"""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from .protocol import sanitize

CACHE = Path(__file__).resolve().parents[1] / ".cache" / "materials"
BOHR_A = 0.529177210903
HA_EV = 27.211386245988


class MaterialsWorker:
    def __init__(self, publish_json: Callable[[dict], None]) -> None:
        self._pub = publish_json
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="materials", daemon=True)
        self._model = None

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
            while not self._q.empty():
                cmd = self._q.get_nowait()
            try:
                kind = cmd["type"]
                if kind == "matter.hello":
                    self._hello()
                elif kind == "matter.md":
                    self._md(cmd)
                elif kind == "matter.block":
                    self._block(cmd)
            except Exception as exc:
                traceback.print_exc()
                self._pub({"type": "error", "message": f"Materials: {type(exc).__name__}: {exc}"})

    def _cancelled(self) -> bool:
        return not self._q.empty() or self._stop.is_set()

    # ---------------------------------------------------------------- state
    def _load_model(self):
        if self._model is None:
            from engine.materials.eam import EAM
            path = max(CACHE.glob("al_eam_round*.npz"), default=None, key=lambda p: p.name)
            if path is None:
                return None
            self._model = EAM.load(path)
        return self._model

    def _chain(self) -> dict:
        out = {}
        for name in ("al_crystal.json", "al_results.json", "al_eam_errors.json"):
            try:
                out[name.split(".")[0]] = json.loads((CACHE / name).read_text())
            except (OSError, ValueError):
                pass
        return out

    def _hello(self) -> None:
        from engine.materials.continuum import REFERENCE, Block, derived_aluminium
        d = derived_aluminium()
        self._pub(sanitize({
            "type": "matter.hello", "ready": self._load_model() is not None, "chain": self._chain(),
            "block": Block(d).summary() if d else None,
            "reference": {k: v[0] for k, v in REFERENCE.items()},
        }))

    def _block(self, cmd: dict) -> None:
        from engine.materials.continuum import Block, derived_aluminium
        d = derived_aluminium()
        if d is None:
            self._pub({"type": "matter.block", "available": False})
            return
        T = float(cmd.get("T", 293.15))
        b = Block(d, T=293.15)
        P = float(cmd.get("P_GPa", 0.0))
        stress = float(cmd.get("stress_MPa", 0.0))
        hot = Block(d, T=min(T, d.T_melt))
        self._pub(sanitize({
            "type": "matter.block", "available": True, "T": T, "P_GPa": P, "stress_MPa": stress,
            "side_cm": b.side_at_cm(min(T, d.T_melt)) * b.squeeze(P)["side_cm"],
            "density": hot.density(), "heat_J": b.heat_to_warm_J(293.15, min(T, d.T_melt)) + (
                d.latent_eV * 1.602176634e-19 * b.atoms() if T >= d.T_melt else 0.0),
            "melted": T >= d.T_melt, "stretch": b.stretch(stress), "squeeze": b.squeeze(P),
            "summary": b.summary(),
        }))

    # ---------------------------------------------------------------- dynamics
    def _md(self, cmd: dict) -> None:
        from engine.materials.md import MD, al_state, temperature, pressure, HA_PER_BOHR3_GPA, kinetic
        from engine.core.units import AU_TIME_FS
        model = self._load_model()
        if model is None:
            self._pub({"type": "matter.done", "error": "The learned potential has not been trained yet."})
            return
        T = float(np.clip(cmd.get("T", 300.0), 10.0, 3000.0))
        P = float(np.clip(cmd.get("P_GPa", 0.0), -5.0, 50.0))
        cells = int(np.clip(cmd.get("cells", 5), 3, 7))
        start = cmd.get("start", "crystal")
        a0 = 4.0 / BOHR_A
        state = al_state(model, a0, (cells, cells, cells))
        md = MD(model, state, seed=int(time.time()) % 1000)
        N = len(state.pos)
        md.thermalise(2500.0 if start == "liquid" else T)
        self._pub(sanitize({"type": "matter.start", "N": N, "T": T, "P_GPa": P, "start": start}))
        t_fs, step, last = 0.0, 0, 0.0
        melt_steps = 1500 if start == "liquid" else 0
        while step < 40000:
            if self._cancelled():
                self._pub({"type": "matter.done", "cancelled": True})
                return
            target = 2500.0 if step < melt_steps else T
            for _ in range(10):
                md.step(target, P)
                step += 1
            t_fs += 10 * md.dt * AU_TIME_FS
            now = time.perf_counter()
            if now - last > 0.1:
                last = now
                s = md.s
                self._pub(sanitize({
                    "type": "matter.frame", "step": step, "t_ps": t_fs / 1000,
                    "pos": np.round((s.pos % s.box) * BOHR_A, 3).ravel().tolist(),
                    "box": (s.box * BOHR_A).tolist(), "T": temperature(s),
                    "P_GPa": pressure(s, md.W) * HA_PER_BOHR3_GPA,
                    "U_eV": md.E / N * HA_EV, "E_eV": (md.E + kinetic(s)) / N * HA_EV,
                    "a_A": float((np.prod(s.box) / (N / 4)) ** (1 / 3) * BOHR_A),
                    "melting_in": max(0, melt_steps - step),
                }))
        self._pub({"type": "matter.done"})
