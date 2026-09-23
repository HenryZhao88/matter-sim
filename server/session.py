"""Engine thread: owns the Simulation, applies commands, publishes state.

All physics runs on one background thread so the asyncio server stays
responsive. Commands arrive through a queue and are applied between steps.
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Callable

import numpy as np

from engine.core.backend import get_backend
from engine.core.elements import ELEMENTS
from engine.core.grid import Grid
from engine.electrons.scf import SCFSolver
from engine.scenes.presets import catalog, preset, system_from_preset
from engine.simulation import MODES, QUALITY_H, Params, Simulation, supported_elements
from engine.system import System

from engine.core.accel import describe, have_mlx
from .protocol import encode_snapshot, sanitize

DEFAULT_PRESET = "water"


class Session:
    def __init__(self, publish_json: Callable[[dict], None], publish_bytes: Callable[[bytes], None]):
        self._pub_json = publish_json
        self._pub_bytes = publish_bytes
        self._cmds: queue.Queue = queue.Queue()
        self.running = True
        self.busy = False
        self.preset_id = DEFAULT_PRESET
        self.latest_snapshot: bytes | None = None
        self.sim: Simulation | None = None
        self._last_progress = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine", daemon=True)

    # ------------------------------------------------------------ public
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, cmd: dict) -> None:
        self._cmds.put(cmd)

    def hello(self) -> dict:
        return {
            "type": "hello",
            "presets": catalog(),
            "elements": [
                {"Z": e.Z, "symbol": e.symbol, "name": e.name, "color": e.color,
                 "available": e.Z in supported_elements()}
                for e in ELEMENTS.values()
            ],
            "modes": list(MODES),
            "qualities": QUALITY_H,
            "functionals": ["lda", "hf", "none"],
            "backends": ["mlx", "numpy"] if have_mlx() else ["numpy"],
            "accelerator": describe(),
        }

    def status(self) -> dict:
        s = self.sim
        return sanitize({
            "type": "status",
            "running": self.running,
            "busy": self.busy,
            "preset": self.preset_id,
            "idle": bool(s and s.is_idle),
            "params": s.params.__dict__ if s else None,
        })

    # ------------------------------------------------------------- loop
    def _loop(self) -> None:
        self.sim = Simulation(system_from_preset(DEFAULT_PRESET),
                              Params(mode=preset(DEFAULT_PRESET)["mode"]))
        self.sim.on_scf_progress = self._progress
        self._pub_json(self.status())
        while not self._stop.is_set():
            try:
                while True:
                    self._apply(self._cmds.get_nowait())
            except queue.Empty:
                pass
            if self.running and not self.sim.is_idle:
                self._guarded(self._step)
            else:
                try:
                    self._apply(self._cmds.get(timeout=0.1))
                except queue.Empty:
                    pass

    def _guarded(self, fn, *args) -> None:
        self.busy = True
        try:
            fn(*args)
        except Exception as exc:  # report, never kill the engine thread
            traceback.print_exc()
            self.running = False
            self._pub_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.busy = False
            self._pub_json(self.status())

    def _step(self) -> None:
        was_idle = self.sim.is_idle
        self.sim.step()
        self._publish_snapshot()
        if self.sim.is_idle and not was_idle:
            msg = "Relaxed: forces below threshold." if self.sim.params.mode == "relax" else "Converged."
            self._pub_json({"type": "log", "level": "info", "message": msg})

    def _publish_snapshot(self) -> None:
        snap = self.sim.snapshot()
        snap["meta"] = sanitize(snap["meta"])
        data = encode_snapshot(snap)
        self.latest_snapshot = data
        self._pub_bytes(data)

    def _progress(self, row: dict) -> None:
        now = time.perf_counter()
        if now - self._last_progress > 0.1:
            self._last_progress = now
            self._pub_json(sanitize({"type": "scf_progress", **row}))

    # --------------------------------------------------------- commands
    def _apply(self, cmd: dict) -> None:
        self._guarded(self._dispatch, cmd)

    def _prepare_species(self, charges) -> None:
        """Tell the viewer when a pseudopotential is being derived (first use only)."""
        from engine.atoms import species
        for Z in sorted(set(charges)):
            if species.is_pseudized(Z) and not (species.CACHE_DIR / f"v{species.CACHE_VERSION}_Z{Z}.pkl").exists():
                self._pub_json({"type": "log", "level": "info",
                                "message": f"Deriving the {ELEMENTS[Z].name.lower()} pseudopotential from its "
                                           "all-electron atom. This happens once."})
                species.pseudopotential(Z)

    def _dispatch(self, cmd: dict) -> None:
        t = cmd.get("type")
        sim = self.sim
        if t == "run":
            self.running = True
        elif t == "pause":
            self.running = False
        elif t == "step":
            self.running = False
            self._step()
        elif t == "load_preset":
            p = preset(cmd["id"])
            self._prepare_species([Z for Z, _ in p["atoms"]])
            self.preset_id = p["id"]
            sim.params.mode = p["mode"]
            sim.set_system(system_from_preset(p["id"]))
            self.running = True
        elif t == "load_scene":
            self.preset_id = None
            atoms = cmd["atoms"]
            sim.set_system(System([a["Z"] for a in atoms], [a["pos"] for a in atoms],
                                  charge=int(cmd.get("charge", 0)),
                                  multiplicity=cmd.get("multiplicity")))
        elif t == "add_atom":
            self._edit_atoms(add=(int(cmd["Z"]), cmd.get("pos")))
        elif t == "remove_atom":
            self._edit_atoms(remove=int(cmd["index"]))
        elif t == "move_atom":
            P = sim.system.positions.copy()
            P[int(cmd["index"])] = cmd["pos"]
            sim.move_nuclei(P)
            sim.system.velocities[int(cmd["index"])] = 0.0
            if not self.running:
                self._step()
        elif t == "set_spin":
            s = sim.system
            sim.set_system(System(s.charges, s.positions, s.charge, int(cmd["multiplicity"])))
        elif t == "set_charge":
            s = sim.system
            sim.set_system(System(s.charges, s.positions, int(cmd["charge"]), None))
        elif t == "set_params":
            sim.set_params(**cmd["params"])
            if not self.running:
                self._step()
        elif t == "find_spin":
            self._pub_json({"type": "log", "level": "info", "message": "Trying every total spin at this geometry…"})
            rows = sim.find_spin()
            best = min(rows, key=lambda r: r["energy"])
            self._pub_json(sanitize({"type": "spin_scan", "rows": rows, "best": best["multiplicity"]}))
            self._step()
        elif t == "verify":
            self._verify()
        elif t == "truth":
            self._truth()
        elif t == "snapshot":
            if self.latest_snapshot:
                self._pub_bytes(self.latest_snapshot)
        else:
            raise ValueError(f"unknown command {t!r}")

    def _edit_atoms(self, add=None, remove=None) -> None:
        s = self.sim.system
        Z, P = list(s.charges), [list(p) for p in s.positions]
        if add is not None:
            z, pos = add
            self._prepare_species([z])
            if z not in supported_elements():
                raise ValueError(f"{ELEMENTS[z].symbol} needs a pseudopotential that is not built yet")
            if pos is None:
                # Place beside the existing nuclei, 1.5 Å from the nearest one.
                c = np.mean(P, axis=0) if P else np.zeros(3)
                far = max((np.linalg.norm(np.array(p) - c) for p in P), default=0.0)
                pos = (c + np.array([far + 2.8, 0.0, 0.0])).tolist()
            Z.append(z)
            P.append(list(pos))
        if remove is not None:
            if len(Z) <= 1:
                raise ValueError("cannot remove the last nucleus")
            Z.pop(remove)
            P.pop(remove)
        charge = min(s.charge, sum(Z) - 1) if sum(Z) > 1 else 0
        self.preset_id = None
        self.sim.set_system(System(Z, P, charge=max(charge, 0)))

    def _truth(self) -> None:
        """Truth mode: every electron, the exact Hamiltonian, a neural-network wavefunction (VMC)."""
        if not have_mlx():
            raise RuntimeError("Truth mode needs the MLX GPU backend (Apple silicon); "
                               "a CUDA version is not written yet")
        from engine.truth.vmc import VMC, Molecule
        s = self.sim.system
        if s.n_up > 3 or s.n_dn > 3:
            raise ValueError("Truth mode handles up to three electrons of each spin for now (H₂, He, Li, LiH, Be…)")
        mol = Molecule([float(z) for z in s.charges], np.asarray(s.positions, float),
                       int(round(s.n_up)), int(round(s.n_dn)))
        self._pub_json({"type": "log", "level": "info",
                        "message": "Truth mode: training a neural-network wavefunction on the exact Hamiltonian…"})
        t0 = time.perf_counter()
        v = VMC(mol, walkers=512, hidden=24, layers=2, dets=2)
        v.train(iters=400, log=lambda it, e, sec: self._pub_json(sanitize(
            {"type": "truth_progress", "iter": it, "iters": 400, "energy": e})))
        E, err = v.evaluate(12, 5)
        all_electron = all(z <= 2 for z in s.charges)
        dft = self.sim.result.free_energy if (self.sim.result is not None and all_electron) else None
        self._pub_json(sanitize({"type": "truth_result", "energy": E, "error": err, "dft_energy": dft,
                                 "functional": self.sim.params.functional, "seconds": time.perf_counter() - t0}))

    def _verify(self) -> None:
        """Re-solve the current geometry in float64 on the CPU and compare."""
        sim = self.sim
        if sim.result is None:
            raise RuntimeError("nothing to verify yet")
        self._pub_json({"type": "log", "level": "info", "message": "Verifying in float64 on the CPU…"})
        t0 = time.perf_counter()
        g = Grid(sim.grid.L, sim.params.h, get_backend("numpy"))
        solver = SCFSolver(g, sim.system.copy(), functional=sim.params.functional, T_e=sim.params.T_e)
        res = solver.run()
        F64 = solver.forces()
        dE = sim.result.free_energy - res.free_energy
        dF = float(np.max(np.abs(sim.forces - F64))) if len(F64) else 0.0
        self._pub_json(sanitize({
            "type": "verify_result",
            "energy_f32": sim.result.free_energy, "energy_f64": res.free_energy,
            "delta_energy": dE, "max_delta_force": dF, "seconds": time.perf_counter() - t0,
        }))
