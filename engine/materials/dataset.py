"""Training data for learned interatomic forces: aluminium in many environments, labelled by DFT.

Every label is a periodic DFT calculation from engine/crystal (energy and Hellmann–Feynman
forces). The configurations deliberately cover what molecular dynamics will visit:
compressed and stretched crystals, sheared cells, thermally jostled atoms, other crystal
structures, and disordered, liquid-like arrangements. Later rounds add configurations the
learned potential itself produces in hot dynamics (active learning).
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import numpy as np

from ..crystal.periodic import Crystal, PeriodicDFT, cubic

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "materials"
K_SPACING = 45.0    # bohr: k-mesh n ≈ K_SPACING / L along each axis (≈ 7 meV/atom converged, uniform
                    # across cells; 26 bohr left ~30 meV/atom errors that differed from cell to cell)
T_E = 0.01          # Ha, Fermi–Dirac smearing of the labels


def _kmesh(cell):
    return tuple(max(1, math.ceil(K_SPACING / L)) for L in cell)


class NotConverged(RuntimeError):
    """A DFT label whose self-consistent field did not converge."""


def _label_or_none(conf: dict, solver: str = "numpy"):
    try:
        return label(conf, solver)
    except NotConverged as e:
        print(f"refused: {e}", flush=True)
        return None


def _solver(name: str, c: Crystal):
    """The DFT path, and the provenance it leaves in the label."""
    kw = dict(h=0.3, kmesh=_kmesh(c.cell), T_e=T_E, symmetry=False)
    if name == "numpy":
        return PeriodicDFT(c, **kw), {"solver": "numpy", "precision": "float64", "device": "cpu"}
    if name == "torch":
        from ..crystal.periodic_torch import PeriodicDFTTorch
        dft = PeriodicDFTTorch(c, **kw)
        dev = dft.dev.type
        if dev == "cuda":
            import torch
            dev = f"cuda ({torch.cuda.get_device_name(dft.dev)})"
        return dft, {"solver": "torch", "precision": "complex64 eigensolver, float64 energy and force sums",
                     "device": dev}
    raise ValueError(f"unknown DFT solver {name!r}")


def label(conf: dict, solver: str = "numpy") -> dict:
    """Run DFT on one configuration (cached by content hash).

    ``solver`` is "numpy" (float64, the reference) or "torch" (engine/crystal/periodic_torch.py,
    single precision on a GPU: 0.0065 meV/atom and 1.4e-4 Ha/bohr from float64 on a production
    label, 17x faster on CUDA, slower than NumPy on Apple's MPS). Either satisfies the cache,
    since both are far inside the fit's own error; each label records which produced it. Labels
    from before this field existed were all NumPy float64."""
    key = hashlib.sha1(pickle.dumps((np.round(conf["cell"], 6).tolist(), conf["charges"],
                                     np.round(conf["positions"], 6).tolist(), K_SPACING, T_E))).hexdigest()[:16]
    path = CACHE / "dft" / f"{key}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    c = Crystal(conf["cell"], conf["charges"], conf["positions"])
    dft, provenance = _solver(solver, c)
    r = dft.run(forces=True)
    if not r.converged:
        # a non-converged SCF still returns an energy, several meV off and indistinguishable from a
        # real one; it is neither cached nor handed on
        raise NotConverged(f"{conf.get('tag', '')}: SCF not converged in {r.iterations} iterations")
    out = {"cell": c.cell, "charges": c.charges, "positions": c.positions,
           "energy": r.free_energy, "forces": r.forces, "converged": r.converged, "tag": conf.get("tag", ""),
           "kspacing": K_SPACING, "T_e": T_E, **provenance, "scf_iterations": r.iterations}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def label_all(confs, workers: int | None = None, progress=None, solver: str = "numpy") -> list[dict]:
    """Label every configuration, in parallel, skipping whatever the cache already holds.

    Each worker holds the projectors for every k-point (~1.5 GB at this mesh density), so the
    worker count is bounded by memory rather than cores, and children are recycled so that does
    not creep. multiprocessing.Pool rather than ProcessPoolExecutor: the latter's
    ``max_tasks_per_child`` deadlocked here, parent waiting on a lock while the workers sat
    blocked at startup.

    A configuration whose SCF does not converge is refused, not returned: the result holds only
    converged labels, and the count of refusals is ``len(confs) - len(result)``.

    ``solver="torch"`` labels one configuration at a time in this process: the GPU does the
    work, and every extra process would pay for its own CUDA context (~1 GB of host memory)."""
    if solver != "numpy":
        out = []
        for i, conf in enumerate(confs):
            res = _label_or_none(conf, solver)
            if res is not None:
                out.append(res)
            if progress:
                progress(i + 1, len(confs))
        return out
    n = workers or max(1, min((os.cpu_count() or 2) - 2, 5))
    ctx = mp.get_context("spawn")
    out: list[dict] = []
    with ctx.Pool(processes=n, maxtasksperchild=4) as pool:
        for i, res in enumerate(pool.imap(_label_or_none, confs)):
            if res is not None:
                out.append(res)
            if progress:
                progress(i + 1, len(confs))
    return out


def initial_configurations(Z: int, a0: float, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    confs = []

    def add(c: Crystal, tag, sigma=0.0):
        pos = c.positions + rng.normal(0, sigma, c.positions.shape)
        confs.append({"cell": c.cell, "charges": c.charges, "positions": pos, "tag": tag})

    for s in np.linspace(0.90, 1.10, 9):                       # equation of state
        add(cubic("fcc", a0 * s, Z), "fcc-volume", 0.0)
    for _ in range(24):                                          # thermal jostling at several volumes
        s = rng.uniform(0.95, 1.06)
        add(cubic("fcc", a0 * s, Z), "fcc-thermal", rng.uniform(0.05, 0.35))
    for _ in range(16):                                          # strained cells
        e = rng.uniform(-0.05, 0.05, 3)
        add(cubic("fcc", a0, Z, strain=tuple(e)), "fcc-strain", rng.uniform(0.0, 0.2))
    for _ in range(16):                                          # 8-atom cells, larger motion
        base = cubic("fcc", a0 * rng.uniform(0.97, 1.08), Z)
        cell = base.cell * np.array([2, 1, 1])
        pos = np.concatenate([base.positions, base.positions + np.array([base.cell[0], 0, 0])])
        add(Crystal(cell, [Z] * 8, pos), "fcc-8", rng.uniform(0.2, 0.55))
    for kind, n in (("bcc", 6), ("sc", 5)):                      # other structures
        for s in np.linspace(0.93, 1.08, n):
            v = a0 ** 3 / 4 * s ** 3
            a = (v * {"bcc": 2, "sc": 1}[kind]) ** (1 / 3)
            add(cubic(kind, a, Z), kind, rng.uniform(0.0, 0.1))
    for _ in range(18):                                          # disordered / liquid-like
        v_atom = a0 ** 3 / 4 * rng.uniform(1.02, 1.12)
        n = 8
        L = (v_atom * n) ** (1 / 3)
        pos = _random_packing(rng, n, L, 4.0)
        confs.append({"cell": np.array([L, L, L]), "charges": [Z] * n, "positions": pos, "tag": "disordered"})
    return confs


def md_snapshots(model, a0: float, temps=(600.0, 1000.0, 1400.0), per_T: int = 6, seed: int = 0) -> list[dict]:
    """Configurations the metal actually visits: 8-atom cells run with a learned potential
    (hot enough to disorder, then held at T), sampled every few hundred femtoseconds."""
    from ..core.units import AMU_ME, AU_TIME_FS, KELVIN_HARTREE
    from .eam import energy_forces
    rng = np.random.default_rng(seed)
    base = cubic("fcc", a0, 13)
    cell = base.cell * np.array([2, 1, 1])
    out = []
    mass = 26.9815385 * AMU_ME
    dt = 3.0 / AU_TIME_FS
    for T in temps:
        pos = np.concatenate([base.positions, base.positions + np.array([base.cell[0], 0, 0])])
        vel = rng.normal(0, math.sqrt(KELVIN_HARTREE * max(T, 1500.0 if T > 900 else T) / mass), pos.shape)
        _, F = energy_forces(model, cell, pos)
        for step in range(400 * per_T + 600):
            target = (2500.0 if step < 300 else T) if T > 900 else T
            vel += 0.5 * dt * F / mass
            pos = (pos + dt * vel) % cell
            _, F = energy_forces(model, cell, pos)
            vel += 0.5 * dt * F / mass
            vel -= vel.mean(axis=0)
            kT = mass * np.sum(vel ** 2) / (3 * (len(pos) - 1))
            vel *= math.sqrt(1 + 0.05 * (target * KELVIN_HARTREE / max(kT, 1e-12) - 1))   # weak rescaling
            if step >= 600 and (step - 600) % 400 == 399:
                out.append({"cell": cell, "charges": [13] * 8, "positions": pos.copy(), "tag": f"md-{int(T)}"})
    return out


def _random_packing(rng, n, L, dmin):
    """Random positions at least dmin apart (restarts if random placement jams)."""
    while True:
        pos, tries = [], 0
        while len(pos) < n and tries < 20000:
            tries += 1
            p = rng.uniform(0, L, 3)
            if all(np.linalg.norm(((p - q) + L / 2) % L - L / 2) > dmin for q in pos):
                pos.append(p)
        if len(pos) == n:
            return np.array(pos)
