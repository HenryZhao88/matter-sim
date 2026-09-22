"""Starting states. Each is only nuclei (Z, position), net charge and spin.

Positions are in Å here for readability and converted to bohr on load. The
starting geometries are deliberately *not* the known answers: the point is to
watch the solver find the structure.
"""

from __future__ import annotations

from ..core.units import angstrom_to_bohr
from ..system import System

PRESETS: list[dict] = [
    {
        "id": "h", "name": "Hydrogen atom", "formula": "H",
        "blurb": "One proton, one electron. The ground state is exact here.",
        "atoms": [(1, (0, 0, 0))], "reference": {"energy_ha": -0.5, "note": "exact non-relativistic"},
        "mode": "frozen",
    },
    {
        "id": "he", "name": "Helium atom", "formula": "He",
        "blurb": "Two electrons share the 1s shell with opposite spins.",
        "atoms": [(2, (0, 0, 0))], "reference": {"energy_ha": -2.90372, "note": "exact non-relativistic"},
        "mode": "frozen",
    },
    {
        "id": "h2_form", "name": "Two hydrogen atoms", "formula": "H + H",
        "blurb": "Released 1.6 Å apart with opposite spins. Watch them find a bond.",
        "atoms": [(1, (-0.8, 0, 0)), (1, (0.8, 0, 0))], "reference": {"distances_A": [0.7414], "binding_ev": 4.747, "note": "experiment",
                      "method_note": "LDA itself predicts about 0.765 Å; the rest of the gap is the approximation, not the grid."},
        "mode": "relax",
    },
    {
        "id": "h2_triplet", "name": "Two H, parallel spins", "formula": "H↑ + H↑",
        "blurb": "Same atoms, both spins up. Pauli exclusion forbids the bond — they repel.",
        "atoms": [(1, (-0.6, 0, 0)), (1, (0.6, 0, 0))], "multiplicity": 3, "reference": {"note": "unbound: no minimum exists"},
        "mode": "relax",
    },
    {
        "id": "h2_vibrate", "name": "Vibrating H₂", "formula": "H₂",
        "blurb": "A compressed bond released in real time: a molecular vibration.",
        "atoms": [(1, (-0.3, 0, 0)), (1, (0.3, 0, 0))], "reference": {"distances_A": [0.7414], "period_fs": 7.58, "note": "experiment, vibration period 7.6 fs"},
        "mode": "dynamics",
    },
    {
        "id": "h2plus", "name": "Hydrogen molecular ion", "formula": "H₂⁺",
        "blurb": "A single electron holding two protons together.",
        "atoms": [(1, (-0.7, 0, 0)), (1, (0.7, 0, 0))], "charge": 1, "reference": {"distances_A": [1.052], "note": "exact"},
        "mode": "relax",
    },
    {
        "id": "h3plus", "name": "Trihydrogen cation", "formula": "H₃⁺",
        "blurb": "Three protons and two electrons, started as a bent chain. What shape wins?",
        "atoms": [(1, (-0.85, -0.3, 0)), (1, (0, 0.25, 0)), (1, (0.8, -0.35, 0.05))], "charge": 1,
        "reference": {"distances_A": [0.873, 0.873, 0.873], "angles_deg": [60.0], "note": "experiment: equilateral triangle"},
        "mode": "relax",
    },
    {
        "id": "heh", "name": "Helium hydride ion", "formula": "HeH⁺",
        "blurb": "The first molecule to form in the early universe.",
        "atoms": [(2, (-0.45, 0, 0)), (1, (0.6, 0, 0))], "charge": 1, "reference": {"distances_A": [0.772], "note": "experiment"},
        "mode": "relax",
    },
]


def preset(pid: str) -> dict:
    for p in PRESETS:
        if p["id"] == pid:
            return p
    raise KeyError(pid)


def system_from_preset(pid: str) -> System:
    p = preset(pid)
    return System(
        [Z for Z, _ in p["atoms"]],
        [[angstrom_to_bohr(c) for c in pos] for _, pos in p["atoms"]],
        charge=p.get("charge", 0),
        multiplicity=p.get("multiplicity"),
    )


def catalog() -> list[dict]:
    return [{k: p[k] for k in ("id", "name", "formula", "blurb", "mode", "reference")} for p in PRESETS]
