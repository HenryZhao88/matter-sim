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
        "id": "h", "group": "Atoms", "name": "Hydrogen atom", "formula": "H",
        "blurb": "One proton, one electron. The ground state is exact here.",
        "atoms": [(1, (0, 0, 0))], "reference": {"energy_ha": -0.5, "note": "exact non-relativistic"},
        "mode": "frozen",
    },
    {
        "id": "he", "group": "Atoms", "name": "Helium atom", "formula": "He",
        "blurb": "Two electrons share the 1s shell with opposite spins.",
        "atoms": [(2, (0, 0, 0))], "reference": {"energy_ha": -2.90372, "note": "exact non-relativistic"},
        "mode": "frozen",
    },
    {
        "id": "h2_form", "group": "Hydrogen and helium", "name": "Two hydrogen atoms", "formula": "H + H",
        "blurb": "Released 1.6 Å apart with opposite spins. Watch them find a bond.",
        "atoms": [(1, (-0.8, 0, 0)), (1, (0.8, 0, 0))], "reference": {"distances_A": [0.7414], "binding_ev": 4.747, "note": "experiment",
                      "method_note": "LDA itself predicts about 0.765 Å; the rest of the gap is the approximation, not the grid."},
        "mode": "relax",
    },
    {
        "id": "h2_triplet", "group": "Hydrogen and helium", "name": "Two H, parallel spins", "formula": "H↑ + H↑",
        "blurb": "Same atoms, both spins up. Pauli exclusion forbids the bond — they repel.",
        "atoms": [(1, (-0.6, 0, 0)), (1, (0.6, 0, 0))], "multiplicity": 3, "reference": {"note": "unbound: no minimum exists"},
        "mode": "relax",
    },
    {
        "id": "h2_vibrate", "group": "Hydrogen and helium", "name": "Vibrating H₂", "formula": "H₂",
        "blurb": "A compressed bond released in real time: a molecular vibration.",
        "atoms": [(1, (-0.3, 0, 0)), (1, (0.3, 0, 0))], "reference": {"distances_A": [0.7414], "period_fs": 7.58, "note": "experiment, vibration period 7.6 fs"},
        "mode": "dynamics",
    },
    {
        "id": "h2plus", "group": "Hydrogen and helium", "name": "Hydrogen molecular ion", "formula": "H₂⁺",
        "blurb": "A single electron holding two protons together.",
        "atoms": [(1, (-0.7, 0, 0)), (1, (0.7, 0, 0))], "charge": 1, "reference": {"distances_A": [1.052], "note": "exact"},
        "mode": "relax",
    },
    {
        "id": "h3plus", "group": "Hydrogen and helium", "name": "Trihydrogen cation", "formula": "H₃⁺",
        "blurb": "Three protons and two electrons, started as a bent chain. What shape wins?",
        "atoms": [(1, (-0.85, -0.3, 0)), (1, (0, 0.25, 0)), (1, (0.8, -0.35, 0.05))], "charge": 1,
        "reference": {"distances_A": [0.873, 0.873, 0.873], "angles_deg": [60.0], "note": "experiment: equilateral triangle"},
        "mode": "relax",
    },
    {
        "id": "heh", "group": "Hydrogen and helium", "name": "Helium hydride ion", "formula": "HeH⁺",
        "blurb": "The first molecule to form in the early universe.",
        "atoms": [(2, (-0.45, 0, 0)), (1, (0.6, 0, 0))], "charge": 1, "reference": {"distances_A": [0.772], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "o_atom", "group": "Atoms", "name": "Oxygen atom", "formula": "O",
        "blurb": "Eight electrons, six of them in play. Find the lowest spin to see which state wins.",
        "atoms": [(8, (0, 0, 0))], "reference": {"note": "experiment: triplet ground state (two unpaired spins)"},
        "mode": "frozen",
    },
    {
        "id": "water", "name": "Water", "formula": "H₂O",
        "blurb": "Released at 150° with stretched arms. Nothing here knows about lone pairs.",
        "atoms": [(8, (0, 0, 0)), (1, (1.014, 0.272, 0)), (1, (-1.014, 0.272, 0))],
        "reference": {"distances_A": [0.9572, 0.9572, 1.5139], "angles_deg": [104.52], "note": "experiment",
                      "method_note": "LDA itself predicts about 0.970 Å and 104.9°."},
        "mode": "relax",
    },
    {
        "id": "ammonia", "name": "Ammonia", "formula": "NH₃",
        "blurb": "Started almost flat. Does it stay flat?",
        "atoms": [(7, (0, 0, 0.12)), (1, (1.08, 0, 0)), (1, (-0.54, 0.935, 0)), (1, (-0.54, -0.935, 0))],
        "reference": {"distances_A": [1.012, 1.012, 1.012, 1.624, 1.624, 1.624], "angles_deg": [106.7],
                      "note": "experiment: a pyramid"},
        "mode": "relax",
    },
    {
        "id": "methane", "name": "Methane", "formula": "CH₄",
        "blurb": "Four hydrogens placed unevenly around a carbon.",
        "atoms": [(6, (0, 0, 0)), (1, (1.2, 0.1, 0)), (1, (-0.35, 0.95, 0.3)),
                  (1, (-0.4, -0.5, 0.9)), (1, (-0.3, -0.45, -1.0))],
        "reference": {"distances_A": [1.087] * 4 + [1.775] * 6, "angles_deg": [109.47],
                      "note": "experiment: a regular tetrahedron"},
        "mode": "relax",
    },
    {
        "id": "lih", "name": "Lithium hydride", "formula": "LiH",
        "blurb": "A metal and hydrogen. Watch where the electrons go.",
        "atoms": [(3, (-0.9, 0, 0)), (1, (0.9, 0, 0))], "reference": {"distances_A": [1.5949], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "n2", "name": "Nitrogen", "formula": "N₂",
        "blurb": "Two nitrogen atoms, too far apart. The air's strongest bond.",
        "atoms": [(7, (-0.68, 0, 0)), (7, (0.68, 0, 0))], "reference": {"distances_A": [1.0977], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "co", "name": "Carbon monoxide", "formula": "CO",
        "blurb": "Carbon and oxygen, stretched apart.",
        "atoms": [(6, (-0.68, 0, 0)), (8, (0.68, 0, 0))], "reference": {"distances_A": [1.1283], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "o2", "name": "Oxygen", "formula": "O₂",
        "blurb": "Started with every spin paired. Find the lowest spin: is oxygen magnetic?",
        "atoms": [(8, (-0.72, 0, 0)), (8, (0.72, 0, 0))],
        "reference": {"distances_A": [1.2075], "note": "experiment: triplet (paramagnetic)"},
        "mode": "relax",
    },
    {
        "id": "h2s", "name": "Hydrogen sulfide", "formula": "H₂S",
        "blurb": "Sulfur sits right below oxygen. Is its angle the same as water's?",
        "atoms": [(16, (0, 0, 0)), (1, (1.2, 0.5, 0)), (1, (-1.2, 0.5, 0))],
        "reference": {"distances_A": [1.336, 1.336, 1.923], "angles_deg": [92.1], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "nacl", "name": "Sodium chloride", "formula": "NaCl",
        "blurb": "A metal and a halogen. Watch the density move from one atom to the other.",
        "atoms": [(11, (-1.4, 0, 0)), (17, (1.4, 0, 0))], "reference": {"distances_A": [2.3609], "note": "experiment"},
        "mode": "relax",
    },
    {
        "id": "al2", "name": "Aluminium dimer", "formula": "Al₂",
        "blurb": "Two aluminium atoms, all spins paired. Find the lowest spin.",
        "atoms": [(13, (-1.5, 0, 0)), (13, (1.5, 0, 0))],
        "reference": {"distances_A": [2.70], "note": "experiment: triplet ground state"},
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
    return [{k: p[k] for k in ("id", "name", "formula", "blurb", "mode", "reference")} | {"group": p.get("group", "Molecules")}
            for p in PRESETS]
