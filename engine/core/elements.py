"""Measured per-element inputs.

The only per-element data the physics uses is the nuclear charge Z and the
nuclear mass. Colours are for display. Nothing about chemistry (valence,
radii, preferred bonds) is stored here — that must come out of the solver.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    Z: int
    symbol: str
    name: str
    mass_amu: float   # most abundant isotope's atomic mass (NIST)
    color: str        # display only


ELEMENTS: dict[int, Element] = {
    e.Z: e
    for e in [
        Element(1, "H", "Hydrogen", 1.00782503, "#e8eef7"),
        Element(2, "He", "Helium", 4.00260325, "#b6f0ff"),
        Element(3, "Li", "Lithium", 7.01600344, "#c792ff"),
        Element(4, "Be", "Beryllium", 9.01218307, "#b5ff7a"),
        Element(5, "B", "Boron", 11.00930536, "#ffb88a"),
        Element(6, "C", "Carbon", 12.0, "#9aa4b5"),
        Element(7, "N", "Nitrogen", 14.00307400, "#6f8dff"),
        Element(8, "O", "Oxygen", 15.99491462, "#ff5a5a"),
        Element(9, "F", "Fluorine", 18.99840316, "#8cffb0"),
        Element(10, "Ne", "Neon", 19.99244018, "#ff9ad5"),
        Element(11, "Na", "Sodium", 22.98976928, "#b18cff"),
        Element(12, "Mg", "Magnesium", 23.9850417, "#9dff9a"),
        Element(13, "Al", "Aluminium", 26.98153853, "#d7c9c0"),
        Element(14, "Si", "Silicon", 27.97692653, "#e8c28f"),
        Element(15, "P", "Phosphorus", 30.97376200, "#ffa45c"),
        Element(16, "S", "Sulfur", 31.97207117, "#ffe066"),
        Element(17, "Cl", "Chlorine", 34.96885268, "#7dff7a"),
        Element(18, "Ar", "Argon", 39.96238312, "#85d8ff"),
    ]
}

MAX_Z = max(ELEMENTS)


def by_symbol(symbol: str) -> Element:
    for e in ELEMENTS.values():
        if e.symbol.lower() == symbol.lower():
            return e
    raise KeyError(symbol)
