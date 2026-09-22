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
        Element(19, "K", "Potassium", 38.96370649, "#a07aff"),
        Element(20, "Ca", "Calcium", 39.96259086, "#8cf07a"),
        Element(21, "Sc", "Scandium", 44.95590828, "#d8d8e0"),
        Element(22, "Ti", "Titanium", 47.94794198, "#bfc4cc"),
        Element(23, "V", "Vanadium", 50.94395704, "#a8a8b8"),
        Element(24, "Cr", "Chromium", 51.94050623, "#8ea0c8"),
        Element(25, "Mn", "Manganese", 54.93804391, "#b08ac8"),
        Element(26, "Fe", "Iron", 55.93493633, "#e0864f"),
        Element(27, "Co", "Cobalt", 58.93319429, "#f08ca8"),
        Element(28, "Ni", "Nickel", 57.93534241, "#78d078"),
        Element(29, "Cu", "Copper", 62.92959772, "#e0955a"),
        Element(30, "Zn", "Zinc", 63.92914201, "#9098c8"),
        Element(31, "Ga", "Gallium", 68.9255735, "#c89090"),
        Element(32, "Ge", "Germanium", 73.92117776, "#90a8a8"),
        Element(33, "As", "Arsenic", 74.92159457, "#c080e0"),
        Element(34, "Se", "Selenium", 79.9165218, "#ffb040"),
        Element(35, "Br", "Bromine", 78.9183376, "#c05050"),
        Element(36, "Kr", "Krypton", 83.9114977, "#60c8e0"),
    ]
}

MAX_Z = max(ELEMENTS)


def by_symbol(symbol: str) -> Element:
    for e in ELEMENTS.values():
        if e.symbol.lower() == symbol.lower():
            return e
    raise KeyError(symbol)
