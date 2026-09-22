"""Everyday matter: a 1 cm³ block of aluminium, from properties the simulation derived itself.

No aluminium data enters here. Every input comes from the rungs below:

  lattice constant a(T)     molecular dynamics (NPT) with the learned potential
  bulk modulus, elastic     periodic DFT (crystal rung)
  enthalpy H(T)             molecular dynamics: heat capacity is dH/dT
  melting point, latent     solid–liquid coexistence in molecular dynamics
  cohesive energy           periodic DFT minus the free-atom energy

The block is ~6 × 10²² atoms; it cannot be simulated atom by atom, but it does not need to
be: it is a continuum whose response is set by these per-atom properties. That handoff
(atoms → per-atom averages → continuum) is how the ladder reaches everyday scales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


AVOGADRO = 6.02214076e23
EV_J = 1.602176634e-19
AL_MASS_AMU = 26.9815385            # from the nucleus rung (isotope mass)
AMU_KG = 1.66053906660e-27


@dataclass
class Derived:
    """Per-atom properties, each with where it came from."""
    a_of_T: list[tuple[float, float]]           # (T K, fcc lattice constant Å)
    H_of_T: list[tuple[float, float]]           # (T K, enthalpy per atom eV) — solid
    T_melt: float                               # K
    latent_eV: float                            # per atom
    B_GPa: float
    C11: float
    C12: float
    C44: float
    E_coh_eV: float

    def a(self, T: float) -> float:
        t, a = np.array(self.a_of_T).T
        return float(np.polyval(np.polyfit(t, a, 2 if len(t) > 3 else 1), T))

    def cp_per_atom_eV(self, T: float) -> float:
        t, h = np.array(self.H_of_T).T
        c = np.polyfit(t, h, 2 if len(t) > 3 else 1)
        return float(np.polyval(np.polyder(c), T))


class Block:
    def __init__(self, props: Derived, side_cm: float = 1.0, T: float = 293.15) -> None:
        self.p = props
        self.side_cm = side_cm
        self.T = T

    # ------------------------------------------------------------ at rest
    def atom_volume_m3(self, T: float | None = None) -> float:
        a = self.p.a(self.T if T is None else T) * 1e-10
        return a ** 3 / 4

    def density(self, T: float | None = None) -> float:
        """kg/m³."""
        return AL_MASS_AMU * AMU_KG / self.atom_volume_m3(T)

    def volume_m3(self) -> float:
        return (self.side_cm * 1e-2) ** 3

    def atoms(self) -> float:
        return self.volume_m3() / self.atom_volume_m3()

    def mass_g(self) -> float:
        return self.atoms() * AL_MASS_AMU * AMU_KG * 1e3

    # ------------------------------------------------------------ heat
    def linear_expansion_per_K(self, T: float | None = None) -> float:
        T = self.T if T is None else T
        return (self.p.a(T + 5) - self.p.a(T - 5)) / 10 / self.p.a(T)

    def specific_heat_J_per_gK(self, T: float | None = None) -> float:
        cp = self.p.cp_per_atom_eV(self.T if T is None else T) * EV_J
        return cp / (AL_MASS_AMU * AMU_KG * 1e3)

    def heat_to_warm_J(self, T1: float, T2: float) -> float:
        t = np.linspace(T1, T2, 200)
        cp = np.array([self.p.cp_per_atom_eV(x) for x in t]) * EV_J
        return float(np.trapezoid(cp, t)) * self.atoms()

    def heat_to_melt_J(self) -> float:
        """From the current temperature to fully liquid at the melting point."""
        return self.heat_to_warm_J(self.T, self.p.T_melt) + self.p.latent_eV * EV_J * self.atoms()

    def side_at_cm(self, T: float) -> float:
        return self.side_cm * self.p.a(T) / self.p.a(self.T)

    # ------------------------------------------------------------ force
    def squeeze(self, P_GPa: float) -> dict:
        """Uniform pressure: volume change from the bulk modulus."""
        dV = -P_GPa / self.p.B_GPa
        return {"dV_frac": dV, "side_cm": self.side_cm * (1 + dV) ** (1 / 3)}

    def stretch(self, stress_MPa: float) -> dict:
        """Uniaxial stress along a cube axis: Young's modulus and Poisson ratio from C11, C12."""
        c11, c12 = self.p.C11, self.p.C12
        E = (c11 - c12) * (c11 + 2 * c12) / (c11 + c12)
        nu = c12 / (c11 + c12)
        strain = stress_MPa * 1e-3 / E
        return {"young_GPa": E, "poisson": nu, "strain": strain,
                "side_cm": self.side_cm * (1 + strain), "width_cm": self.side_cm * (1 - nu * strain)}

    def shear_GPa(self) -> float:
        """Isotropic (Voigt–Reuss–Hill) shear modulus from the cubic elastic constants."""
        c11, c12, c44 = self.p.C11, self.p.C12, self.p.C44
        gv = (c11 - c12 + 3 * c44) / 5
        gr = 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12))
        return (gv + gr) / 2

    def sound_speeds(self) -> dict:
        rho = self.density()
        G = self.shear_GPa() * 1e9
        K = self.p.B_GPa * 1e9
        return {"longitudinal": math.sqrt((K + 4 * G / 3) / rho), "transverse": math.sqrt(G / rho)}

    def cohesive_energy_J(self) -> float:
        """Energy to take the whole block apart into free atoms."""
        return self.p.E_coh_eV * EV_J * self.atoms()

    def summary(self) -> dict:
        s = self.sound_speeds()
        return {
            "side_cm": self.side_cm, "T": self.T, "atoms": self.atoms(), "mass_g": self.mass_g(),
            "density": self.density(), "alpha_per_K": self.linear_expansion_per_K(),
            "c_J_per_gK": self.specific_heat_J_per_gK(), "T_melt": self.p.T_melt,
            "heat_to_melt_J": self.heat_to_melt_J(), "latent_J_per_g": self.p.latent_eV * EV_J / (AL_MASS_AMU * AMU_KG * 1e3),
            "B_GPa": self.p.B_GPa, "young_GPa": self.stretch(0)["young_GPa"], "shear_GPa": self.shear_GPa(),
            "poisson": self.stretch(0)["poisson"], "sound_long": s["longitudinal"], "sound_trans": s["transverse"],
            "cohesive_J": self.cohesive_energy_J(),
        }


# ------------------------------------------------------------------ assembling the chain
from pathlib import Path as _Path
import json as _json

_CACHE = _Path(__file__).resolve().parents[2] / ".cache" / "materials"

# Measured values, used only to show how close the derived ones come. Nothing reads them.
REFERENCE = {
    "a0_A": (4.046, "lattice constant at 293 K"), "B_GPa": (76.0, "bulk modulus"),
    "C11": (107.0, ""), "C12": (61.0, ""), "C44": (28.0, ""), "E_coh_eV": (3.39, "cohesive energy"),
    "T_melt": (933.5, "melting point, K"), "latent_eV": (0.111, "latent heat of fusion, eV/atom"),
    "density": (2699.0, "kg/m³ at 293 K"), "young_GPa": (70.0, "Young's modulus"), "alpha_per_K": (23.1e-6, "linear expansion, 1/K"),
    "c_J_per_gK": (0.897, "specific heat at 298 K"), "sound_long": (6420.0, "m/s"), "sound_trans": (3040.0, "m/s"),
}


def derived_aluminium() -> Derived | None:
    """Everything the block needs, read from what the lower rungs computed (None until they have)."""
    try:
        crys = _json.loads((_CACHE / "al_crystal.json").read_text())
        md = _json.loads((_CACHE / "al_results.json").read_text())
    except (OSError, ValueError):
        return None
    th = md["thermal"]
    return Derived(a_of_T=[(r["T"], r["a_A"]) for r in th], H_of_T=[(r["T"], r["H_eV"]) for r in th],
                   T_melt=md["melting"]["T_melt"], latent_eV=md["latent"]["latent_eV"],
                   B_GPa=crys["B_GPa"], C11=crys["C11"], C12=crys["C12"], C44=crys["C44"],
                   E_coh_eV=crys["E_coh_eV"])
