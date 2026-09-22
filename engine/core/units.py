"""Physical constants and unit conversions.

The engine works in Hartree atomic units (hbar = m_e = e = 4*pi*eps0 = 1).
Conversions to human units happen only at the edges (UI, validation reports).
CODATA 2018 values.
"""

BOHR_ANGSTROM = 0.529177210903          # 1 bohr in Å
HARTREE_EV = 27.211386245988            # 1 Hartree in eV
AU_TIME_FS = 2.4188843265857e-2         # 1 atomic time unit in fs
AMU_ME = 1822.888486209                 # 1 dalton in electron masses
KELVIN_HARTREE = 3.1668115634556e-6     # k_B * 1 K in Hartree
HC_EV_NM = 1239.84198                   # h*c in eV*nm


def hartree_to_nm(delta_e: float) -> float:
    """Photon wavelength (nm) for a transition energy given in Hartree."""
    return HC_EV_NM / (abs(delta_e) * HARTREE_EV)


def angstrom_to_bohr(x: float) -> float:
    return x / BOHR_ANGSTROM


def bohr_to_angstrom(x: float) -> float:
    return x * BOHR_ANGSTROM
