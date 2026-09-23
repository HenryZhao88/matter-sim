"""Truth mode: a neural-network wavefunction trained on the Schrödinger equation alone.

Variational Monte Carlo with a FermiNet-style ansatz; the method is described in vmc_mlx.py.
There are two implementations of the same network, sampler and optimiser:

- ``vmc_mlx``   — MLX, the Metal GPU on Apple silicon (preferred there).
- ``vmc_torch`` — PyTorch: CUDA, else MPS, else the CPU.

Which one runs is decided in engine/core/accel.py (``autodiff()``); MATTER_SIM_ACCEL=torch
forces the PyTorch path on a Mac, so the two can be compared on one machine. There is no NumPy
path: the local energy needs exact second derivatives by automatic differentiation.
"""

from __future__ import annotations

from ..core.accel import autodiff
from .molecule import Molecule

BACKEND = autodiff()

if BACKEND == "mlx":
    from .vmc_mlx import VMC, FermiNetLite
elif BACKEND == "torch":
    from .vmc_torch import VMC, FermiNetLite
else:
    raise ImportError("Truth mode needs MLX (Apple silicon) or PyTorch (the optional 'gpu' extra)")

__all__ = ["VMC", "FermiNetLite", "Molecule", "BACKEND"]
