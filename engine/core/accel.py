"""Which accelerator this machine has.

Three backends, in order of preference on the machine that has them:

- ``mlx``   — Apple silicon, Metal GPU (macOS only).
- ``torch`` — CUDA, if PyTorch is installed with a GPU build (Linux/Windows with NVIDIA).
- ``numpy`` — always there, CPU, and the reference every result is checked against.

Nothing here decides physics; it only says which array library the heavy loops may use, so the
same code runs on a MacBook and on a CUDA box.
"""

from __future__ import annotations

import functools
import os


@functools.lru_cache(maxsize=1)
def have_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=1)
def have_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def preferred() -> str:
    """``mlx``, ``torch`` or ``numpy``. MATTER_SIM_ACCEL overrides the choice."""
    forced = os.environ.get("MATTER_SIM_ACCEL", "").strip().lower()
    if forced in ("mlx", "torch", "numpy"):
        return forced
    if have_mlx():
        return "mlx"
    if have_cuda():
        return "torch"
    return "numpy"


def describe() -> str:
    which = preferred()
    if which == "mlx":
        return "Apple GPU (MLX)"
    if which == "torch":
        import torch
        return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
    return "CPU (NumPy)"
