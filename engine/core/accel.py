"""Which accelerator this machine has.

Backends, in order of preference on the machine that has them:

- ``mlx``   — Apple silicon, Metal GPU (macOS only).
- ``torch`` — PyTorch, if installed (the optional ``gpu`` extra): on a CUDA GPU where there is
  one, else on Apple's Metal through MPS, else on the CPU.
- ``numpy`` — always there, CPU, and the reference every result is checked against.

Nothing here decides physics; it only says which array library the heavy loops may use, so the
same code runs on a MacBook and on a CUDA box.
"""

from __future__ import annotations

import functools
import os


def _forced() -> str:
    return os.environ.get("MATTER_SIM_ACCEL", "").strip().lower()


@functools.lru_cache(maxsize=1)
def have_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=1)
def have_torch() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@functools.lru_cache(maxsize=1)
def have_cuda() -> bool:
    if not have_torch():
        return False
    import torch
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def have_mps() -> bool:
    if not have_torch():
        return False
    import torch
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def preferred() -> str:
    """``mlx``, ``torch`` or ``numpy``. MATTER_SIM_ACCEL overrides the choice.

    Automatically, torch is chosen only when it has a GPU to run on (CUDA, then MPS); torch on
    the CPU is no faster than NumPy for array code, so it is used there only where autodiff is
    needed (see :func:`autodiff`).
    """
    forced = _forced()
    if forced in ("mlx", "torch", "numpy"):
        return forced
    if have_mlx():
        return "mlx"
    if have_cuda() or have_mps():
        return "torch"
    return "numpy"


@functools.lru_cache(maxsize=1)
def autodiff() -> str | None:
    """``mlx`` or ``torch`` for code that needs automatic differentiation; None if neither exists.

    Same ordering as :func:`preferred` (MLX first on Apple silicon), except that torch on the CPU
    is an acceptable fallback, since NumPy has no autodiff to fall back to.
    """
    forced = _forced()
    if forced == "mlx" and have_mlx():
        return "mlx"
    if forced == "torch" and have_torch():
        return "torch"
    if have_mlx():
        return "mlx"
    if have_torch():
        return "torch"
    return None


@functools.lru_cache(maxsize=1)
def torch_device() -> str:
    """The device torch code runs on: ``cuda``, ``mps`` or ``cpu``. MATTER_SIM_TORCH_DEVICE overrides."""
    forced = os.environ.get("MATTER_SIM_TORCH_DEVICE", "").strip().lower()
    if forced:
        return forced
    if have_cuda():
        return "cuda"
    if have_mps():
        return "mps"
    return "cpu"


def describe() -> str:
    which = preferred()
    if which == "mlx":
        return "Apple GPU (MLX)"
    if which == "torch":
        import torch
        dev = torch_device()
        if dev == "cuda":
            return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        if dev == "mps":
            return "Apple GPU (PyTorch MPS)"
        return "CPU (PyTorch)"
    return "CPU (NumPy)"
