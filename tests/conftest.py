"""What this machine can run.

The GPU backend is Apple-only, so tests that name it explicitly are skipped elsewhere, and
the physics checks run on whatever backend the machine actually has. A result that depends on
the backend would be a bug, which is what ``BACKENDS`` is for: where both exist, both run.
"""

import pytest

from engine.core.accel import autodiff, have_cuda, have_mlx, have_mps, have_torch

BACKENDS = ["numpy"] + (["mlx"] if have_mlx() else [])
DEFAULT_BACKEND = "mlx" if have_mlx() else "numpy"

requires_mlx = pytest.mark.skipif(not have_mlx(), reason="needs the MLX GPU backend (Apple silicon)")
requires_torch = pytest.mark.skipif(not have_torch(), reason="needs PyTorch (the optional 'gpu' extra)")
requires_autodiff = pytest.mark.skipif(autodiff() is None, reason="truth mode needs MLX or PyTorch")
# GPU implementations of the Wilson–Dirac operator this machine can run (torch only on a GPU)
GPU_DIRAC = [pytest.param("mlx", marks=pytest.mark.skipif(not have_mlx(), reason="needs MLX")),
             pytest.param("torch", marks=pytest.mark.skipif(not (have_cuda() or have_mps()),
                                                            reason="needs torch with CUDA or MPS"))]
