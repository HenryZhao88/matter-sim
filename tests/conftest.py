"""What this machine can run.

The GPU backend is Apple-only, so tests that name it explicitly are skipped elsewhere, and
the physics checks run on whatever backend the machine actually has. A result that depends on
the backend would be a bug, which is what ``BACKENDS`` is for: where both exist, both run.
"""

import pytest

from engine.core.accel import have_mlx

BACKENDS = ["numpy"] + (["mlx"] if have_mlx() else [])
DEFAULT_BACKEND = "mlx" if have_mlx() else "numpy"

requires_mlx = pytest.mark.skipif(not have_mlx(), reason="needs the MLX GPU backend (Apple silicon)")
