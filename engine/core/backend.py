"""Array backend switch.

Two backends share one code path:

- ``mlx``   — Apple Metal GPU, float32. Fast; used for live simulation.
- ``numpy`` — CPU, float64. Slower; used to verify that float32 did not
  distort a result.

The solver is written against ``backend.xp`` using only the subset of the
array API that numpy and mlx.core both implement (arithmetic, ``@``,
``sum``, ``sqrt``, ``exp``, ``fft.rfftn`` / ``fft.irfftn``, ``zeros``).
Everything else goes through the small helpers on :class:`Backend`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx
except ImportError:  # pragma: no cover - mlx missing on non-Apple machines
    mx = None


@dataclass(frozen=True)
class Backend:
    name: str
    xp: Any
    dtype: Any

    def asarray(self, a) -> Any:
        a = np.asarray(a, dtype=np.float64 if self.name == "numpy" else np.float32)
        return a if self.name == "numpy" else mx.array(a)

    def to_numpy(self, a) -> np.ndarray:
        if self.name == "numpy":
            return np.asarray(a, dtype=np.float64)
        return np.array(a, copy=True).astype(np.float64)

    def eval(self, *arrays) -> None:
        """Force lazy (mlx) computation so graphs do not grow without bound."""
        if self.name == "mlx":
            mx.eval(*[a for a in arrays if a is not None])

    def rfftn(self, a):
        return self.xp.fft.rfftn(a, axes=[-3, -2, -1])

    def irfftn(self, a, shape):
        return self.xp.fft.irfftn(a, s=list(shape), axes=[-3, -2, -1])

    @property
    def is_gpu(self) -> bool:
        return self.name == "mlx"


def get_backend(name: str = "auto") -> Backend:
    if name == "auto":
        name = "mlx" if mx is not None else "numpy"
    if name == "numpy":
        return Backend("numpy", np, np.float64)
    if name == "mlx":
        if mx is None:
            raise RuntimeError("mlx is not installed; use the numpy backend")
        return Backend("mlx", mx, mx.float32)
    raise ValueError(f"unknown backend {name!r}")
