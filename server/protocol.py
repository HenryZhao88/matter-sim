"""Wire format between engine and viewer.

Text frames carry JSON events/commands. Binary frames carry snapshots:

    uint32 little-endian  header length H
    H bytes               UTF-8 JSON header (meta + field encodings)
    pad to 4-byte boundary
    n³ uint8              electron density, log-scaled (see header "rho")
    n³ int8               spin density ρ↑−ρ↓, sqrt-scaled (see header "spin")

Density arrays are C-ordered with index (x, y, z) -> x*n*n + y*n + z.
"""

from __future__ import annotations

import json
import math
import struct

import numpy as np

RHO_REF = 1e-4  # e/bohr³ knee of the log curve


def encode_snapshot(snap: dict) -> bytes:
    rho = np.maximum(snap["rho"], 0.0)
    spin = snap["spin"]
    n = rho.shape[0]
    rho_max = float(rho.max()) if rho.size else 1.0
    rho_max = max(rho_max, 1e-12)
    norm = math.log1p(rho_max / RHO_REF)
    q_rho = np.round(255 * np.log1p(rho / RHO_REF) / norm).astype(np.uint8)
    s_max = max(float(np.abs(spin).max()) if spin.size else 0.0, 1e-12)
    q_spin = np.round(127 * np.sign(spin) * np.sqrt(np.abs(spin) / s_max)).astype(np.int8)

    header = {
        "meta": snap["meta"],
        "n": n,
        "rho": {"encoding": "log1p", "ref": RHO_REF, "max": rho_max},
        "spin": {"encoding": "sqrt", "max": s_max},
    }
    hb = json.dumps(header, separators=(",", ":"), allow_nan=False, default=_json_default).encode()
    pad = (-(4 + len(hb))) % 4
    return b"".join([struct.pack("<I", len(hb)), hb, b" " * pad, q_rho.tobytes(), q_spin.tobytes()])


def decode_snapshot(data: bytes) -> dict:
    """Inverse of :func:`encode_snapshot` (used by tests and tools)."""
    (hl,) = struct.unpack_from("<I", data, 0)
    header = json.loads(data[4:4 + hl])
    off = 4 + hl + ((-(4 + hl)) % 4)
    n = header["n"]
    q_rho = np.frombuffer(data, np.uint8, n ** 3, off).reshape(n, n, n)
    q_spin = np.frombuffer(data, np.int8, n ** 3, off + n ** 3).reshape(n, n, n)
    r = header["rho"]
    rho = r["ref"] * np.expm1(q_rho / 255 * math.log1p(r["max"] / r["ref"]))
    spin = np.sign(q_spin) * (q_spin / 127.0) ** 2 * header["spin"]["max"]
    return {"header": header, "rho": rho, "spin": spin}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sanitize(obj):
    """Replace NaN/inf (not valid JSON) with None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    return obj
