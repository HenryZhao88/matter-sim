"""Share computed results between machines through git.

Each machine's `.cache/` holds results that took hours (DFT labels, E(V) points, magnetic scans)
and lives only on that machine. This script writes them as plain JSON under
`results/machines/<machine>/`, one file per cache directory, and reads them back into a local
cache. JSON rather than the pickles themselves: loading someone else's pickle can run code, and git
can show a JSON diff.

    uv run python scripts/share_cache.py export <machine>     # mac | linux | windows | downstairs
    uv run python scripts/share_cache.py import               # every machine's files into .cache/

Import never overwrites. A file that exists locally with different content is reported as a
conflict (two machines computed the same item and disagree), which is worth a look, not a fix.
Pseudopotentials, collider tables and parton densities are not shared: every machine regenerates
or downloads them.
"""

import json
import pickle
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
OUT = ROOT / "results" / "machines"
MACHINES = ("mac", "linux", "windows", "downstairs")
SHARED = ("materials/dft", "materials/eos", "materials/melt_*", "fe_grid", "fe_magnetism")


def _enc(x):
    if isinstance(x, np.ndarray):
        return {"__ndarray__": x.tolist(), "dtype": str(x.dtype)}
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _enc(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_enc(v) for v in x]
    return x


def _dec(x):
    if isinstance(x, dict):
        if "__ndarray__" in x:
            return np.array(x["__ndarray__"], dtype=x["dtype"])
        return {k: _dec(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dec(v) for v in x]
    return x


def _same(a, b) -> bool:
    return json.dumps(_enc(a), sort_keys=True) == json.dumps(_enc(b), sort_keys=True)


def _read(path: Path):
    return pickle.loads(path.read_bytes()) if path.suffix == ".pkl" else json.loads(path.read_text())


def _dirs():
    for pattern in SHARED:
        yield from sorted(p for p in CACHE.glob(pattern) if p.is_dir())


def export(machine: str) -> None:
    dest = OUT / machine
    dest.mkdir(parents=True, exist_ok=True)
    counts = {}
    for d in _dirs():
        rel = d.relative_to(CACHE).as_posix()
        out = dest / (rel.replace("/", "__") + ".json")
        items = json.loads(out.read_text())["items"] if out.exists() else {}
        for f in sorted(d.iterdir()):
            if f.suffix in (".pkl", ".json"):
                items[f.name] = _enc(_read(f))
        out.write_text(json.dumps({"cache_dir": rel, "items": dict(sorted(items.items()))}, indent=1) + "\n")
        counts[rel] = len(items)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest = {"machine": machine, "host": platform.node(), "platform": platform.platform(),
                "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
                "items": counts}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    for rel, n in counts.items():
        print(f"{rel:28s} {n:5d}")
    print(f"-> {dest.relative_to(ROOT)}")


def import_all() -> None:
    added = kept = 0
    conflicts = []
    for f in sorted(OUT.glob("*/*.json")):
        if f.name == "manifest.json":
            continue
        doc = json.loads(f.read_text())
        d = CACHE / doc["cache_dir"]
        d.mkdir(parents=True, exist_ok=True)
        for name, item in doc["items"].items():
            obj, path = _dec(item), d / name
            if path.exists():
                if _same(_read(path), obj):
                    kept += 1
                else:
                    conflicts.append(f"{doc['cache_dir']}/{name} (from {f.parent.name})")
                continue
            if path.suffix == ".pkl":
                path.write_bytes(pickle.dumps(obj))
            else:
                path.write_text(json.dumps(item))
            added += 1
    print(f"added {added}, already present {kept}, conflicts {len(conflicts)}")
    for c in conflicts:
        print("  conflict:", c)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "export" and sys.argv[2] in MACHINES:
        export(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "import":
        import_all()
    else:
        sys.exit(__doc__)
