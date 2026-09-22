"""Command line entry point: ``matter-sim`` (viewer) and ``matter-sim validate``."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import threading
import webbrowser
from pathlib import Path

VIEWER = Path(__file__).resolve().parent.parent / "viewer"


def ensure_viewer_built() -> None:
    """Build the browser viewer on first launch (needs Node.js)."""
    if (VIEWER / "dist" / "index.html").exists():
        return
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("The viewer needs Node.js to build once. Install it (brew install node) and rerun.")
    print("Building the viewer (first launch only)…")
    if not (VIEWER / "node_modules").exists():
        subprocess.run([npm, "install", "--silent"], cwd=VIEWER, check=True)
    subprocess.run([npm, "run", "build", "--silent"], cwd=VIEWER, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(prog="matter-sim", description="First-principles matter simulator")
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    v = sub.add_parser("validate", help="run the experiment-vs-simulation checks")
    v.add_argument("--quick", action="store_true", help="coarser grids, fewer checks")
    args = ap.parse_args()

    if args.cmd == "validate":
        from validation.run import main as validate
        validate(quick=args.quick)
        return

    ensure_viewer_built()
    from server.app import serve
    url = f"http://127.0.0.1:{args.port}/"
    print(f"matter-sim  →  {url}   (Ctrl+C to quit)")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    serve(port=args.port)


if __name__ == "__main__":
    main()
