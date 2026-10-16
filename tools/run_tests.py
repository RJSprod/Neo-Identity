"""Run the suite, against the real Forge Neo transformer when one is in reach.

    python3 tools/run_tests.py                 every test, on every core when pytest-xdist is there
    python3 tools/run_tests.py --serial        one core, in file order
    python3 tools/run_tests.py -- -k geometry  anything after -- goes to pytest

The parity tests load ``backend/nn/krea.py`` from a Forge Neo checkout. It is
found from ``NEO_IDENTITY_FORGE_ROOT`` or, when this repository sits in Forge's
``extensions`` folder, two directories up; without one those tests skip, and the
run says how many did.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import forge_source  # noqa: E402


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list = []
    if "--" in argv:
        at = argv.index("--")
        argv, extra = argv[:at], argv[at + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--serial", action="store_true", help="one core, in file order")
    parser.add_argument("--workers", default="auto", help="pytest-xdist workers (default: auto)")
    options = parser.parse_args(argv)

    root = forge_source.forge_root()
    if root is None:
        print("no Forge Neo checkout found: the parity tests will skip (set NEO_IDENTITY_FORGE_ROOT)", file=sys.stderr)
    else:
        print(f"Forge Neo checkout: {root}", file=sys.stderr)

    line = [sys.executable, "-m", "pytest", "-q", "-rs", str(ROOT / "tests")]
    if not options.serial and importlib.util.find_spec("xdist") is not None:
        line += ["-n", options.workers]
    elif not options.serial:
        print("pytest-xdist is not installed: running on one core", file=sys.stderr)
    line += extra
    return subprocess.call(line, cwd=str(ROOT), env=dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
