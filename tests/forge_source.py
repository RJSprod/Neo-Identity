"""Pieces of the real Forge Neo, lifted out of its source for the tests.

The parity tests need the host's own transformer, ``backend/nn/krea.py``, and
the handful of functions it calls. Importing the host's packages pulls in its
whole runtime (comfy-kitchen, psutil, the loaders), so each piece is cut out of
its file by name with ``ast`` and executed with only the dependencies it uses.
Nothing is copied into this repository: the source is read from the Forge
checkout the tests are pointed at.

The checkout is found from ``NEO_IDENTITY_FORGE_ROOT``, or by looking two
directories up, where Forge keeps its ``extensions`` folder. Without one, the
tests that need the host skip and say so.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

PINNED = "46af880c003d31846bbea984ce00fc7be247ce5b"


def forge_root() -> Path | None:
    """The Forge Neo checkout to read, or None."""
    candidates = []
    env = os.environ.get("NEO_IDENTITY_FORGE_ROOT")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO.parent.parent)  # <forge>/extensions/<this repo>
    for root in candidates:
        if (root / "backend" / "nn" / "krea.py").is_file() and (root / "backend" / "nn" / "flux.py").is_file():
            return root
    return None


def source_of(root: Path, relative: str, names: tuple[str, ...]) -> str:
    """The source text of the named top-level definitions in ``relative``."""
    path = root / relative
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {name: None for name in names}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.name in wanted:
            wanted[node.name] = ast.get_source_segment(path.read_text(encoding="utf-8"), node)
    missing = [name for name, text in wanted.items() if text is None]
    if missing:
        raise LookupError(f"{relative} no longer defines {missing}")
    return "\n\n".join(wanted.values())


def load_definitions(root: Path, relative: str, names: tuple[str, ...], namespace: dict) -> dict:
    """Execute the named definitions from ``relative`` in ``namespace`` and
    return it. The namespace must already hold what they refer to."""
    code = compile(source_of(root, relative, names), str(root / relative), "exec")
    exec(code, namespace)
    return namespace


def load_module_from(root: Path, relative: str, name: str, sys_modules: dict):
    """Import ``relative`` as module ``name`` with ``sys_modules`` providing its
    dependencies (the caller installs the stubs first)."""
    import importlib.util

    path = root / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys_modules[name] = module
    spec.loader.exec_module(module)
    return module
