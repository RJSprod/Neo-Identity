#!/bin/bash
# Claude Code on the web: make the test suite runnable in a fresh container.
#
# The tests need torch (CPU is enough), einops, numpy, pillow and pytest; the
# parity tests also need a Forge Neo checkout, which this hook clones at the
# pinned commit into the scratch area and names through NEO_IDENTITY_FORGE_ROOT.
# Nothing is installed when everything is already there, and nothing runs
# outside Claude Code on the web.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

wanted=""
python3 -c "import torch" >/dev/null 2>&1 || wanted="$wanted torch"
python3 -c "import einops" >/dev/null 2>&1 || wanted="$wanted einops"
python3 -c "import numpy" >/dev/null 2>&1 || wanted="$wanted numpy"
python3 -c "import PIL" >/dev/null 2>&1 || wanted="$wanted pillow"
python3 -c "import pytest" >/dev/null 2>&1 || wanted="$wanted pytest"
python3 -c "import xdist" >/dev/null 2>&1 || wanted="$wanted pytest-xdist"

if [ -n "$wanted" ]; then
  # shellcheck disable=SC2086  # one argument per package, deliberately
  python3 -m pip install --quiet --disable-pip-version-check --root-user-action=ignore --break-system-packages $wanted || true
fi

PIN="46af880c003d31846bbea984ce00fc7be247ce5b"
TARGET="${NEO_IDENTITY_FORGE_ROOT:-/tmp/neo-identity-forge-neo}"
if [ ! -f "$TARGET/backend/nn/krea.py" ]; then
  mkdir -p "$TARGET" && cd "$TARGET" && git init -q && git remote add origin https://github.com/Haoming02/sd-webui-forge-classic 2>/dev/null || true
  (git fetch -q --depth 1 origin "$PIN" && git checkout -q FETCH_HEAD) || echo "could not fetch Forge Neo; the parity tests will skip"
fi
echo "export NEO_IDENTITY_FORGE_ROOT=$TARGET" >> "${CLAUDE_ENV_FILE:-/dev/null}" 2>/dev/null || true
