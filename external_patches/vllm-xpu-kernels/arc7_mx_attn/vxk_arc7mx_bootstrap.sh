#!/bin/bash
# vxk_arc7mx_bootstrap.sh — ONE-TIME venv reconstruction inside a FRESH
# vxk-arc3-builder container (the original builder was removed from the
# rig after the P6.34 jm builds; its /opt/arcvenv did not survive). Runs
# INSIDE the container (docker exec). Recipe = the arc3 postmortems
# (external_patches/vllm-xpu-kernels/README.md):
#   - omix 0.2.0-devel ships a hollow /opt/venv (python symlinks, no pip)
#   - Ubuntu ensurepip needs python3.12-venv (not installed) -> create
#     the venv --without-pip and bootstrap pip via get-pip.py
#   - astral.com (uv) is unreachable from the rig; pypi/pytorch indexes
#     are NOT (arc3 note) -> plain pip works
#   - ABI law: torch pin == the arc image venv (2.13.0+xpu, cp312) and
#     oneAPI 2026.0.0 (the omix 0.2.0 toolchain in this image)
# Idempotent: exits 0 immediately if /opt/arcvenv already has torch.
set -euo pipefail

VENV=/opt/arcvenv

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "VXK_ARC7MX_BOOTSTRAP_SKIP: $VENV carries torch $($VENV/bin/python -c 'import torch; print(torch.__version__)')"
  exit 0
fi

echo "=== VXK_ARC7MX bootstrap $(date) ==="
python3 --version

# CMake FindPython needs the dev headers/libpython (the omix image ships
# none; diagnosed live on the first r198 configure, 2026-11-19). Ubuntu
# archives are reachable from the rig (same egress as pypi).
if [ ! -f /usr/include/python3.12/Python.h ]; then
  apt-get update -qq && apt-get install -y -qq python3.12-dev
fi

rm -rf "$VENV"
python3 -m venv --without-pip "$VENV"

curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
"$VENV/bin/python" /tmp/get-pip.py

# torch: the XPU wheel from the pytorch xpu index (same train the arc
# image venv installed from). Exact pin 2.13.0+xpu (ABI law).
"$VENV/bin/pip" install --no-cache-dir "torch==2.13.0+xpu" \
  --index-url https://download.pytorch.org/whl/xpu

# build deps from the tree's requirements.txt (cmake, ninja, ...). The
# source bind may not be swapped in yet during early bring-up -- fall
# back to the explicit minimum set.
if [ -f /src/vllm-xpu-kernels/requirements.txt ]; then
  "$VENV/bin/pip" install --no-cache-dir -r /src/vllm-xpu-kernels/requirements.txt
else
  "$VENV/bin/pip" install --no-cache-dir cmake ninja wheel setuptools
fi

"$VENV/bin/python" - <<'PY'
import torch, sys
print("BOOTSTRAP_OK torch", torch.__version__, "python", sys.version.split()[0])
PY
