#!/bin/bash
# vxk_arc4_inner.sh — in-container build script for the arc4 wheel
# (P6.30 FIX-P). Runs INSIDE vxk-arc3-builder (same container/venv as the
# proven arc3 build: oneAPI 2026.0.0 == venv intel_sycl_rt 2026.0.0,
# torch 2.13.0+xpu). Only difference vs arc3 inner: version suffix +arc4
# and the .deps keep (toolchain unchanged -> the rls-v3.13 oneDNN source
# and build cache stay; our csrc wrapper layer is what changed).
set -euo pipefail

echo "=== VXK_ARC4 toolchain probe $(date) ==="
# setvars' env scripts reference unset vars (compiler/latest/env/vars.sh:258
# OCL_ICD_FILENAMES) — under set -u that expansion error kills the SHELL.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

# target-stack venv at /opt/arcvenv (persistent builder; arc3 left it warm)
if [ ! -x /opt/arcvenv/bin/python ]; then
  python3 -m venv --without-pip /opt/arcvenv
  curl -LsSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  /opt/arcvenv/bin/python /tmp/get-pip.py -q
fi
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
if ! /opt/arcvenv/bin/python -c "import torch" >/dev/null 2>&1; then
  pip install -q --upgrade pip wheel setuptools
  pip install -q torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
fi
/opt/arcvenv/bin/python -c "import cmake" >/dev/null 2>&1 || \
  pip install -q -r /src/vllm-xpu-kernels/requirements.txt

icpx --version | head -1
echo "CMPLR_ROOT=${CMPLR_ROOT:-unset}"
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"
ls -d /src/vllm-xpu-kernels/.deps/onednn-src >/dev/null 2>&1 && \
  echo "ONEDNN_DEPS_PRESENT (kept: same toolchain as arc3)" || \
  echo "ONEDNN_DEPS_MISSING (FetchContent will refetch — rig github reachable)"
echo "=== probe done ==="

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

cd /src/vllm-xpu-kernels
# clean only build/ (object files reference the changed headers); .deps
# KEPT — see header comment. arc3's full-clean rationale (cross-toolchain
# ABI hazard) does not apply to arc4.
rm -rf build
rm -rf ./*.egg-info

export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24
export KERNELS_MAX_JOBS=24
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc4

echo "VXK_ARC4_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e
echo "VXK_ARC4_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
