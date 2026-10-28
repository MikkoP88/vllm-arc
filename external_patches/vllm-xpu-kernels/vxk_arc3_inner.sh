#!/bin/bash
# vxk_arc3_inner.sh — in-container build script (runs INSIDE the arc3 builder
# container = intel/omix:0.2.0-devel-ubuntu24.04; driven by
# build_vxk_wheel_arc3.sh). Recipe shape = the D-lineage-proven v78/v88
# builder (rig build_vxk_v78.sh): oneAPI setvars + uv venv with the TARGET
# torch, then pip wheel. Two pin changes vs v78: oneAPI 2025.3 -> 2026.0
# (omix 0.2.0) and torch 2.11.0+xpu -> 2.13.0+xpu (our 0.30 stack).
#
# Why arc3 (arc2 postmortem): arc2 (DPC++ 7.2.0 = oneAPI 2026.2) linked the
# right soname family (libsycl.so.9) but emitted calls to
# sycl::detail::reportAdjustedRange — a symbol the arc venv runtime
# (intel_sycl_rt 2026.0.0) does not export -> undefined symbol at import.
# Law: the COMPILER GENERATION must equal the RUNTIME GENERATION. The venv
# runtime is 2026.0.0 (= PyPI release train 2026-04-24; intel-cmplr-lib-rt
# 2026.0.0 installed); omix 0.2.0 = "Intel(R) oneAPI DPC++/C++ Compiler
# 2026.0.0 (2026.0.0.20260331)" — the exact match.
set -euo pipefail

echo "=== VXK_ARC3 toolchain probe $(date) ==="
# setvars' env scripts reference unset vars (compiler/latest/env/vars.sh:258
# OCL_ICD_FILENAMES) — under set -u that expansion error kills the SHELL
# (|| true cannot recover from it; this is why the v78 recipe used plain
# set -eo pipefail). Guard with set +u around the source only.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

# target-stack venv at /opt/arcvenv (NOT /opt/venv — the omix image ships a
# hollow /opt/venv placeholder: python symlinks, no pip, likely a mount
# point). Ubuntu's ensurepip lives in the python3.12-venv deb which is NOT
# installed, so create with --without-pip and bootstrap pip via get-pip.py
# (pypi endpoints are reachable from the rig; astral.com is not).
# Persistent across retries in the persistent builder.
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
echo "=== probe done ==="

# setup.py imports torch.utils.cpp_extension.SYCL_HOME (env-fed) and runs
# $SYCL_HOME/bin/icpx -v; setvars exports CMPLR_ROOT — mirror it to SYCL_HOME.
export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

cd /src/vllm-xpu-kernels
# full clean: arc2's build/ + .deps were produced by the 7.2.0 toolchain —
# incremental reuse across compiler generations is an ABI hazard.
rm -rf build .deps
rm -rf ./*.egg-info

export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=52
export KERNELS_MAX_JOBS=52
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc3

echo "VXK_ARC3_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e
echo "VXK_ARC3_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
