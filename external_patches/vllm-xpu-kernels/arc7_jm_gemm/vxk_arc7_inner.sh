#!/bin/bash
# vxk_arc7_inner.sh — in-container build script for the arc7 wheel
# (P6.34 JM/DPAS skinny ops on top of the arc6 tree). Same flow as
# vxk_arc6_inner.sh: runs INSIDE vxk-arc3-builder (carried /opt/arcvenv —
# oneAPI 2026.0.0, torch 2.13.0+xpu; the HOST bind was host-swapped to the
# arc7 tree, so the container sees it at /src/vllm-xpu-kernels where the
# carried .deps cache lives). Differences vs arc6 inner: version suffix
# +arc7, echo tags VXK_ARC7, jm sanity gate, wheel lands in /wheels.
# docker exec does NOT propagate host env — VXK_ARC7_CLEAN must be passed
# with `docker exec -e` for resume mode.
set -euo pipefail

echo "=== VXK_ARC7 toolchain probe $(date) ==="
# setvars' env scripts reference unset vars (compiler/latest/env/vars.sh:258
# OCL_ICD_FILENAMES) — under set -u that expansion error kills the SHELL.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

if [ ! -x /opt/arcvenv/bin/python ]; then
  echo "FATAL: /opt/arcvenv missing — commit did not carry the venv"
  exit 1
fi
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
/opt/arcvenv/bin/python -c "import cmake" >/dev/null 2>&1 || \
  pip install -q -r /src/vllm-xpu-kernels/requirements.txt

icpx --version | head -1
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"
# arc7 sanity: the jm module must be in the tree the container sees (and
# the arc6 skinny module beneath it)
ls /src/vllm-xpu-kernels/csrc/xpu/jm_gemm/jm_gemm_interface.cpp \
   /src/vllm-xpu-kernels/csrc/xpu/jm_gemm/jm_gemm_kernel.hpp \
   /src/vllm-xpu-kernels/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp \
   >/dev/null || { echo "FATAL: arc7 jm sources not visible"; exit 1; }
echo "=== probe done ==="

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

cd /src/vllm-xpu-kernels
# clean only build/ (object files reference the changed headers); .deps
# KEPT — same toolchain as arc3..arc6.
# resume mode: default full-clean; VXK_ARC7_CLEAN=0 keeps build/ for
# incremental resume after a late-TU failure
if [ "${VXK_ARC7_CLEAN:-1}" = "1" ]; then rm -rf build; fi
rm -rf ./*.egg-info

export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24
export KERNELS_MAX_JOBS=24
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc7

echo "VXK_ARC7_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e
echo "VXK_ARC7_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
