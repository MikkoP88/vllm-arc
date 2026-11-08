#!/bin/bash
# vxk_arc6_inner.sh — in-container build script for the arc6 wheel
# (P6.33 skinny-M ops + v0.1.17 Option-B backports). Runs INSIDE the
# existing vxk-arc3-builder container (carried /opt/arcvenv — oneAPI
# 2026.0.0 == venv intel_sycl_rt 2026.0.0, torch 2.13.0+xpu; the HOST
# bind /root/build/vxk-arc-src was host-swapped to the arc6 tree, so the
# container sees it at the same /src/vllm-xpu-kernels path its carried
# .deps cache expects). Differences vs arc5 inner: version suffix +arc6,
# echo tags VXK_ARC6. Toolchain unchanged -> .deps stays, only build/ is
# cleaned (object files reference the changed headers). NOTE: docker exec
# does NOT propagate host env, so VXK_ARC6_CLEAN must be passed with
# `docker exec -e` for resume mode.
set -euo pipefail

echo "=== VXK_ARC6 toolchain probe $(date) ==="
# setvars' env scripts reference unset vars (compiler/latest/env/vars.sh:258
# OCL_ICD_FILENAMES) — under set -u that expansion error kills the SHELL.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

# target-stack venv at /opt/arcvenv (carried by the container)
if [ ! -x /opt/arcvenv/bin/python ]; then
  echo "FATAL: /opt/arcvenv missing — commit did not carry the venv"
  exit 1
fi
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
/opt/arcvenv/bin/python -c "import cmake" >/dev/null 2>&1 || \
  pip install -q -r /src/vllm-xpu-kernels/requirements.txt

icpx --version | head -1
echo "CMPLR_ROOT=${CMPLR_ROOT:-unset}"
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"
ls -d /src/vllm-xpu-kernels/.deps/onednn-src >/dev/null 2>&1 && \
  echo "ONEDNN_DEPS_PRESENT (kept: same toolchain as arc3/arc4/arc5)" || \
  echo "ONEDNN_DEPS_MISSING (FetchContent will refetch — rig github reachable)"
# arc6 sanity: the skinny module must be in the tree the container sees
ls /src/vllm-xpu-kernels/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp \
   /src/vllm-xpu-kernels/csrc/xpu/skinny_gemm/skinny_gemm_kernel.hpp \
   >/dev/null || { echo "FATAL: arc6 skinny sources not visible"; exit 1; }
echo "=== probe done ==="

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

cd /src/vllm-xpu-kernels
# clean only build/ (object files reference the changed headers); .deps
# KEPT — see header comment.
# resume mode: default full-clean; VXK_ARC6_CLEAN=0 keeps
# build/ for incremental resume after a late-TU failure
if [ "${VXK_ARC6_CLEAN:-1}" = "1" ]; then rm -rf build; fi
rm -rf ./*.egg-info

export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24
export KERNELS_MAX_JOBS=24
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc6

echo "VXK_ARC6_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e
echo "VXK_ARC6_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
