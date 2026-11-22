#!/bin/bash
# vxk_arc7mx_inner.sh — in-container build script for the arc7mx wheel
# (P6.47 Phase 8: MX-native paged-decode kernel on the certified arc6
# base). Same flow as vxk_arc6_inner.sh / vxk_arc7_inner.sh: runs INSIDE
# the vxk-arc3-builder container (oneAPI 2026.0.0 from omix 0.2.0-devel;
# the HOST bind was host-swapped to the arc7mx tree, so the container
# sees it at /src/vllm-xpu-kernels where the carried .deps cache lives).
# Differences vs the arc6 inner: version suffix +arc7mx (the +arc7 label
# belongs to the P6.34 jm lab line; this wheel is arc6-certified content
# + the mx module only), echo tags VXK_ARC7MX, mx sanity gate.
# docker exec does NOT propagate host env — VXK_ARC7MX_CLEAN must be
# passed with `docker exec -e` for resume mode.
set -euo pipefail

echo "=== VXK_ARC7MX toolchain probe $(date) ==="
# setvars' env scripts reference unset vars (compiler/latest/env/vars.sh:258
# OCL_ICD_FILENAMES) — under set -u that expansion error kills the SHELL.
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

if [ ! -x /opt/arcvenv/bin/python ]; then
  echo "FATAL: /opt/arcvenv missing — run vxk_arc7mx_bootstrap.sh first"
  exit 1
fi
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
/opt/arcvenv/bin/python -c "import cmake" >/dev/null 2>&1 || \
  pip install -q -r /src/vllm-xpu-kernels/requirements.txt

icpx --version | head -1
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__)"
# sanity: the tree the container sees must carry the arc7mx module on the
# CERTIFIED arc6 base (skinny present, jm ABSENT — provenance law)
ls /src/vllm-xpu-kernels/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
   /src/vllm-xpu-kernels/csrc/xpu/attn/xe_2/mx/mx_paged_decode.hpp \
   /src/vllm-xpu-kernels/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp \
   >/dev/null || { echo "FATAL: arc7mx sources not visible"; exit 1; }
[ ! -e /src/vllm-xpu-kernels/csrc/xpu/jm_gemm ] || { echo "FATAL: tree carries jm (wrong base)"; exit 1; }
grep -q "cutlass_mx_paged_decode_interface" /src/vllm-xpu-kernels/csrc/xpu/attn/attn_interface.h \
  || { echo "FATAL: attn_interface.h unpatched (run apply_arc7mx.py first)"; exit 1; }
grep -q '"mx_paged_decode(Tensor q' /src/vllm-xpu-kernels/csrc/flash_attn/flash_api.cpp \
  || { echo "FATAL: flash_api.cpp unpatched (run apply_arc7mx.py first)"; exit 1; }
echo "=== probe done ==="

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

cd /src/vllm-xpu-kernels
# clean only build/ (object files reference the changed headers); .deps
# KEPT — same toolchain as arc3..arc6.
# resume mode: default full-clean; VXK_ARC7MX_CLEAN=0 keeps build/ for
# incremental resume after a late-TU failure (the arc8 2h-build-death
# lesson: pre-flight the CHANGED TU with ninja before the full wheel)
if [ "${VXK_ARC7MX_CLEAN:-1}" = "1" ]; then rm -rf build; fi
rm -rf ./*.egg-info

export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=24
export KERNELS_MAX_JOBS=24
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc7mx

echo "VXK_ARC7MX_BUILD_START $(date)"
set +e
/opt/arcvenv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e
echo "VXK_ARC7MX_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
