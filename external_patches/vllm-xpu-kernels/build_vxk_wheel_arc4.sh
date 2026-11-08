#!/bin/bash
# build_vxk_wheel_arc4.sh — rebuild vllm-xpu-kernels with the P6.30 FIX-P
# mxfp4 W4A16 route (new csrc/xpu/onednn/fp4_gemm_w4a16.h + joint dtypes
# bf16_mxfp4/f16_mxfp4 + op fp4_gemm_w4a16). Same builder container and
# toolchain as arc3 (vxk-arc3-builder, oneAPI 2026.0.0, torch 2.13.0+xpu)
# — the ABI law (compiler generation == runtime generation) is unchanged,
# so we REUSE the warm container + venv; only the source tree moved.
#
# Deviation vs arc3 inner (documented): arc3 did `rm -rf build .deps`
# because arc2->arc3 changed the TOOLCHAIN. arc4 changes only our csrc
# wrapper layer (oneDNN itself untouched) -> keep .deps (rls-v3.13 source
# + build cache present and generation-matched), clean only build/.
#
# Wheel: vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl
# -> /root/build/wheels-arc-vxk/ (mounted /wheels in-container).
set -euo pipefail

VXK_SRC=/root/build/vxk-arc-src
WHEELS=/root/build/wheels-arc-vxk
INNER=/root/build/vxk_arc4_inner.sh
BASE=intel/omix:0.2.0-devel-ubuntu24.04

[ -f "$VXK_SRC/csrc/xpu/onednn/fp4_gemm_w4a16.h" ] || { echo "FATAL: FIX-P header missing"; exit 1; }
[ -f "$INNER" ] || { echo "FATAL: inner script $INNER missing"; exit 1; }
docker image inspect "$BASE" >/dev/null || { echo "FATAL: image $BASE missing"; exit 1; }
mkdir -p "$WHEELS"

if ! docker inspect vxk-arc3-builder >/dev/null 2>&1; then
  docker run -d --name vxk-arc3-builder \
    -v "$VXK_SRC":/src/vllm-xpu-kernels \
    -v "$WHEELS":/wheels \
    "$BASE" sleep infinity >/dev/null
fi
docker start vxk-arc3-builder >/dev/null 2>&1 || true
docker cp "$INNER" vxk-arc3-builder:/inner.sh
docker exec vxk-arc3-builder bash /inner.sh
