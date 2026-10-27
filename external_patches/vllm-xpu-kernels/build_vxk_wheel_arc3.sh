#!/bin/bash
# build_vxk_wheel_arc3.sh — rebuild vllm-xpu-kernels (p13 int64 wedge fix)
# inside intel/omix:0.2.0-devel-ubuntu24.04 (oneAPI 2026.0.0 = the compiler
# generation matching the arc image venv runtime intel_sycl_rt 2026.0.0),
# against a uv venv pinned to the target stack torch 2.13.0+xpu.
#
# Lineage: the v78/v88 D-recipe (rig build_vxk_v78.sh) with two pin changes
# (oneAPI generation + torch). arc1 (omix 0.1.0 = oneAPI 2025.3) linked
# libsycl.so.8 and ImportError-d against the so.9 venv; arc2 (DPC++ 7.2.0 =
# 2026.2) linked so.9 but referenced reportAdjustedRange, absent from the
# 2026.0 runtime — undefined symbol at import. arc3 = generation-matched.
#
# Persistent builder container (venv + toolchain warm across retries).
# Wheel: vllm_xpu_kernels-0.1.14.1+arc3-cp312-cp312-linux_x86_64.whl
# -> /root/build/wheels-arc-vxk/ (mounted /wheels in-container).
set -euo pipefail

VXK_SRC=/root/build/vxk-arc-src
WHEELS=/root/build/wheels-arc-vxk
INNER=/root/build/vxk_arc3_inner.sh
BASE=intel/omix:0.2.0-devel-ubuntu24.04

[ -f "$VXK_SRC/CMakeLists.txt" ] || { echo "FATAL: $VXK_SRC missing/incomplete"; exit 1; }
[ -f "$VXK_SRC/requirements.txt" ] || { echo "FATAL: $VXK_SRC/requirements.txt missing"; exit 1; }
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
