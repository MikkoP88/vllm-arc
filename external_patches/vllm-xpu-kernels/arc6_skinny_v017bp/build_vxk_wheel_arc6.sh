#!/bin/bash
# build_vxk_wheel_arc6.sh — rebuild vllm-xpu-kernels with the P6.33 arc6
# additions (csrc/xpu/skinny_gemm/ skinny-M decode GEMM module + the
# three skinny ops + W4A8 dispatch arms for the arc5 enum values + the
# v0.1.17 Option-B backports #611/#645/#650/#600). Same toolchain as
# arc3/arc4/arc5 (oneAPI 2026.0.0, torch 2.13.0+xpu) — ABI law unchanged,
# .deps kept (the tree carries the rls-v3.13 oneDNN source + build cache).
#
# Strategy (arc5 v3 host-side swap, unchanged): dockerd re-resolves the
# HOST path of the builder's bind on container start, so: stop the
# builder, keep the certified arc5 tree at /root/build/vxk-arc5-kept
# (nothing deleted), rename the arc6 tree into /root/build/vxk-arc-src,
# start the builder. The container then sees the arc6 tree at the exact
# path its carried .deps cache expects.
#
# Wheel: vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl
# -> /root/build/wheels-arc-vxk/ (mounted /wheels in-container).
set -euo pipefail

VXK_NEW=/root/build/vxk-arc6-src
VXK_PATH=/root/build/vxk-arc-src
VXK_KEPT=/root/build/vxk-arc5-kept
WHEELS=/root/build/wheels-arc-vxk
INNER=/root/build/vxk_arc6_inner.sh
B=vxk-arc3-builder

[ -f "$INNER" ] || { echo "FATAL: inner script $INNER missing"; exit 1; }
docker inspect "$B" >/dev/null 2>&1 || { echo "FATAL: builder $B missing"; exit 1; }
mkdir -p "$WHEELS"

if [ -d "$VXK_KEPT" ]; then
  # v3.1: swap already performed (re-run after a failed build)
  echo "[resume] host swap already done (arc5 kept at $VXK_KEPT)"
  [ -f "$VXK_PATH/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp" ] || { echo "FATAL: $VXK_PATH lacks arc6 skinny module"; exit 1; }
else
  [ -f "$VXK_NEW/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp" ] || { echo "FATAL: arc6 skinny module missing (run apply_arc6.py first)"; exit 1; }
  [ -f "$VXK_NEW/csrc/xpu/skinny_gemm/skinny_gemm_kernel.hpp" ] || { echo "FATAL: arc6 skinny kernel missing (run apply_arc6.py first)"; exit 1; }
  grep -q "mx_single_k_block" "$VXK_NEW/csrc/xpu/onednn/fp8_gemm_w8a8.h" || { echo "FATAL: backports missing (run apply_arc6.py first)"; exit 1; }
  grep -q "case joint_dtypes_t::f8_e4m3_mxfp4_bf16:" "$VXK_NEW/csrc/xpu/onednn/onednn_ext.h" || { echo "FATAL: W4A8 arms missing (run apply_arc6.py first)"; exit 1; }
  echo "=== v3 host-side swap $(date +%H:%M:%S) ==="
  docker stop "$B" >/dev/null
  mv "$VXK_PATH" "$VXK_KEPT"
  mv "$VXK_NEW" "$VXK_PATH"
  docker start "$B" >/dev/null
  sleep 2
  # the mount must now show the ARC6 tree at the cache-expected path
  docker exec "$B" ls /src/vllm-xpu-kernels/csrc/xpu/skinny_gemm/skinny_gemm_interface.cpp >/dev/null || { echo "FATAL: swap failed — /src/vllm-xpu-kernels does not show the arc6 module"; exit 1; }
  echo "[ok] builder sees the arc6 tree at /src/vllm-xpu-kernels (arc5 kept at $VXK_KEPT)"
fi

docker cp "$INNER" "$B":/inner.sh
docker exec "$B" bash /inner.sh
