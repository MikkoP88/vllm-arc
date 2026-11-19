#!/bin/bash
# build_vxk_wheel_arc7mx.sh — rebuild vllm-xpu-kernels with the P6.47
# Phase-8 arc7mx additions (csrc/xpu/attn/xe_2/mx/ MX-native paged-decode
# module + the attn_interface/flash_api integration edits) on the
# CERTIFIED arc6 base. Toolchain unchanged from arc3..arc6 (oneAPI
# 2026.0.0, torch 2.13.0+xpu — ABI law), .deps kept from the base tar.
#
# Strategy (arc5 v3 host-side swap, unchanged): dockerd re-resolves the
# HOST path of the builder's bind on container start, so: stop the
# builder, keep the current live tree (arc6+jm) at
# /root/build/vxk-arc7jm-kept (nothing deleted), rename the arc7mx tree
# into /root/build/vxk-arc-src, start the builder. The container then
# sees the arc7mx tree at the exact path its carried .deps cache expects.
#
# The original builder container was removed after the P6.34 jm builds;
# this script RECREATES it (arc3 docker run law) and bootstraps its
# /opt/arcvenv once (vxk_arc7mx_bootstrap.sh) before the first build.
#
# Wheel: vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
# -> /root/build/wheels-arc-vxk/ (mounted /wheels in-container).
set -euo pipefail

VXK_NEW=/root/build/vxk-arc7mx-src
VXK_PATH=/root/build/vxk-arc-src
VXK_KEPT=/root/build/vxk-arc7jm-kept
WHEELS=/root/build/wheels-arc-vxk
INNER=/root/build/vxk_arc7mx_inner.sh
BOOT=/root/build/vxk_arc7mx_bootstrap.sh
B=vxk-arc3-builder
BASE=intel/omix:0.2.0-devel-ubuntu24.04

[ -f "$INNER" ] || { echo "FATAL: inner script $INNER missing"; exit 1; }
[ -f "$BOOT" ] || { echo "FATAL: bootstrap script $BOOT missing"; exit 1; }
mkdir -p "$WHEELS"

# --- builder (re)creation: the jm-era builder is gone; rebuild it on the
# omix 0.2.0-devel image with the same binds, then bootstrap the venv
# once (torch 2.13.0+xpu + cmake; see the bootstrap header).
if ! docker inspect "$B" >/dev/null 2>&1; then
  docker image inspect "$BASE" >/dev/null \
    || { echo "FATAL: image $BASE missing"; exit 1; }
  echo "=== builder recreation $(date +%H:%M:%S) ==="
  docker run -d --name "$B" \
    -v "$VXK_PATH":/src/vllm-xpu-kernels \
    -v "$WHEELS":/wheels \
    "$BASE" sleep infinity >/dev/null
fi
docker start "$B" >/dev/null 2>&1 || true
sleep 2
docker exec "$B" test -x /opt/arcvenv/bin/python 2>/dev/null || {
  docker cp "$BOOT" "$B":/bootstrap.sh
  docker exec "$B" bash /bootstrap.sh \
    || { echo "FATAL: venv bootstrap failed"; exit 1; }
}

# --- host-side swap (resume-aware, arc6 law)
if [ -d "$VXK_KEPT" ] && [ -d "$VXK_NEW" ]; then
  echo "FATAL: both $VXK_KEPT and $VXK_NEW exist — resolve manually (swap half-done or stale)"
  exit 1
fi
if [ -d "$VXK_KEPT" ]; then
  echo "[resume] host swap already done (live tree kept at $VXK_KEPT)"
  [ -f "$VXK_PATH/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp" ] \
    || { echo "FATAL: $VXK_PATH lacks the arc7mx module"; exit 1; }
else
  [ -f "$VXK_NEW/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp" ] \
    || { echo "FATAL: arc7mx module missing (run apply_arc7mx.py first)"; exit 1; }
  grep -q "cutlass_mx_paged_decode_interface" "$VXK_NEW/csrc/xpu/attn/attn_interface.h" \
    || { echo "FATAL: integration edits missing (run apply_arc7mx.py first)"; exit 1; }
  grep -q '"mx_paged_decode(Tensor q' "$VXK_NEW/csrc/flash_attn/flash_api.cpp" \
    || { echo "FATAL: flash_api edits missing (run apply_arc7mx.py first)"; exit 1; }
  echo "=== v3 host-side swap $(date +%H:%M:%S) ==="
  docker stop "$B" >/dev/null
  mv "$VXK_PATH" "$VXK_KEPT"
  mv "$VXK_NEW" "$VXK_PATH"
  docker start "$B" >/dev/null
  sleep 2
  docker exec "$B" ls /src/vllm-xpu-kernels/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp >/dev/null \
    || { echo "FATAL: swap failed — /src/vllm-xpu-kernels does not show the arc7mx module"; exit 1; }
  echo "[ok] builder sees the arc7mx tree at /src/vllm-xpu-kernels (live tree kept at $VXK_KEPT)"
fi

docker cp "$INNER" "$B":/inner.sh
docker exec "$B" bash /inner.sh
