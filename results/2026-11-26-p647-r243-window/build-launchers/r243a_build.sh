#!/bin/bash
# r243a_build.sh -- P6.47 P-C2 r243a: build33 wheel = build4 (r242d) +
# the windowed work-list fix, both sides of the interface/kernel
# contract: (1) mx_attn_interface.py prices the host split plan from
# the WINDOWED tile extent (ceil(L/16) - k_block0) when a window rides
# the call -- the kernel adds k_block0 to every work item's tile start,
# so the pre-fix full-length plan walked tiles past k_blocks, past the
# block-table row (r243 mxfp4 DEVICE_LOST at W=64/ctx 8192 forced-8;
# mxfp8 survived on mapped garbage with every OOB column masked);
# (2) mx_paged_decode_kernel.hpp clamps the work-list walk to k_blocks
# (defense in depth: a mispriced plan under-covers visibly instead of
# killing the device). Incremental (VXK_ARC7MX_CLEAN=0 -- one kernel
# header + one python file). The build4 wheel is preserved
# (.r243aprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build33_r243a.log

fail() { echo "R243A_ABORT: $*"; }

grep -q "r243a WINDOWED WORK-LIST CLAMP" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_kernel.hpp \
  || { fail "kernel clamp not applied"; exit 1; }
grep -q "r243a (2026-11-26): with a window riding" \
  $SRC/vllm_xpu_kernels/mx_attn_interface.py \
  || { fail "windowed plan pricing not applied"; exit 1; }
grep -q "r242d" \
  $SRC/vllm_xpu_kernels/mx_attn_interface.py \
  || { fail "r242d split-plan tile-unit fix lost"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r243aprebak"

echo "R243A_BUILD_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }
md5sum "$WHEEL" | tee ${WHEEL}.r243a.md5
echo "R243A_BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
