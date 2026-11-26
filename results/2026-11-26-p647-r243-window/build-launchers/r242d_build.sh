#!/bin/bash
# r242d_build.sh -- P6.47 P-C1 r242d: build32 wheel = build31 + the
# split-plan tile-unit fix: the host plan priced tiles by the POOL's
# block size (_kv_tile_from_block_size -> 64 at page 64) while every
# fork geometry walks 16-row k-tiles (r212c _16 policy) -- forced-split
# work items covered only the first quarter of each seq (r242 forced8
# cos~0.51 at D=512 page 64; legacy/auto always priced 16 and was
# correct). kv_tile=16 in the interface plan. Python-only delta; the
# build31 wheel is preserved (.r242dprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build32_r242d.log

fail() { echo "R242D_ABORT: $*"; }

grep -q "P-C1 r242b V-SLICE LAW" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { fail "V-slice patch not applied"; exit 1; }
grep -q "SLICE WINDOW" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { fail "dequant window patch not applied"; exit 1; }
grep -q "window_size_left" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { fail "P-C2 xe2 window args not applied"; exit 1; }
grep -q "window_size_left" \
  $SRC/csrc/flash_attn/flash_api.cpp \
  || { fail "P-C2 flash_api schema not applied"; exit 1; }
grep -q "r242d" \
  $SRC/vllm_xpu_kernels/mx_attn_interface.py \
  || { fail "r242d split-plan tile-unit fix not applied"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r242dprebak"

echo "R242D_BUILD_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }
md5sum "$WHEEL" | tee ${WHEEL}.r242d.md5
echo "R242D_BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
