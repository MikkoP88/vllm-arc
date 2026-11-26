#!/bin/bash
# r245a_build.sh -- P6.47 P-C1 r245a: build34 wheel = build5 (r243a) +
# the page-128 admission for the D=512 full-attention class. The gemma
# hybrid KV manager banks the D=512 full_attention group a PAGE-128
# pool (the sliding D=256 group rides page 64 -- the same boot banks
# both pools). The r245 e2e grid died at the FIRST REAL REQUEST with
# "mx_paged_decode: block_size 128 not yet compiled" (engine.py:1261
# -> native_decode.py:135) while every sliding layer's capture-time
# native call with window=1024 had already succeeded -- the masked
# twin never saw the full-attention pool until a real decode ran.
# Surface: the TORCH_CHECK admits block_size 128 + a new dispatch
# branch (head_size == 512 && block_size == 128) riding the SAME
# certified _512/_16 sub-tile frame as 512/64 -- the mainloop's page
# indirection is runtime-generic (tiles_per_page = page_size/16 -> 8
# sub-tiles per 128-page vs 4 at page 64; page_local_idx = tile*16/
# page_size, mainloop L918-926), so this is an ADMISSION, not a new
# kernel surface. Windowless class (same !is_local refusal as the
# 512/64 branch). Incremental (VXK_ARC7MX_CLEAN=0 -- one .cpp TU).
# The build5 wheel is preserved (.r245aprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build34_r245a.log

fail() { echo "R245A_ABORT: $*"; }

grep -q "r245a (2026-11-26)" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { fail "page-128 dispatch branch not applied"; exit 1; }
grep -q "arc7 covers page 16/64/128" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { fail "block_size 128 admission not applied"; exit 1; }
grep -q "r243a WINDOWED WORK-LIST CLAMP" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_kernel.hpp \
  || { fail "r243a kernel clamp lost"; exit 1; }
PREV=$(cat "$WHEEL.r243a.md5" 2>/dev/null | cut -d" " -f1)
[ -n "$PREV" ] || { fail "build5 md5 sidecar missing (r243a_build.sh not completed)"; exit 1; }
CUR=$(md5sum "$WHEEL" 2>/dev/null | cut -d" " -f1)
[ "$CUR" = "$PREV" ] || { fail "wheel drifted from build5: $CUR != $PREV"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r245aprebak"

echo "R245A_BUILD_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }
md5sum "$WHEEL" | tee ${WHEEL}.r245a.md5
echo "R245A_BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
