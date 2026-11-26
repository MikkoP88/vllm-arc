#!/bin/bash
# r242c_build.sh -- P6.47 P-C1 r242c: build31 wheel = build30 (r242b
# V-slice fix + P-C2 window surface) + the MISSING python interface
# sync: build30's live tree still carried the pre-P-C2-schema
# mx_attn_interface.py (13 positional args), so every windowless
# default-splits call crashed -- splits_per_seq_dev=None landed on the
# new schema's window_size_right (positional 10). This build ships the
# fixed interface (wl/wr inserted at schema positions 9/10,
# window_size None -> (-1, -1)); the C++ is byte-identical to build30.
# Incremental (VXK_ARC7MX_CLEAN=0; only the python package file
# changes -- no C++ recompile expected). The build30 wheel is
# preserved (.r242cprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build31_r242c.log

fail() { echo "R242C_ABORT: $*"; }

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
grep -q "wl, wr" \
  $SRC/vllm_xpu_kernels/mx_attn_interface.py \
  || { fail "P-C2 python interface (schema-arg fix) not applied"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r242cprebak"

echo "R242C_BUILD_START $(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }
md5sum "$WHEEL" | tee ${WHEEL}.r242c.md5
echo "R242C_BUILD_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
