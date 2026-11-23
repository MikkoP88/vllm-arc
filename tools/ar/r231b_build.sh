#!/bin/bash
# r231b_build.sh -- P6.47 r231b: build27 wheel (Q pre-stage) via the
# arc7mx builder, incremental (VXK_ARC7MX_CLEAN=0; only the mainloop
# TU + its includers rebuild -- the .deps/build cache from build25 is
# kept, the arc8 2h-build-death law). Extracts the fresh MX artifacts
# (libattn_kernels_xe_2.so -- the .so that carries the MX decode
# kernels -- plus the _vllm_fa2_C stub) + md5 into /root/build/p647/
# r231b/ for the offline leg. The pre-r231b wheel is preserved
# (.r231bprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build27_r231b.log
P=$R/p647/r231b

fail() { echo "R231B_ABORT: $*"; }

grep -q "r231b Q PRE-STAGE" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { fail "patch not applied (run r231b_apply.py first)"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
mkdir -p "$P"
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r231bprebak"

echo "R231B_BUILD_START $(date +%H:%M:%S)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }

# extract the fresh MX artifacts (build27 identity for the offline
# leg) -- via the file-shipped extractor (inline-python heredocs die
# on nested quoting, the r231b-1 lesson)
docker cp "$P/r231b_extract.py" vxk-arc3-builder:/tmp/r231b_extract.py \
  || { fail "extractor ship failed"; exit 1; }
docker exec vxk-arc3-builder bash -c "rm -rf /tmp/r231b_so && /opt/arcvenv/bin/python /tmp/r231b_extract.py" \
  || { fail "extract failed"; exit 1; }
docker cp vxk-arc3-builder:/tmp/r231b_so/vllm_xpu_kernels/. "$P/" \
  || { fail "extract copyout failed"; exit 1; }
SO="$P/libattn_kernels_xe_2.so"
[ -f "$SO" ] && [ -f "$P/_vllm_fa2_C.abi3.so" ] \
  || { fail "so extraction failed"; exit 1; }
md5sum "$SO" "$P/_vllm_fa2_C.abi3.so" | tee "$P/build27_so.md5"
echo "R231B_BUILD_DONE $(date +%H:%M:%S) so=$SO"
