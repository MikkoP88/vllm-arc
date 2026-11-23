#!/bin/bash
# r231d_build.sh -- P6.47 r231d: build28 wheel (seqused-tightened
# splits) via the arc7mx builder, incremental (VXK_ARC7MX_CLEAN=0; the
# two patched headers' TUs + includers rebuild -- the .deps/build cache
# from build25 is kept, the arc8 2h-build-death law). Extracts the
# fresh MX artifacts (libattn_kernels_xe_2.so + the _vllm_fa2_C stub,
# the r231b-1 identity law) + md5 into /root/build/p647/r231d/ for the
# offline leg. The pre-r231d wheel is preserved (.r231dprebak).
set -uo pipefail
R=/root/build
SRC=$R/vxk-arc-src
WHEELS=$R/wheels-arc-vxk
WHEEL=$WHEELS/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
LOG=$R/arc7mx_build28_r231d.log
P=$R/p647/r231d

fail() { echo "R231D_ABORT: $*"; }

grep -q "r231d SEQUSED-TIGHTENED SPLITS" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_paged_decode_kernel.hpp \
  || { fail "kernel patch not applied (run r231d_apply.py first)"; exit 1; }
grep -q "r231d MIRROR" \
  $SRC/csrc/xpu/attn/xe_2/mx/mx_reduce_split.hpp \
  || { fail "reduce patch not applied (run r231d_apply.py first)"; exit 1; }
docker inspect vxk-arc3-builder >/dev/null 2>&1 \
  || { fail "builder container missing"; exit 1; }
mkdir -p "$P"
[ -f "$WHEEL" ] && cp "$WHEEL" "$WHEEL.r231dprebak"

echo "R231D_BUILD_START $(date +%H:%M:%S)"
docker exec -i -e VXK_ARC7MX_CLEAN=0 vxk-arc3-builder \
  bash < $R/vxk_arc7mx_inner.sh > "$LOG" 2>&1
RC=$?
grep -E "BUILD_EXIT rc" "$LOG" || true
[ "$RC" -eq 0 ] || { fail "build rc=$RC (see $LOG)"; exit 1; }
[ -f "$WHEEL" ] || { fail "wheel not produced"; exit 1; }

# extract the fresh MX artifacts (build28 identity for the offline
# leg) -- via the committed r231b extractor (generic: exact-name wheel,
# libattn + fa2 stub; inline-python heredocs die on nested quoting)
docker cp "$P/r231b_extract.py" vxk-arc3-builder:/tmp/r231b_extract.py \
  || { fail "extractor ship failed"; exit 1; }
docker exec vxk-arc3-builder bash -c "rm -rf /tmp/r231b_so && /opt/arcvenv/bin/python /tmp/r231b_extract.py" \
  || { fail "extract failed"; exit 1; }
docker cp vxk-arc3-builder:/tmp/r231b_so/vllm_xpu_kernels/. "$P/" \
  || { fail "extract copyout failed"; exit 1; }
SO="$P/libattn_kernels_xe_2.so"
[ -f "$SO" ] && [ -f "$P/_vllm_fa2_C.abi3.so" ] \
  || { fail "so extraction failed"; exit 1; }
md5sum "$SO" "$P/_vllm_fa2_C.abi3.so" | tee "$P/build28_so.md5"
echo "R231D_BUILD_DONE $(date +%H:%M:%S) so=$SO"
