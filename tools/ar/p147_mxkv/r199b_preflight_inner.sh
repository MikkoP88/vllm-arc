#!/bin/bash
# r199b_preflight_inner.sh -- container-side preflight compile of the
# single changed TU (mx_paged_decode_xe2.cpp includes BOTH r199b-edited
# headers: mx_dequant.hpp + mx_paged_decode_mainloop.hpp) before the
# full wheel build. Arc8 2h-build-death law: pre-flight the changed TU
# with ninja; on failure the full log is kept (clang first-error law:
# first deduplicated hard error, not the tail).
# Runs INSIDE vxk-arc3-builder: docker cp to /preflight.sh, then
#   nohup docker exec vxk-arc3-builder bash /preflight.sh > log 2>&1 &
set -uo pipefail

echo "=== VXK r199b PREFLIGHT $(date) ==="
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

export PATH=/opt/arcvenv/bin:$PATH
export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"

icpx --version | head -1
cd /src/vllm-xpu-kernels/build/temp || { echo "FATAL: no ninja tree"; exit 1; }

OBJ=csrc/xpu/attn/xe_2/CMakeFiles/attn_kernels_xe_2.dir/mx/mx_paged_decode_xe2.cpp.o
echo "PREFLIGHT_START $(date)"
ninja "$OBJ"
rc=$?
echo "PREFLIGHT_EXIT rc=$rc $(date)"
exit $rc
