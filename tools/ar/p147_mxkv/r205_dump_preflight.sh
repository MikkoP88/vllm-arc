#!/bin/bash
# r205 form-dump preflight: compile the mx TU and surface the DumpT error.
docker exec vxk-arc3-builder bash -lc '
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
cd /src/vllm-xpu-kernels/build/temp
ninja csrc/xpu/attn/xe_2/CMakeFiles/attn_kernels_xe_2.dir/mx/mx_paged_decode_xe2.cpp.o 2>&1
' | grep -A2 -B2 'DumpT' | head -60
