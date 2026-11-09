#!/bin/bash
# r128g_aot_compile.sh -- compile a probe with the WHEEL's exact AOT flags
# (CMakeLists SYCL block): spir64_gen + ocloc -device pvc,bmg,bmg-g21-a0,
# bmg-g31-a0 + the spirv-translator ext list. Runs inside vxk-arc3-builder.
set -e
ICPX=/opt/intel/oneapi/compiler/latest/bin/icpx
cd /tmp
OUT=${2:-r128e3_v5_AOT.bin}
$ICPX -fsycl -O3 -DNDEBUG \
  -fsycl-targets=spir64_gen \
  -Xsycl-target-backend=spir64_gen "-device $1" \
  -Xspirv-translator '-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate' \
  "$3" -o "$OUT"
ls -la "$OUT"
