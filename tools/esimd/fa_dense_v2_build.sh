#!/bin/bash
# fa_dense_v2_build.sh — P6.21 v2 dense-FA ESIMD kernel build (rig esimd
# container, CPU-only — the W14 incumbent keeps serving on the GPUs).
# Same DPC++ 2026.2 recipe as fa_dense_build.sh (plan Appendix D); only
# v2 sources are touched/rebuilt (the v1 object stays cached, md5-stable).
#
# TOUCH LAW (W14 build6): touch the sources or ninja silently reuses the
# previous .so.
set -euo pipefail
export MAX_JOBS=52 KERNELS_MAX_JOBS=52
export CC=/src/dpcpp/bin/clang CXX=/src/dpcpp/bin/clang++
mkdir -p /src/dpcpp-shim
ln -sf /src/dpcpp/bin/clang++ /src/dpcpp-shim/icpx
ln -sf /src/dpcpp/bin/clang /src/dpcpp-shim/icx
export PATH=/src/dpcpp-shim:$PATH
export LD_LIBRARY_PATH=/src/dpcpp/lib
export TORCH_XPU_ARCH_LIST=bmg
cd /src/esimd-fa
touch csrc/fa_dense_v2.fp8.h csrc/fa_dense_v2.sycl setup_fa_dense.py
mkdir -p python   # package_dir target
/opt/venv/bin/python setup_fa_dense.py build_ext --inplace
echo BUILD_RC=$?
find . -name "fa_dense_v2_ops*.so" -exec md5sum {} \;
echo STAGED
