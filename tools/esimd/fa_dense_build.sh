#!/bin/bash
# fa_dense_build.sh — p72 dense-FA ESIMD kernel build (rig esimd container).
# Donor: w14_build.sh verbatim recipe (plan Appendix D): DPC++ 2026.2
# (intel/llvm v7.2.0, so.9) + icpx shim + TORCH_XPU_ARCH_LIST=bmg + JIT
# spir64. Run INSIDE the esimd container with the fork's kernel tree
# staged at /src/esimd-fa (tools/esimd/stage_to_rig.sh pushes it).
#
# TOUCH LAW (W14 build6): docker cp preserves mtimes — touch the sources
# or setuptools/ninja silently reuses the previous .so.
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
touch csrc/fa_dense.fp8.h csrc/fa_dense.sycl setup_fa_dense.py
rm -rf build python
mkdir -p python   # package_dir target (donor tree ships one; inplace copy needs it)
/opt/venv/bin/python setup_fa_dense.py build_ext --inplace
echo BUILD_RC=$?
find . -name "fa_dense_ops*.so" -exec md5sum {} \;
echo STAGED
