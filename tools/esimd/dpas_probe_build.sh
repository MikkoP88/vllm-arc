#!/bin/bash
# dpas_probe_build.sh — P6.21 r92 dpas layout probe build (rig esimd
# container, CPU-only). Same recipe as fa_dense_v2_build.sh; only the
# probe source is new (v1/v2 objects stay cached, md5-stable).
# TOUCH LAW (W14 build6): touch the sources or ninja silently reuses.
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
touch csrc/dpas_probe.sycl setup_fa_dense.py
mkdir -p python
/opt/venv/bin/python setup_fa_dense.py build_ext --inplace
echo BUILD_RC=$?
find . -name "dpas_probe_ops*.so" -exec md5sum {} \;
echo STAGED
