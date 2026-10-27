#!/bin/bash
# vxk_arc2_inner.sh — in-container build script (runs INSIDE the arc2 builder
# container, mounted at /inner.sh; driven by build_vxk_wheel_arc2.sh).
#
# Env law = W14-validated recipe (rig /src/esimd/w14_build.sh) adapted to the
# arc image venv: DPC++ 7.2.0 drivers as icpx/icx, SYCL_HOME=/src/dpcpp,
# venv-first LD_LIBRARY_PATH with dpcpp libs second, TORCH_XPU_ARCH_LIST=bmg.
# vxk CMakeLists.txt appends -fsycl / -fsycl-targets=spir64_gen itself, so the
# plain clang drivers suffice (no wrapper needed).
set -euo pipefail

export PATH=/src/dpcpp/bin:/opt/venv/bin:$PATH
export CC=/src/dpcpp/bin/clang
export CXX=/src/dpcpp/bin/clang++
export SYCL_HOME=/src/dpcpp
# vxk cmake/toolchain.cmake reads $ENV{CMPLR_ROOT} and derives
# {CMAKE_C,CXX}_COMPILER as ${CMPLR_ROOT}/bin/{icx,icpx} — without this the
# empty root yields /bin/icpx and CMake dies with "not a full path".
export CMPLR_ROOT=/src/dpcpp
# torch share/cmake FindSYCLToolkit picks its parser from $ENV{XPU_SYCL_COMPILER}
# (public/xpu.cmake:13; default "icx"). The icx parser expects an Intel(R)
# oneAPI banner and CMake math()-dies on the raw clang banner this toolchain
# prints. The toolchain IS the distro-style DPC++: it ships dpclang++ whose
# --version banner ("DPC++ compiler 7.2.0 ...") matches the dpclang parser
# (mapped 2019+7 = oneAPI 2026.2) AND lib/pkgconfig/sycl-dpcpp-7.pc — exactly
# the module that branch queries. This is torch's own supported route.
export XPU_SYCL_COMPILER=dpclang
export PKG_CONFIG_PATH=/src/dpcpp/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}
# venv-first law: the venv runtime (intel_sycl_rt 2026.0.0, libsycl so.9) is
# what the produced wheel must link and later load against; toolchain libs
# second. This is the whole point of arc2 (arc1 linked so.8 from omix oneAPI
# 2025.3 and ImportError-d at load).
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:/src/dpcpp/lib
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=52
export KERNELS_MAX_JOBS=52
export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc2

echo "=== VXK_ARC2 toolchain probe $(date) ==="
$SYCL_HOME/bin/icpx -v 2>&1 | head -8
echo "--- which ---"
which icx icpx cmake ninja python pip
cmake --version | head -1
ninja --version
/opt/venv/bin/python -c "import torch; print('torch', torch.__version__)"
echo "=== probe done ==="

cd /src/vllm-xpu-kernels
rm -rf build
rm -rf ./*.egg-info
echo "VXK_ARC2_BUILD_START $(date)"

set +e
/opt/venv/bin/pip wheel --no-build-isolation --no-deps . -w /wheels 2>&1
rc=$?
set -e

echo "VXK_ARC2_BUILD_EXIT rc=$rc $(date)"
ls -la /wheels/
exit "$rc"
