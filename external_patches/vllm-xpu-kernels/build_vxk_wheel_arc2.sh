#!/bin/bash
# build_vxk_wheel_arc2.sh — rebuild vllm-xpu-kernels (with the p13 int64
# wedge fix) INSIDE the arc runtime image, with the staged DPC++ 7.2.0
# toolchain. Runs on the rig host; launch under nohup (see p13 README).
#
# Why arc2 (arc1 postmortem, results/2026-10-27-p13-vxk-int64fix + p1 leg):
# arc1 was built in the omix devel image (oneAPI 2025.3) and linked
# libsycl.so.8; the arc image venv runtime is intel_sycl_rt 2026.0.0
# (libsycl.so.9) -> ImportError at wheel load. ABI law: never reuse a wheel
# across SYCL runtime majors — build against the target runtime. The staged
# toolchain /root/build/dpcpp72/dpcpp (DPC++ 7.2.0 = clang 23.1.1,
# intel/llvm 5680988, docker-cp'd from esimd-w11:/src/dpcpp) produces the
# so.9 family — the W14 ESIMD .so it built ran on the radiance-test venv.
#
# Wheel: vllm_xpu_kernels-0.1.14.1+arc2-cp312-cp312-linux_x86_64.whl
# -> /root/build/wheels-arc-vxk/ (mounted /wheels in-container, :rw).
set -euo pipefail

VXK_SRC=/root/build/vxk-arc-src
WHEELS=/root/build/wheels-arc-vxk
DPCPP=/root/build/dpcpp72/dpcpp
INNER=/root/build/vxk_arc2_inner.sh
BASE=vllm-arc:v0.30.0-b70-p12

[ -f "$VXK_SRC/CMakeLists.txt" ] || { echo "FATAL: $VXK_SRC missing/incomplete"; exit 1; }
[ -x "$DPCPP/bin/clang++" ] || { echo "FATAL: $DPCPP toolchain missing"; exit 1; }
[ -f "$INNER" ] || { echo "FATAL: inner script $INNER missing"; exit 1; }
docker image inspect "$BASE" >/dev/null || { echo "FATAL: image $BASE missing"; exit 1; }
mkdir -p "$WHEELS"

# oneAPI-layout compiler symlinks inside OUR PRIVATE toolchain copy (the
# intel/llvm distro ships clang/clang++; setup.py runs $SYCL_HOME/bin/icpx -v
# and resolves CMAKE_*_COMPILER via which icx/icpx — W14 recipe mapping).
ln -sf clang++ "$DPCPP/bin/icpx"
ln -sf clang "$DPCPP/bin/icx"

docker run --rm \
  --name vxk-arc2-builder \
  --device /dev/dri \
  -v "$VXK_SRC":/src/vllm-xpu-kernels \
  -v "$WHEELS":/wheels \
  -v "$DPCPP":/src/dpcpp:ro \
  -v "$INNER":/inner.sh:ro \
  --entrypoint /bin/bash "$BASE" /inner.sh
