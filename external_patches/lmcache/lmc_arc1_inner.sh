#!/bin/bash
# lmc_arc1_inner.sh — in-container lmcache SYCL build (runs INSIDE the
# builder container; driven by build_lmcache_arc.sh). Mirrors vxk_arc3_inner
# laws: setvars guarded against set -u, /opt/arcvenv target-stack venv,
# SYCL_HOME=$CMPLR_ROOT, TORCH_XPU_ARCH_LIST=bmg, full clean.
set -euo pipefail

echo "=== LMC_ARC toolchain probe $(date) ==="
set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -u

if [ ! -x /opt/arcvenv/bin/python ]; then
  python3 -m venv --without-pip /opt/arcvenv
  curl -LsSf https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  /opt/arcvenv/bin/python /tmp/get-pip.py -q
fi
export VIRTUAL_ENV=/opt/arcvenv PATH=/opt/arcvenv/bin:$PATH
if ! /opt/arcvenv/bin/python -c "import torch" >/dev/null 2>&1; then
  pip install -q --upgrade pip wheel setuptools
  pip install -q torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
fi
# lmcache build deps (upstream build_xpu_wheel.sh set + repair tools)
pip install -q ninja "setuptools>=77.0.3,<81.0.0" setuptools_scm \
  grpcio==1.78.0 grpcio-tools==1.78.0 wheel pybind11 auditwheel patchelf

icpx --version | head -1
/opt/arcvenv/bin/python -c "import torch; print('ENV_READY torch', torch.__version__, 'xpu:', hasattr(torch, 'xpu'))"

export SYCL_HOME="${SYCL_HOME:-$CMPLR_ROOT}"
export LD_LIBRARY_PATH=/opt/arcvenv/lib:/opt/arcvenv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
export TORCH_XPU_ARCH_LIST=bmg
export MAX_JOBS=52

cd /src/lmcache
rm -rf build dist dist_xpu lmcache.egg-info
echo "LMC_ARC_BUILD_START $(date)"
export BUILD_WITH_SYCL=1
set +e
/opt/arcvenv/bin/python setup.py bdist_wheel --dist-dir=dist_xpu 2>&1 | tail -25
rc=$?
set -e
echo "LMC_ARC_BUILD_EXIT rc=$rc $(date)"
[ "$rc" = 0 ] || exit "$rc"

echo "=== auditwheel repair ==="
/opt/arcvenv/bin/auditwheel repair \
  --plat manylinux_2_35_x86_64 \
  --exclude 'libtorch*.so*' --exclude 'libc10*.so*' \
  --exclude 'libsycl.so*' --exclude 'libze_loader.so*' \
  --exclude 'libur_loader.so*' --exclude 'libOpenCL.so*' \
  --exclude 'libiomp5.so*' --exclude 'libtbb.so*' \
  --exclude 'libmkl_sycl*.so*' --exclude 'libmkl_intel_ilp64.so*' \
  --exclude 'libmkl_intel_lp64.so*' --exclude 'libmkl_core.so*' \
  -w dist dist_xpu/*.whl

echo "=== XPU extension check (device image in bundle) ==="
python3 -c "
import zipfile, glob
w = glob.glob('dist/*.whl')[0]
zipfile.ZipFile(w).extractall('/tmp/whcheck_lmc')
"
SO=$(find /tmp/whcheck_lmc -name 'xpu_ops*.so')
echo "SO=$SO"
readelf -S -W "$SO" | grep -E '(__CLANG_OFFLOAD_BUNDLE__sycl-|\.llvm\.offloading)' || { echo "FATAL: no SYCL offload bundle"; exit 6; }
readelf -d "$SO" | grep NEEDED || true
echo "=== final wheel ==="
ls -la dist/
cp dist/*.whl /wheels/ 2>/dev/null || true
