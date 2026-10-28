#!/bin/bash
# build_vxk_wheel_arc.sh — rebuild vllm-xpu-kernels from the int64-patched
# v0.1.14 source, torch-matched to the arc image stack (torch 2.13.0+xpu —
# never reuse a wheel across torch ABIs, wedgefix-v88 law / plan §6).
#
# Port of donor build_vxk_wheel_v88.sh (iexperimental kit, read-only import;
# that build used torch 2.11.0+xpu for the 0.26 pin — this one pins OUR
# stack). Version is forced to 0.1.14.1+arc1 so the fixed wheel is
# unmistakable in pip freeze and can never be confused with the PyPI
# 0.1.14.1 release it replaces.
#
# Usage (rig host):  bash build_vxk_wheel_arc.sh <patched-src-root> <out-dir>
set -eo pipefail

SRC_ROOT=${1:?usage: build_vxk_wheel_arc.sh <patched-src-root> <out-dir>}
WHEELS=${2:?usage: build_vxk_wheel_arc.sh <patched-src-root> <out-dir>}
WHEELS=$(readlink -f "$WHEELS")
mkdir -p "$WHEELS"

docker run --rm \
  -v "$(readlink -f "$SRC_ROOT")":/src/vllm-xpu-kernels \
  -v "$WHEELS":/wheels \
  -e KERNELS_MAX_JOBS=52 \
  intel/omix:0.1.0-devel-ubuntu24.04 \
  bash -c '
    set -exo pipefail
    source /opt/intel/oneapi/setvars.sh --force
    # No apt: the omix devel image ships git/curl/cmake 3.28/pkg-config, and
    # the rig route to archive.ubuntu.com stalled the first build attempt
    # (rig-proven today). ninja comes from PyPI via requirements.txt;
    # numactl is a runtime serving tool, not a build dep.
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    # Managed python (headers + libpython bundled) — system python3 has no
    # Python.h in this image and python3-dev would need the wedged apt.
    uv venv /opt/venv --python 3.12 --python-preference only-managed
    export VIRTUAL_ENV=/opt/venv
    export PATH=/opt/venv/bin:$PATH
    uv pip install pip wheel setuptools
    # torch-matched to the arc image stack (Dockerfile pin, P0 ground truth)
    uv pip install torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
    # torch 2.13 imports its bundled sycl/UR (LIBUR_LOADER_0.12, urDeviceWaitExp
    # via the pip intel-cmplr set in /opt/venv/lib); omix oneAPI setvars put an
    # older loader first and the setup.py `import torch` died on the symbol.
    # Venv libs FIRST (same ordering law as serve_arc_b70.sh LD_LIBRARY_PATH).
    export LD_LIBRARY_PATH="/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
    cd /src/vllm-xpu-kernels
    rm -rf build vllm_xpu_kernels.egg-info dist
    # PyPI stays primary (cmake>=3.26 lives there), xpu index allowed for
    # torch-family locals; without unsafe-best-match uv pins cmake to the
    # 3.25.0 on the pytorch index and the resolve fails (rig-proven today)
    uv pip install -r requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/xpu \
      --index-strategy unsafe-best-match
    export CMAKE_PREFIX_PATH="$(python3 -c "import site; print(site.getsitepackages()[0])"):${CMAKE_PREFIX_PATH:-}"
    # fixed-wheel identity: never confusable with the PyPI 0.1.14.1 release
    export SETUPTOOLS_SCM_PRETEND_VERSION=0.1.14.1+arc1
    MAX_JOBS=$KERNELS_MAX_JOBS CMAKE_BUILD_PARALLEL_LEVEL=$KERNELS_MAX_JOBS \
      pip wheel --no-build-isolation --no-deps . -w /wheels
    ls -la /wheels/
  '

echo "BUILD-OK: $(ls "$WHEELS")"
