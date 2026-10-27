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
    apt-get update -y && apt-get install -y --no-install-recommends \
        git curl ca-certificates python3-dev ninja-build cmake pkg-config numactl
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=/root/.local/bin:$PATH
    uv venv /opt/venv
    export VIRTUAL_ENV=/opt/venv
    export PATH=/opt/venv/bin:$PATH
    uv pip install pip wheel setuptools
    # torch-matched to the arc image stack (Dockerfile pin, P0 ground truth)
    uv pip install torch==2.13.0+xpu --index-url https://download.pytorch.org/whl/xpu
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
