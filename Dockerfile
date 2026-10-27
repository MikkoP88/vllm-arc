# vllm-arc:v0.30.0-b70 — Intel Arc Pro B70 (BMG-G31, Xe2) serve image.
#
# Pipeline (plan §8): oneapi-base -> [esimd-build, P2] -> runtime.
#   stage 1 oneapi-base: the validated Intel XPU runtime base — Intel
#     compute runtime, level-zero, oneCCL, /opt/venv python 3.12 — the same
#     base every validated rig image descends from (iexperimental
#     base-v0.26.0 lineage).
#   stage 2 esimd-build: DPC++ 2026.2 (intel/llvm v7.2.0, so.9) kernel build
#     for the ESIMD .so set — lands with p90/p91 at P2. Until then the
#     runtime build stays kernel-less (fallback law: the plugin imports and
#     serves without ESIMD artifacts; docker build --target vllm-arc never
#     pulls the toolchain stage).
#   stage 3 runtime (target `vllm-arc`): vLLM 0.30.0 core from the pinned
#     +xpu wheel (md5-guarded, wheel_guard law) + the venv ground-truth
#     stack (torch 2.13.0+xpu & co. from the pytorch XPU index — exactly the
#     validated radiance freeze) + `pip install .` of the fork dist
#     (package `vllm_gaudi`, shadows upstream's, radiance-proven).
#
# Build (rig): tools/build_image.sh stages context + runs this file.

ARG BASE_IMAGE=vllm-iexperimental:base-v0.26.0

# ---- stage 1: oneapi-base -------------------------------------------------
FROM ${BASE_IMAGE} AS oneapi-base

# ---- stage 3: runtime ------------------------------------------------------
FROM oneapi-base AS vllm-arc

# Stage-scope ARGs: the global BASE_IMAGE above is only visible to FROM —
# re-declare it here or ${BASE_IMAGE} trips set -u in the manifest step.
ARG BASE_IMAGE=vllm-iexperimental:base-v0.26.0
ARG FORK_SHA=unknown
ARG BUILD_HOST=rig

# Clear the base's 0.26-era constraints (they pin torch==2.12.0+xpu and would
# conflict with the 0.30 stack swap below — radiance-proven) and the base's
# VLLM_QUANTIZE_Q40_LIB arc-fork knob (no meaning on 0.30 core; not carried
# by this project — no-unevidenced-customization law, plan §7).
ENV PIP_CONSTRAINT= \
    VLLM_QUANTIZE_Q40_LIB= \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY wheels/ /wheels/arc/
COPY pins.md5 /etc/vllm-arc/pins.md5
COPY tools/wheel_guard.sh /usr/local/bin/wheel_guard.sh

# Stack swap to the validated 0.30 set (venv ground truth,
# results/2026-10-27-p0-venv-groundtruth/): uninstall the base 0.26 stack,
# install pinned wheels + torch 2.13.0+xpu family. Layer-order law: this RUN
# precedes COPY src/ so fork-source changes never invalidate the multi-GB
# stack install — rebuilds after a source edit cache everything but the
# fork dist install + gates (~1 min instead of ~12).
RUN set -euxo pipefail; \
    bash /usr/local/bin/wheel_guard.sh /etc/vllm-arc/pins.md5 /wheels/arc; \
    pip uninstall -y vllm triton-xpu vllm-xpu-kernels; \
    pip install \
        /wheels/arc/vllm-0.30.0+xpu-cp38-abi3-manylinux_2_34_x86_64.whl \
        /wheels/arc/triton-3.7.2+xpu-py3-none-any.whl \
        "torch==2.13.0" \
        "torchvision==0.28.0+xpu" \
        "vllm-xpu-kernels==0.1.14.1" \
        "xgrammar==0.2.8"

# Fork dist on top of the pinned stack (re-installed on every source change).
COPY src/ /build/vllm-arc/
RUN set -euxo pipefail; \
    pip install /build/vllm-arc; \
    pip freeze > /opt/vllm-arc-v0.30.0.constraints.txt; \
    pip check || true

# ---- in-build gates: fail the build on any port-level defect (no GPU here —
# device-dependent probes run in the P0/P1 rig leg, not at build time). ----
RUN set -euxo pipefail; \
    python -c "import vllm, torch; assert vllm.__version__ == '0.30.0', vllm.__version__; assert torch.__version__.startswith('2.13.0'), torch.__version__"; \
    python -c "import triton; assert triton.__version__.startswith('3.7.2'), triton.__version__"; \
    python -c "import vllm_gaudi; print('PLUGIN_IMPORT_OK')"; \
    python -c "from importlib.metadata import entry_points; eps = {e.name: e.value for e in entry_points(group='vllm.platform_plugins')}; assert eps.get('arc') == 'vllm_gaudi.xpu:register', eps; print('ARC_ENTRYPOINT_OK')"; \
    python -c "import vllm; from vllm.platforms import current_platform; print('PLATFORM_RESOLVE_OK', type(current_platform).__name__)"; \
    python -c "from vllm_gaudi.xpu.platform import ArcXpuPlatform, is_xe2_arch; print('ARC_PLATFORM_IMPORT_OK')"

# ---- IMAGE_MANIFEST (governance: every certified image reproducible) ------
RUN set -euxo pipefail; \
    { echo "image=vllm-arc:v0.30.0-b70"; \
      echo "fork_sha=${FORK_SHA}"; \
      echo "build_host=${BUILD_HOST}"; \
      echo "base_image=${BASE_IMAGE}"; \
      echo "vllm_core=0.30.0+xpu $(md5sum /wheels/arc/vllm-0.30.0+xpu-*.whl)"; \
      echo "triton=3.7.2+xpu $(md5sum /wheels/arc/triton-3.7.2+xpu-*.whl)"; \
      echo "stack_freeze=/etc/vllm-arc/venv-freeze.txt"; \
      echo "dpcpp=none (esimd-build stage lands at P2, plan §8)"; \
      echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST; \
    cp /opt/vllm-arc-v0.30.0.constraints.txt /etc/vllm-arc/venv-freeze.txt; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
