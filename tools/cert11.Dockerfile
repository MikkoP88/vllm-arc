# SPDX-License-Identifier: Apache-2.0
# cert11.Dockerfile — vllm-arc:v0.30.0-b70-cert11 = cert10 + the P6.26
# platform patch (ArcXpuPlatform quark + supports_mx, commit 7ea0f27b).
#
# Construction law (cert10.Dockerfile precedent, minimal-delta): the
# certified chain advances by OVERLAY on the previous certified image —
# every other byte (venv incl. the oneCCL 2022.0.0+arc1 fp8-wire overlay
# libs, wheels, lmcache kit) carries byte-identical from cert10, proven
# post-build by an exhaustive vllm_gaudi/ md5 walk. A full
# build_image.sh rebuild was validated in parallel (candidate cert11c:
# full r105 release battery + both P6.26 phase-B legs) but resolves
# transitive wheels from the live dirs (kaldi_native_fbank 1.22.5 ->
# 1.23.0 drift, r105 [0] finding) and drops the oneccl overlay rows —
# the overlay path keeps the certified artifact's freeze byte-identical
# to cert10, so it is the one that ships.
#
# This layer replaces EXACTLY one source file (vllm_gaudi/xpu/platform.py)
# and removes its now-stale __pycache__ entry (first import recompiles;
# the docker-cp retry leg proved runtime compile of this file, r103b).
# Inputs all tracked-or-pinned (cert7 reproducibility law):
#   - base: certified cert10 (IMAGE_MANIFEST fork_sha aa2ee17d + oneccl rows)
#   - vllm_gaudi/xpu/platform.py: tracked file, content = tag rel/v0.30.0
#     -b70-cert11's tree (build-arg FORK_SHA pins the exact commit)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + vllm_gaudi/xpu/platform.py taken
# FROM THE TAG ARCHIVE — see tools/ar/r105_cert11_gate.sh note):
#   docker build -f cert11.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> -t vllm-arc:v0.30.0-b70-cert11 .
ARG BASE=vllm-arc:v0.30.0-b70-cert10
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY platform.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py
RUN set -euxo pipefail; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc; \
    /opt/venv/bin/python -c \
      "from vllm_gaudi.xpu.platform import ArcXpuPlatform as P; \
       assert 'quark' in P.supported_quantization and P.supports_mx(); \
       print('CERT11_FIX_LIVE quark+supports_mx OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert11=P6.26 quark quant + supports_mx on ArcXpuPlatform (MXFP4 lane; vllm_gaudi/xpu/platform.py replaced, commit 7ea0f27b)"; \
      echo "cert11_fork_sha=${FORK_SHA}"; \
      echo "cert11_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p626; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p626 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
