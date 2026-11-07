# SPDX-License-Identifier: Apache-2.0
# cert12.Dockerfile — vllm-arc:v0.30.0-b70-cert12 = cert11 + the P6.28
# B70 prefill launch-config table (external_patches/vllm/
# patch_arc_xpu_attn_launcher.py, upstream PR#60418 pattern; kit README
# for provenance and validation).
#
# Construction law (cert11.Dockerfile precedent, minimal-delta overlay):
# the certified chain advances by OVERLAY on the previous certified image
# — every other byte (venv incl. the oneCCL 2022.0.0+arc1 fp8-wire overlay
# libs, wheels, lmcache kit, cert11 platform.py) carries byte-identical
# from cert11, proven post-build by an exhaustive vllm_gaudi/ md5 walk.
#
# This layer replaces EXACTLY one wheel file (vllm/v1/attention/ops/
# triton_unified_attention.py — patched in place by the tracked kit
# script; the wheel pin itself is untouched) and removes the stale
# __pycache__ entry (first import recompiles). Inputs all
# tracked-or-pinned (cert7 reproducibility law):
#   - base: certified cert11 (IMAGE_MANIFEST rows carry forward)
#   - external_patches/vllm/patch_arc_xpu_attn_launcher.py: tracked
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + the patcher):
#   docker build -f cert12.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> -t vllm-arc:v0.30.0-b70-cert12 .
ARG BASE=vllm-arc:v0.30.0-b70-cert11
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY patch_arc_xpu_attn_launcher.py /tmp/patch_arc_xpu_attn_launcher.py
RUN set -euxo pipefail; \
    /opt/venv/bin/python /tmp/patch_arc_xpu_attn_launcher.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/__pycache__/triton_unified_attention.cpython-312.pyc; \
    rm -f /tmp/patch_arc_xpu_attn_launcher.py; \
    /opt/venv/bin/python -c \
      "import vllm.v1.attention.ops.triton_unified_attention as m; \
       src = open(m.__file__, encoding='utf-8').read(); \
       assert src.count('[ARC-P628-XPU-TUNE]') == 3; \
       assert m._ARC_XPU_PREFILL_CONFIGS[256][2] == (64, 64, 8, None); \
       assert m._ARC_XPU_PREFILL_CONFIGS[512][2] == (64, 32, 8, None); \
       assert m._arc_xpu_prefill_config(512, 8192) == (64, 32, 8, None); \
       assert m._arc_xpu_prefill_config(512, 1024) == (64, 32, 8, None); \
       assert m._arc_xpu_prefill_config(512, 8) is None; \
       assert m._arc_xpu_prefill_config(128, 8192) is None; \
       print('CERT12_TUNE_LIVE prefill table OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert12=P6.28 B70 prefill launch-config table in triton unified-attention launcher (upstream PR#60418 pattern; external_patches/vllm/patch_arc_xpu_attn_launcher.py)"; \
      echo "cert12_fork_sha=${FORK_SHA}"; \
      echo "cert12_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p628; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p628 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
