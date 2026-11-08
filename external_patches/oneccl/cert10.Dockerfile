# SPDX-License-Identifier: Apache-2.0
# cert10.Dockerfile — vllm-arc:v0.30.0-b70-cert10 = cert9 + oneCCL
# 2022.0.0+arc1 fp8-wire overlay (P6.24).
#
# The vllm tree is UNCHANGED from cert9 (same fork_sha — proven post-build
# by an exhaustive vllm_gaudi/ md5 walk vs cert9); this layer replaces
# EXACTLY two files (the fat libccl pair — cert9 ships libccl.so.1 AND
# libccl.so.1.0 as two regular files; both get the rebuilt fat lib, the
# thin libccl.so.2 shim stays byte-identical) and appends IMAGE_MANIFEST
# rows. Inputs are all tracked-or-pinned (cert7 reproducibility law):
#   - base: certified cert9 (IMAGE_MANIFEST fork_sha aa2ee17d...)
#   - libccl.so.1.0: produced by tools/build_oneccl_arc.sh from pinned
#     upstream tag 2022.0.0 (super 1318c3aa..., libccl 4ceafd15...) +
#     tracked patches/arc-fp8-allreduce.patch — sha256-pinned below
#   - this Dockerfile (tracked)
#
# Build (rig host, from a context holding stage-arc/libccl.so.1.0 + this
# file — see tools/ar/r99_cert10_gate.sh [0]):
#   docker build -f cert10.Dockerfile \
#     --build-arg LIB_SHA=<sha256 of libccl.so.1.0> \
#     --build-arg PATCH_SHA=<sha256 of arc-fp8-allreduce.patch> \
#     --build-arg BUILT=<utc ts> -t vllm-arc:v0.30.0-b70-cert10 .
#
# Runtime posture: the env trio CCL_SYCL_ALLREDUCE_FP8 / _MIN_COUNT /
# _STATS defaults 0 / 2097152 / 0 — silent OFF in the certified chain
# (knob documented in docs/envs.md; default-off — opt-in per the
# r99 A/B table).
ARG BASE=vllm-arc:v0.30.0-b70-cert9
FROM ${BASE}
ARG LIB_SHA=unknown
ARG PATCH_SHA=unknown
ARG BUILT=unknown
COPY libccl.so.1.0 /opt/venv/lib/libccl.so.1.0
COPY libccl.so.1.0 /opt/venv/lib/libccl.so.1
RUN set -euxo pipefail; \
    grep -aq CCL_SYCL_ALLREDUCE_FP8 /opt/venv/lib/libccl.so.1.0; \
    [ "$(sha256sum /opt/venv/lib/libccl.so.1.0 | cut -d' ' -f1)" = "${LIB_SHA}" ]; \
    [ "$(sha256sum /opt/venv/lib/libccl.so.1   | cut -d' ' -f1)" = "${LIB_SHA}" ]; \
    ls -la /opt/venv/lib/libccl.so.1 /opt/venv/lib/libccl.so.1.0 /opt/venv/lib/libccl.so.2.0; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "oneccl=2022.0.0+arc1 (fp8-wire allreduce overlay, P6.24; env trio CCL_SYCL_ALLREDUCE_FP8/_MIN_COUNT/_STATS default 0/2097152/0 = silent off)"; \
      echo "oneccl_lib_sha256=${LIB_SHA} (fat libccl.so.1 + .so.1.0, both replaced; .so.2 untouched)"; \
      echo "oneccl_patch_sha256=${PATCH_SHA} (external_patches/oneccl/patches/arc-fp8-allreduce.patch v4)"; \
      echo "oneccl_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p624; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p624 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
