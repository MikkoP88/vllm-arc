# SPDX-License-Identifier: Apache-2.0
# cert19.Dockerfile — vllm-arc:v0.30.0-b70-cert19 = cert18 + the P6.29
# FA2 route for sliding-window head-256 layers (external_patches/vllm/
# patch_arc_xpu_fa2_slide_route.py; kit README for provenance and validation).
#
# Construction law (cert11/cert12/cert18 overlay precedent, minimal delta):
# the certified chain advances by OVERLAY on the previous certified image —
# every other byte carries byte-identical from cert18.
#
# This layer patches EXACTLY one wheel file in place (vllm/v1/attention/
# ops/triton_unified_attention.py — the same file cert18 patched; the new
# patcher ASSERTS the cert18 v3 state and refuses anything else) and
# removes the stale __pycache__ entry (first import recompiles). The patch
# adds a SECOND wrapper-level dispatch immediately after the cert18
# hd512-full route: sliding-window head-256 calls (gemma-4's 50/60 local
# layers: window left >= 0 right <= 0, i.e. the (w-1, 0) causal form the
# backend passes; scalar causal; paged block_table; per-tensor fp8-KV
# scalar descales or fp16-KV; every cert18 triton-only exclusion) call
# vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func with
# window_size=[left, 0] (the r108ac/r109a-proven call form; the
# bidirectional encoder form (w-1, w-1) keeps triton).
# Evidence r109a/r109a3/r109a6 (results/2026-11-07-p629/phaseA/):
# triton 0.147-0.158 s/call at q8192 fp8-KV, k-INDEPENDENT (0.1466 k8192 /
# 0.1580 k32k / 0.1576 k98k) while FA2 windowed runs 0.006-0.009 s
# (window-bounded) = 18-25x; decode S8x32k triton 4.38 ms vs FA2 0.3-0.4
# ms = 11-15x; 50 layers x ~0.148 s ~= 7.4 s per q8192 chunk removed.
# Numeric guards: kernels agree 1.2e-4..9.8e-4 (kernel-vs-kernel) and
# 1.3e-4..1.1e-3 prefill / 8e-5 decode vs global-softmax table-faithful
# window-masked GQA torch refs (r109a6; the earlier ~4 ref misses were a
# per-chunk-softmax bug in the TEST refs, both kernels correct — softcap
# also eliminated: the triton wrapper cannot receive None, production
# passes 0.0 and USE_SOFTCAP=(0.0 > 0) skips it). Env ARC_ATTN_FA2=0
# disables BOTH routes (shared cert18 module gate; escape hatch).
#
# Inputs all tracked-or-pinned (cert7 reproducibility law):
#   - base: certified cert18 (IMAGE_MANIFEST rows carry forward)
#   - external_patches/vllm/patch_arc_xpu_fa2_slide_route.py: tracked
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + the patcher):
#   docker build -f cert19.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> -t vllm-arc:v0.30.0-b70-cert19 .
ARG BASE=vllm-arc:v0.30.0-b70-cert18
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY patch_arc_xpu_fa2_slide_route.py /tmp/patch_arc_xpu_fa2_slide_route.py
RUN set -euxo pipefail; \
    /opt/venv/bin/python /tmp/patch_arc_xpu_fa2_slide_route.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/__pycache__/triton_unified_attention.cpython-312.pyc; \
    rm -f /tmp/patch_arc_xpu_fa2_slide_route.py; \
    /opt/venv/bin/python -c \
      "import vllm.v1.attention.ops.triton_unified_attention as m; \
       src = open(m.__file__, encoding='utf-8').read(); \
       assert src.count('[ARC-P629-SLIDE]') == 2, 'slide marks'; \
       assert src.count('[ARC-P628-FA2]') == 3, 'cert18 marks carry'; \
       assert src.count('[ARC-P628-XPU-TUNE]') == 3, 'cert12 table still live'; \
       assert 'hd256 sliding -> vxk FA2' in src, 'slide boot marker'; \
       assert m._ARC_FA2_ROUTE is True, 'route enable (env-only at import; \
XPU is checked lazily per dispatch — build containers have no GPU)'; \
       assert getattr(m, '_ARC_SLIDE_LOGGED', None) is False, 'slide logger bound'; \
       assert callable(m._ARC_FA2_IS_XPU), 'lazy xpu check bound'; \
       from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as f; \
       assert callable(f), 'vxk fa2 importable'; \
       print('CERT19_SLIDE_ROUTE_LIVE dispatch + table OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert19=P6.29 FA2 route: sliding-window head-256 calls (gemma-4's 50/60 local layers, window (1023,0)) dispatch to the vllm_xpu_kernels ESVD FA2 kernel windowed in the triton unified-attention wrapper — 18-25x prefill / 11-15x decode vs the triton path at production geometry, k-independent (r109a/r109a3, results/2026-11-07-p629/phaseA/); guards green vs global-softmax refs (r109a6); ARC_ATTN_FA2=0 disables both routes; external_patches/vllm/patch_arc_xpu_fa2_slide_route.py"; \
      echo "cert19_fork_sha=${FORK_SHA}"; \
      echo "cert19_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p629; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p629 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
