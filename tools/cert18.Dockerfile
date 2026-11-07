# SPDX-License-Identifier: Apache-2.0
# cert18.Dockerfile — vllm-arc:v0.30.0-b70-cert18 = cert17 + the P6.28
# FA2 route for full-attention head-512 layers (external_patches/vllm/
# patch_arc_xpu_fa2_route.py; kit README for provenance and validation).
#
# Construction law (cert11/cert12 overlay precedent, minimal delta): the
# certified chain advances by OVERLAY on the previous certified image —
# every other byte carries byte-identical from cert17.
#
# This layer patches EXACTLY one wheel file in place (vllm/v1/attention/
# ops/triton_unified_attention.py — same file, same surface as the cert12
# launcher-table patch; the wheel pin itself is untouched) and removes the
# stale __pycache__ entry (first import recompiles). The patch adds a
# wrapper-level dispatch: full-attention head-512 calls (gemma-4's 10
# global layers: window left < 0, scalar causal, per-tensor fp8-KV scalar
# descales or fp16-KV, paged block_table) call
# vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func directly
# (the stock ESVD FA2 kernel; same paged-KV layout the triton path
# consumes). Descale form normalized patch-side (v3, r108ae): vxk
# requires 1-element-storage scalar views — the engine's
# layer._k_scale.expand passes through; the cert12 warm gate's
# torch.ones(nkvh).expand gets element 0 cloned into fresh storage.
# Evidence r108ac/r108ad (results/2026-11-06-p628-gemma4/
# phaseC/micro_fa2ab/): 0.3197 s vs 10.81 s per k32k fp8-KV call = 34x
# (24.1 vs 0.71 TF/s), k98k ~30x, decode S8x32k 1.5 ms/call, sliding
# hd256 0.0087 s; numeric guards vs a GQA torch reference green;
# head-512 + fp8-e4m3-KV + descale + softcap compiled in the stock
# _vllm_fa2_C.abi3.so (no fallback fired). Every triton-only feature
# (sliding windows, INT4/per-token-head KV, alibi, sinks, mm_prefix,
# rswa, chunked attention, per-seq causal) keeps the triton kernel.
# Env ARC_ATTN_FA2=0 restores the pure-triton path (escape hatch).
#
# Inputs all tracked-or-pinned (cert7 reproducibility law):
#   - base: certified cert17 (IMAGE_MANIFEST rows carry forward)
#   - external_patches/vllm/patch_arc_xpu_fa2_route.py: tracked
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + the patcher):
#   docker build -f cert18.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> -t vllm-arc:v0.30.0-b70-cert18 .
ARG BASE=vllm-arc:v0.30.0-b70-cert17
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
COPY patch_arc_xpu_fa2_route.py /tmp/patch_arc_xpu_fa2_route.py
RUN set -euxo pipefail; \
    /opt/venv/bin/python /tmp/patch_arc_xpu_fa2_route.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/__pycache__/triton_unified_attention.cpython-312.pyc; \
    rm -f /tmp/patch_arc_xpu_fa2_route.py; \
    /opt/venv/bin/python -c \
      "import vllm.v1.attention.ops.triton_unified_attention as m; \
       src = open(m.__file__, encoding='utf-8').read(); \
       assert src.count('[ARC-P628-FA2]') == 3, 'fa2 marks'; \
       assert src.count('[ARC-P628-XPU-TUNE]') == 3, 'cert12 table still live'; \
       assert 'hd512 full-attn -> vxk FA2' in src, 'boot marker'; \
       assert m._ARC_FA2_ROUTE is True, 'route enable (env-only at import; \
XPU is checked lazily per dispatch — build containers have no GPU)'; \
       assert callable(m._ARC_FA2_IS_XPU), 'lazy xpu check bound'; \
       from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as f; \
       assert callable(f), 'vxk fa2 importable'; \
       print('CERT18_FA2_ROUTE_LIVE dispatch + table OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert18=P6.28 FA2 route: full-attention head-512 calls (gemma-4 global layers) dispatch to the vllm_xpu_kernels ESVD FA2 kernel in the triton unified-attention wrapper — 34x prefill / 1.5 ms-class decode vs the triton 2D path at production geometry (r108ac/r108ad, results/2026-11-06-p628-gemma4/phaseC/micro_fa2ab/); ARC_ATTN_FA2=0 escape hatch; external_patches/vllm/patch_arc_xpu_fa2_route.py"; \
      echo "cert18_fork_sha=${FORK_SHA}"; \
      echo "cert18_built=${BUILT}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p628; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p628 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
