# cert15dbg: cert15 + ARC P6.28 launch-key trace (kit patch #3,
# DIAGNOSTIC — debug overlay only, NEVER a cert pin candidate).
#
# cert15's engine-exact warm (v3: the engine's own kv_cache tensors,
# real page sizes/strides, engine block-table stride bucket) reached
# health with all 32 warm loads complete (rank0 1164.8 s fresh, rank1
# cache-hits) — and the engine's own internal 98-token warmup request
# STILL wedged (r108t: jit_monitor warn on kernel_unified_attention
# 5 s after health; RPC sample_tokens timeout at 300 s; banked
# phaseC/cert15_boot/). A full audit of the wrapper launch list
# against the warm signature finds no remaining differing dimension,
# so the trace goes back on for the v3 diff: warm key-set and the
# request-time key in one serve log, plus a wedge-time py-spy that
# also proves kernel identity (unified_attention vs the hybrid's
# other first-request jit kernels). Trace defaults ON here
# (ARC_XPU_TRACE_UNIFIED=0 silences); this image exists only for the
# r108u diagnostic leg.
#
# Build (rig):  docker build -f cert15dbg.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert15dbg <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert15
FROM ${BASE}

COPY patch_arc_xpu_attn_trace.py /tmp/patch_arc_xpu_attn_trace.py

RUN TUA=/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_attn_trace.py "$TUA" && \
    grep -c "\[ARC-P628-XPU-TRACE\]" "$TUA" | grep -q "^1$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/__pycache__/triton_unified_attention.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-TRACE]') == 1, 'trace mark'; assert 'ARCXPU_TRACE fnid=' in s, 'trace line'; print('CERT15DBG_TRACE_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_attn_trace.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert15dbg=P6.28 cert15 + unified-attention launch-key trace (DIAGNOSTIC ONLY, not a cert candidate; r108t residual-miss discrimination)" >> "$MF" && \
    echo "cert15dbg_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert15dbg_built=${BUILT}" >> "$MF"
