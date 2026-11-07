# cert14dbg: cert14 + ARC P6.28 launch-key trace (kit patch #3,
# DIAGNOSTIC — debug overlay only, NEVER a cert pin candidate).
#
# cert14's engine-exact warm still left one fresh
# kernel_unified_attention compile at the first 98-token request
# (r108r: jit_monitor warn 4 s after health on TP0; watchdog kill at
# 300 s; banked phaseC/cert14_boot/). Static diff of the wrapper's full
# launch list against the warm signature finds no differing dimension,
# so this overlay settles it empirically: every unified_attention call
# logs its complete triton cache-key-relevant tuple (plus the
# JITFunction object id, to catch a call-path/cache-namespace split)
# right before the kernel launch. One wedge boot then yields both the
# warm key-set and the engine's request-time key for an exact diff.
# Trace defaults ON here (ARC_XPU_TRACE_UNIFIED=0 silences); this image
# exists only for the r108s diagnostic leg.
#
# Build (rig):  docker build -f cert14dbg.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert14dbg <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert14
FROM ${BASE}

COPY patch_arc_xpu_attn_trace.py /tmp/patch_arc_xpu_attn_trace.py

RUN TUA=/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_attn_trace.py "$TUA" && \
    grep -c "\[ARC-P628-XPU-TRACE\]" "$TUA" | grep -q "^1$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/__pycache__/triton_unified_attention.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-TRACE]') == 1, 'trace mark'; assert 'ARCXPU_TRACE fnid=' in s, 'trace line'; print('CERT14DBG_TRACE_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_attn_trace.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert14dbg=P6.28 cert14 + unified-attention launch-key trace (DIAGNOSTIC ONLY, not a cert candidate; r108r residual-miss discrimination)" >> "$MF" && \
    echo "cert14dbg_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert14dbg_built=${BUILT}" >> "$MF"
