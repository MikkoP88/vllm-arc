# cert15dbg2: cert15dbg + triton cache-key trace (kit patch #4,
# DIAGNOSTIC — second debug overlay, NEVER a cert pin candidate).
#
# r108u (banked cert15dbg_boot/) proved the residual miss empirically:
# the warm gate issued the engine-exact (98, 98) sliding head-256 call
# at 04:43:16 (~1 s, IGC-cache-served) and the engine's first
# request-time call at 05:09:15 — byte-identical on every patch-#3
# traced dimension — cache-MISSED inside JITFunction.run and compiled
# fresh for ~288 s until the 300 s RPC watchdog killed the step (jit
# warn 05:09:19; py-spy in optimize_llvm_mod). The differing cache-key
# input is therefore UNTRACED: binder specialization attrs (pointer
# alignment, integer divisibility) or the options object. This overlay
# instruments the authority itself: one ARCJIT_KEY line per
# JITFunction.run call at the kernel_cache lookup (computed key,
# hit/miss, per-parameter type:attr spec, per-argument data_ptr() % 16,
# warps/stages), env-gated ARC_XPU_TRACE_JITKEY — baked ON here, and
# OFF by default everywhere else. One boot then yields the warm key-set
# and the request key in one serve log; the differing dimension reads
# straight off the diff.
#
# Build (rig):  docker build -f cert15dbg2.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert15dbg2 <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert15dbg
FROM ${BASE}

COPY patch_arc_xpu_jitkey_trace.py /tmp/patch_arc_xpu_jitkey_trace.py

ENV ARC_XPU_TRACE_JITKEY=1

RUN JIT=/opt/venv/lib/python3.12/site-packages/triton/runtime/jit.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_jitkey_trace.py "$JIT" && \
    grep -c "\[ARC-P628-XPU-JITKEY\]" "$JIT" | grep -q "^3$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/triton/runtime/__pycache__/jit.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; py_compile.compile('$JIT', doraise=True); import os; assert os.environ['ARC_XPU_TRACE_JITKEY'] == '1', 'env'; print('CERT15DBG2_JITKEY_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_jitkey_trace.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert15dbg2=P6.28 cert15dbg + triton JIT cache-key/hit trace (DIAGNOSTIC ONLY, not a cert candidate; r108u residual-miss key-dim discrimination)" >> "$MF" && \
    echo "cert15dbg2_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert15dbg2_built=${BUILT}" >> "$MF"
