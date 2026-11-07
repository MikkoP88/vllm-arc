# cert14: cert12 + ARC P6.28 warmup gate v2 (engine-exact warm points)
#
# cert13 (same kit patch, first revision) proved the warm itself runs
# cleanly in-engine rank-serialized (214.9 s, both ranks) but the first
# request STILL wedged: the warm called a variant the engine does not —
# None segment tensors, use_td unset, plain descales. The engine's
# TritonAttentionImpl.forward ALWAYS allocates the softmax segment
# buffers, enables tensor-descriptor mode on XPU by default, passes
# .expand()ed per-head descales, and Gemma-4 layers pair the sliding
# window with mm_prefix_clamp_sliding_window=True. Each difference is a
# distinct triton specialization, so the request-time first load stayed
# fresh and wedged (r108q: jit_monitor re-fired, py-spy identical
# SpirvUtils.load_binary frame). cert14 is the revision that replicates
# the engine call signature exactly. OVERLAYS cert12 DIRECTLY (cert13's
# gpu_worker already carries the v2 marks; the updated patcher would
# no-op against it) — cert13 is interim/superseded, manifest chain
# cert12 -> cert14.
#
# Build (rig):  docker build -f cert14.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert14 <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert12
FROM ${BASE}

COPY patch_arc_xpu_warmup_gate.py /tmp/patch_arc_xpu_warmup_gate.py

RUN GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_warmup_gate.py "$GW" && \
    grep -c "\[ARC-P628-XPU-WARMUP\]" "$GW" | grep -q "^2$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_worker.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-WARMUP]') == 2, 'mark count'; assert '_arc_xpu_warm_tuned_prefill_kernels(self)' in s, 'warmup call'; assert 'kernel_warmup(self)' in s, 'stock anchor intact'; assert 'softmax_segm_output=segm_out' in s, 'engine-exact segments'; assert 'use_td=True' in s, 'engine-exact use_td'; assert 'mm_prefix_clamp_sliding_window=clamp' in s, 'engine-exact clamp pairing'; print('CERT14_WARMUP_V2_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_warmup_gate.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert14=P6.28 cert12 + ARC prefill-tune warmup gate v2 (engine-exact warm points: segment tensors, use_td, expanded descales, clamp pairing; r108q variant-miss fix; supersedes interim cert13)" >> "$MF" && \
    echo "cert14_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert14_built=${BUILT}" >> "$MF"
