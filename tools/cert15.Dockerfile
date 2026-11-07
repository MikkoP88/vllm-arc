# cert15: cert12 + ARC P6.28 warmup gate v3 (engine-cache warm points)
#
# v3 closes the residual wedge that survived cert13 (variant-miss) and
# cert14 (engine-exact arg signature but synthetic caches). r108s
# (cert14dbg launch-key trace) proved the engine's KV cache layout
# cannot be synthesized: the hybrid KV-cache manager gives different
# layer groups different page sizes (gemma-4: 128-token pages for
# head-512 global layers, 64 for head-256 sliding) and BLOCK_SIZE is a
# triton constexpr — cert14's page-64 warm compiled head-512 kernels
# the engine never calls, so the first head-512 global-layer prefill
# was still a fresh compile that wedged both TP ranks (py-spy:
# SpirvUtils.load_binary under Gemma4Attention.forward; head-256
# prefill was a clean cache hit). v3 takes the per-layer caches from
# model_runner.kv_caches, derives k/v exactly like
# TritonAttentionImpl.forward (transpose(1,2) + split(head) + fp8
# view), and sizes the block table at the engine's real stride bucket
# (max_model_len // page + 1) — warm calls differ from engine calls
# only in tensor values, never in cache keys. OVERLAYS cert12 DIRECTLY
# (cert13/cert14 gpu_workers already carry the marks; the updated
# patcher would no-op) — both interim; manifest chain cert12 -> cert15.
#
# Build (rig):  docker build -f cert15.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert15 <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert12
FROM ${BASE}

COPY patch_arc_xpu_warmup_gate.py /tmp/patch_arc_xpu_warmup_gate.py

RUN GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_warmup_gate.py "$GW" && \
    grep -c "\[ARC-P628-XPU-WARMUP\]" "$GW" | grep -q "^2$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_worker.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-WARMUP]') == 2, 'mark count'; assert '_arc_xpu_warm_tuned_prefill_kernels(self)' in s, 'warmup call'; assert 'kernel_warmup(self)' in s, 'stock anchor intact'; assert 'kv_caches' in s, 'engine-cache warm'; assert 'transpose(1, 2)' in s, 'engine view derivation'; print('CERT15_WARMUP_V3_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_warmup_gate.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert15=P6.28 cert12 + ARC prefill-tune warmup gate v3 (engine-cache warm points from model_runner.kv_caches: real page sizes/strides/dtypes per hybrid layer group; r108s BLOCK_SIZE-constexpr fix; supersedes interim cert13/cert14)" >> "$MF" && \
    echo "cert15_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert15_built=${BUILT}" >> "$MF"
