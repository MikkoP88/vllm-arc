# cert16: cert12 + ARC P6.28 warmup gate v4 (engine-exact softcap type)
#
# v4 fixes the root cause of the G2 long-prefill engine freeze, source-
# proven and telemetry-proven in r108v (banked: results/2026-11-06-
# p628-gemma4/phaseC/cert15dbg2_boot/): the v3 warm gate called
# kernel_unified_attention with softcap=0.0 (float -> triton signature
# ('fp32', None)) while the engine impl normalizes logits_soft_cap=None
# to int 0 (triton_attn.py:505-508 -> ('i32', 'D')). The scalar type is
# part of the kernel signature and the signature is part of the triton
# cache key — so EVERY warm-compiled kernel lived in a key space the
# engine never looked up. The engine's own init compiles only decode/
# capture-shaped classes (BLOCK_M=16, BLOCK_Q=8/2, segmented); the
# prefill classes (BLOCK_M=64, BLOCK_Q=32/8, non-3D) are compiled
# nowhere in the boot, and the first real request fresh-compiled
# ~288 s (+ first-launch _init_handles/zeModuleCreate — the r108o wedge
# class) inside the 300 s execute-model RPC watchdog -> EngineCore
# death -> HTTP 500. v4 = one-token fix (softcap=0, int) making the
# warm key space equal to the engine's; request-path classes are then
# compiled AND launched during warm, decode/capture classes during
# engine init (pre-health), and the first request is pure cache hits.
#
# Built from cert12 (clean wheel) exactly like cert15 — same patcher,
# updated in place; no corrective edit on a baked wheel. Supersedes
# cert15 (and interim cert13/cert14); manifest chain cert12 -> cert16.
#
# Build (rig):  docker build -f cert16.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert16 <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert12
FROM ${BASE}

COPY patch_arc_xpu_warmup_gate.py /tmp/patch_arc_xpu_warmup_gate.py

RUN GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_warmup_gate.py "$GW" && \
    grep -c "\[ARC-P628-XPU-WARMUP\]" "$GW" | grep -q "^2$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_worker.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-WARMUP]') == 2, 'mark count'; assert '_arc_xpu_warm_tuned_prefill_kernels(self)' in s, 'warmup call'; assert 'kernel_warmup(self)' in s, 'stock anchor intact'; assert 'kv_caches' in s, 'engine-cache warm'; assert 'transpose(1, 2)' in s, 'engine view derivation'; assert 'softcap=0,' in s, 'v4 int softcap'; assert 'softcap=0.0' not in s, 'v3 float softcap purged'; print('CERT16_WARMUP_V4_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_warmup_gate.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert16=P6.28 cert12 + ARC prefill-tune warmup gate v4 (v3 + engine-exact int softcap: float-vs-int scalar type split the triton cache key, leaving every warm kernel in an unused key space and the first prefill fresh-compiling inside the watchdog; r108v root cause; supersedes interim cert13/cert14/cert15)" >> "$MF" && \
    echo "cert16_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert16_built=${BUILT}" >> "$MF"
