# cert13: cert12 + ARC P6.28 warmup gate (kit patch #2)
#
# cert12 added the B70 prefill launch-config table to the triton
# unified-attention launcher (kit patch #1, external_patches/vllm/
# patch_arc_xpu_attn_launcher.py). r108o then proved the engine-context
# wedge: the FIRST in-engine module load of a tuned config (zeModuleCreate
# via SpirvUtils.load_binary) wedges both TP ranks — graph posture at the
# first real prefill request, eager posture inside compile_or_warm_up
# ("Compile and warming up model for size 8192"). cert13 adds kit patch
# #2 (patch_arc_xpu_warmup_gate.py): a rank-serialized pre-load of every
# tuned kernel point inside Worker.compile_or_warm_up_model, before the
# stock kernel_warmup and cuda-graph capture. No-op on stock wheels and
# uncovered geometries; fail-loud if the warmup itself fails.
#
# Build (rig):  docker build -f cert13.Dockerfile \
#                 --build-arg FORK_SHA=<sha> --build-arg BUILT=<utc> \
#                 -t vllm-arc:v0.30.0-b70-cert13 <ctx with the patcher>
ARG BASE=vllm-arc:v0.30.0-b70-cert12
FROM ${BASE}

COPY patch_arc_xpu_warmup_gate.py /tmp/patch_arc_xpu_warmup_gate.py

RUN GW=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py && \
    /opt/venv/bin/python3 /tmp/patch_arc_xpu_warmup_gate.py "$GW" && \
    grep -c "\[ARC-P628-XPU-WARMUP\]" "$GW" | grep -q "^2$" && \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/__pycache__/gpu_worker.cpython-312.pyc && \
    /opt/venv/bin/python3 -c "import py_compile; src = '/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py'; py_compile.compile(src, doraise=True); s = open(src, encoding='utf-8').read(); assert s.count('[ARC-P628-XPU-WARMUP]') == 2, 'mark count'; assert '_arc_xpu_warm_tuned_prefill_kernels(self)' in s, 'warmup call'; assert 'kernel_warmup(self)' in s, 'stock anchor intact'; assert '_ARC_XPU_PREFILL_CONFIGS' in s, 'table reference'; print('CERT13_WARMUP_LIVE')" && \
    rm -f /tmp/patch_arc_xpu_warmup_gate.py && \
    MF=/etc/vllm-arc/IMAGE_MANIFEST && \
    echo "cert13=P6.28 cert12 + ARC prefill-tune warmup gate (rank-serialized kernel pre-load in compile_or_warm_up_model; r108o wedge fix)" >> "$MF" && \
    echo "cert13_fork_sha=${FORK_SHA}" >> "$MF" && \
    echo "cert13_built=${BUILT}" >> "$MF"
