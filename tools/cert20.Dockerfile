# SPDX-License-Identifier: Apache-2.0
# cert20.Dockerfile — vllm-arc:v0.30.0-b70-cert20 = cert19 + the P6.30
# MXFP4 W4A16 route: vllm-xpu-kernels 0.1.14.1+arc4 (new op
# _xpu_C.fp4_gemm_w4a16, sources external_patches/vllm-xpu-kernels/
# arc4_fp4_gemm_w4a16/csrc/) + the wheel-side reroute patch
# (vllm_gaudi/xpu/patches/mxfp4_w4a16.py, registered in the repo
# platform.py's _XPU_PATCH_MODULES; ARC_MXFP4_MODE gate: unset/w4a16 =
# route, w4a4 = stock control).
#
# Construction law (cert11/cert12/cert18/cert19 overlay precedent,
# minimal delta): the certified chain advances by OVERLAY on the previous
# certified image — every other byte carries byte-identical from cert19.
# This layer changes EXACTLY: the vllm-xpu-kernels wheel (arc3→arc4; ABI
# law unchanged — same builder toolchain oneAPI 2026.0.0, .deps kept),
# vllm_gaudi/xpu/platform.py (one _XPU_PATCH_MODULES entry), and adds
# vllm_gaudi/xpu/patches/mxfp4_w4a16.py.
#
# Why (P6.30 root cause, results/2026-11-08-p630-mxfp4-perf/): stock
# online-MXFP4 is W4A4 — per-call activation quant (≈ the GEMM cost at
# decode M) AND a joint (f4,f4) dtype that can never pass oneDNN
# gemmstone's JIT gate (v3.13 pd.cpp wei_decomp(): f4 not in the
# counter-operand list) → matmul:ref. W4A16 keeps bf16 activations:
# jit:gemm DPAS via weight decomposition (banked verbose line),
# 1.7–2.5× faster than the bf16 GEMM at decode Ms, and e2e vs the
# same-image online-mxfp8 reference: U8k TTFT 1.53× / U32k 1.47× /
# solo decode 1.75× / agg8 1.55× FASTER (r120c legs).
#
# The arc4 wheel is 352 MB — NEVER tracked (GitHub 100 MB artifact law);
# it lives at rig /root/build/wheels-arc-vxk/ and its sha256 is banked in
# the IMAGE_MANIFEST row + the results README. Build context must place
# it at ./vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl.
#
# Inputs (cert7 reproducibility law):
#   - base: certified cert19 (IMAGE_MANIFEST rows carry forward)
#   - rig wheel vllm_xpu_kernels-0.1.14.1+arc4 (sha256 in manifest row)
#   - repo vllm_gaudi/xpu/platform.py + patches/mxfp4_w4a16.py (tracked)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + wheel + the two overlay files):
#   docker build -f cert20.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> \
#     --build-arg WHEEL_SHA256=<sha256 of the arc4 wheel> \
#     -t vllm-arc:v0.30.0-b70-cert20 .
ARG BASE=vllm-arc:v0.30.0-b70-cert19
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
ARG WHEEL_SHA256=unknown
COPY vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl /tmp/vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl
COPY mxfp4_w4a16.py /tmp/mxfp4_w4a16.py
COPY platform.py /tmp/platform.py
RUN set -euxo pipefail; \
    echo "${WHEEL_SHA256}  /tmp/vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl" | sha256sum -c -; \
    /opt/venv/bin/pip install --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl; \
    cp /tmp/mxfp4_w4a16.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mxfp4_w4a16.py; \
    cp /tmp/platform.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/__pycache__/mxfp4_w4a16.cpython-312.pyc; \
    rm -f /tmp/vllm_xpu_kernels-0.1.14.1+arc4-cp312-cp312-linux_x86_64.whl /tmp/mxfp4_w4a16.py /tmp/platform.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import importlib.metadata as md; \
       v = md.version('vllm-xpu-kernels'); \
       assert v == '0.1.14.1+arc4', 'vxk version ' + v; \
       import torch; \
       import vllm_xpu_kernels._xpu_C; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a16'), 'w4a16 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm'), 'stock fp4 op carried'; \
       p = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py', encoding='utf-8').read(); \
       assert p.count('mxfp4_w4a16') >= 1, 'platform registration'; \
       m = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mxfp4_w4a16.py', encoding='utf-8').read(); \
       assert 'ARC_MXFP4_MODE' in m, 'mode gate'; \
       assert 'fp4_gemm_w4a16' in m, 'reroute target'; \
       print('CERT20_W4A16_LIVE wheel+patch+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert20=P6.30 MXFP4 W4A16 route: vllm-xpu-kernels 0.1.14.1+arc4 (new _xpu_C.fp4_gemm_w4a16 — bf16 activations, f4 weights, group-32 e8m0 weight scales, oneDNN jit:gemm weight-decomp path where stock W4A4 lands matmul:ref) + vllm_gaudi/xpu/patches/mxfp4_w4a16.py reroute (ARC_MXFP4_MODE gate, w4a4 control preserved); e2e vs same-image online-mxfp8: U8k 1.53x / U32k 1.47x / solo 1.75x / agg8 1.55x faster (r120c, results/2026-11-08-p630-mxfp4-perf/)"; \
      echo "cert20_fork_sha=${FORK_SHA}"; \
      echo "cert20_built=${BUILT}"; \
      echo "cert20_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p630; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p630 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
