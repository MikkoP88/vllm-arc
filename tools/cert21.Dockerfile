# SPDX-License-Identifier: Apache-2.0
# cert21.Dockerfile — vllm-arc:v0.30.0-b70-cert21 = cert20 + the P6.31
# MXFP8 W8A16 route: vllm-xpu-kernels 0.1.14.1+arc5 (new ops
# _xpu_C.fp8_gemm_w8a16_mxfp8 + _xpu_C.fp4_gemm_w4a8, sources
# external_patches/vllm-xpu-kernels/arc5_mxfp8_lanes/csrc/) + the
# wheel-side reroute patch (vllm_gaudi/xpu/patches/mxfp8_w8a16.py,
# registered in the repo platform.py's _XPU_PATCH_MODULES; ARC_MXFP8_MODE
# gate: unset/w8a16 = route, w8a8 = stock control), the mxfp4_w4a16.py
# update (w4a8 probe branch), and platform.py (one entry).
#
# Construction law (cert11/cert12/cert18/cert19/cert20 overlay precedent,
# minimal delta): the certified chain advances by OVERLAY on the previous
# certified image — every other byte carries byte-identical from cert20.
# This layer changes EXACTLY: the vllm-xpu-kernels wheel (arc4→arc5; ABI
# law unchanged — same builder toolchain oneAPI 2026.0.0, .deps kept),
# vllm_gaudi/xpu/patches/mxfp8_w8a16.py (new),
# vllm_gaudi/xpu/patches/mxfp4_w4a16.py (w4a8 probe branch),
# vllm_gaudi/xpu/platform.py (one _XPU_PATCH_MODULES entry).
#
# Why (P6.31, results/2026-11-08-p631-mxfp8-w8a16/): stock online-MXFP8
# is W8A8 — per-call activation quant plus the joint (f8,f8) dtype that
# lands the slower oneDNN lane. W8A16 keeps bf16 activations against f8
# weights with group-32 e8m0 weight scales. r123 kernel matrix: the
# m816 op is 1.74×/2.68× faster than stock online-mxfp8 GEMM at
# decode/prefill Ms (rel 2e-3) but 1.50×/1.06× slower than the w16pc
# checkpoint-fp8 champion class — so e2e the route closes most of the
# online-quant gap WITHOUT catching the checkpoint: qwen w8a16 vs w8a8
# stock (r124): U8k TTFT 1.55× / U32k 1.47× / solo decode 1.53× /
# agg8 1.37× FASTER, spec gate PASS (pos1 0.9804 vs 0.9758); vs the
# checkpoint-fp8 champion it stays ~1.5× behind at solo decode →
# ADJUDICATION: checkpoint-fp8 remains the perf default;
# ARC_MXFP8_MODE=w8a16 is the online-mxfp8 lane route. The W4A8 joint
# (e4m3,f4) probe is a banked closed-negative: oneDNN primitive-create
# refuses it ("Only support int4 and fp8 gemm", r123). Gemma e2e rows
# (r124b, ARC_MAXLEN=40960 capacity accommodation — online-mxfp8 weights
# are 2× mxfp4 on the 31B model, KV pool at gmu 0.8 cannot hold a 256K
# seq): see the results README.
#
# The arc5 wheel is 352 MB — NEVER tracked (GitHub 100 MB artifact law);
# it lives at rig /root/build/wheels-arc-vxk/ and its sha256 is banked in
# the IMAGE_MANIFEST row + the results README. Build context must place
# it at ./vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl.
#
# Inputs (cert7 reproducibility law):
#   - base: certified cert20 (IMAGE_MANIFEST rows carry forward)
#   - rig wheel vllm_xpu_kernels-0.1.14.1+arc5
#     (sha256 219268cf7d24ebc7a99e5afec5097edd63d88ef26255511c589da71f3615399a)
#   - repo vllm_gaudi/xpu/platform.py + patches/mxfp8_w8a16.py +
#     patches/mxfp4_w4a16.py (tracked)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + wheel + the three overlay files):
#   docker build -f cert21.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> \
#     --build-arg WHEEL_SHA256=219268cf7d24ebc7a99e5afec5097edd63d88ef26255511c589da71f3615399a \
#     -t vllm-arc:v0.30.0-b70-cert21 .
ARG BASE=vllm-arc:v0.30.0-b70-cert20
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
ARG WHEEL_SHA256=unknown
COPY vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl /tmp/vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl
COPY mxfp8_w8a16.py /tmp/mxfp8_w8a16.py
COPY mxfp4_w4a16.py /tmp/mxfp4_w4a16.py
COPY platform.py /tmp/platform.py
RUN set -euxo pipefail; \
    echo "${WHEEL_SHA256}  /tmp/vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl" | sha256sum -c -; \
    /opt/venv/bin/pip install --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl; \
    cp /tmp/mxfp8_w8a16.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mxfp8_w8a16.py; \
    cp /tmp/mxfp4_w4a16.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mxfp4_w4a16.py; \
    cp /tmp/platform.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/__pycache__/mxfp4_w4a16.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/__pycache__/mxfp8_w8a16.cpython-312.pyc; \
    rm -f /tmp/vllm_xpu_kernels-0.1.14.1+arc5-cp312-cp312-linux_x86_64.whl /tmp/mxfp8_w8a16.py /tmp/mxfp4_w4a16.py /tmp/platform.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import importlib.metadata as md; \
       v = md.version('vllm-xpu-kernels'); \
       assert v == '0.1.14.1+arc5', 'vxk version ' + v; \
       import torch; \
       import vllm_xpu_kernels._xpu_C; \
       assert hasattr(torch.ops._xpu_C, 'fp8_gemm_w8a16_mxfp8'), 'm816 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a8'), 'w4a8 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a16'), 'w4a16 op not carried'; \
       p = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py', encoding='utf-8').read(); \
       assert p.count('mxfp8_w8a16') >= 1, 'platform registration'; \
       m = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mxfp8_w8a16.py', encoding='utf-8').read(); \
       assert 'ARC_MXFP8_MODE' in m and 'fp8_gemm_w8a16_mxfp8' in m, 'mode gate / reroute target'; \
       print('CERT21_W8A16_LIVE wheel+patches+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert21=P6.31 MXFP8 W8A16 route: vllm-xpu-kernels 0.1.14.1+arc5 (new _xpu_C.fp8_gemm_w8a16_mxfp8 + _xpu_C.fp4_gemm_w4a8 — bf16 activations, f8 weights, group-32 e8m0 weight scales; w4a8 is a banked oneDNN-refused closed-negative) + vllm_gaudi/xpu/patches/mxfp8_w8a16.py reroute (ARC_MXFP8_MODE gate, w8a8 control preserved) + mxfp4_w4a16.py w4a8 probe branch; e2e qwen vs w8a8 stock (r124): U8k 1.55x / U32k 1.47x / solo 1.53x / agg8 1.37x faster, spec PASS; checkpoint-fp8 stays perf default (results/2026-11-08-p631-mxfp8-w8a16/)"; \
      echo "cert21_fork_sha=${FORK_SHA}"; \
      echo "cert21_built=${BUILT}"; \
      echo "cert21_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p631; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p631 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
