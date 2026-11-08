# SPDX-License-Identifier: Apache-2.0
# cert22.Dockerfile — vllm-arc:v0.30.0-b70-cert22 = cert21 + the P6.33
# skinny-M kernel family: vllm-xpu-kernels 0.1.14.1+arc6 (new ops
# _xpu_C.skinny_fp8_gemm_w8a16 / skinny_fp8_gemm_w8a16_mxfp8 /
# skinny_fp4_gemm_w4a16 — GEMV-regime M<=8 weight-only GEMMs, register
# dequant, deterministic fixed-order split-K; sources
# external_patches/vllm-xpu-kernels/arc6_skinny_v017bp/csrc/) + W4A8
# dispatch case arms for the arc5 enum values + the v0.1.17 Option-B
# backports #611/#645/#650/#600, plus the wheel-side route patch
# (vllm_gaudi/xpu/patches/skinny_m.py; ARC_SKINNY_M gate: unset/on =
# route, off = A/B control; per-lane WARN + permanent fallback),
# platform.py (one _XPU_PATCH_MODULES entry, LAST in the chain) and
# envs.py (ARC_SKINNY_M registry entry).
#
# Construction law (cert11+ overlay precedent, minimal delta): this
# layer changes EXACTLY: the vllm-xpu-kernels wheel (arc5→arc6; ABI law
# unchanged — same builder toolchain oneAPI 2026.0.0, .deps kept),
# vllm_gaudi/xpu/patches/skinny_m.py (new),
# vllm_gaudi/xpu/platform.py (one entry),
# vllm_gaudi/envs.py (one registry entry).
#
# Why (P6.33, results/2026-11-08-p633-skinny-m/): at decode Ms (1..8 =
# batch x spec tokens) every stock oneDNN weight-only route measured
# 3–10.7x above the pure-weights bytes floor (r123 kernel matrix:
# qwen.up f8 floor 58 us vs w16pt 179 / w16pc 218 / m816 327 /
# g16 310 us) — the joint-quant primitives lower to M-tiled gemm
# kernels sized for large M. The skinny kernels replace the M<=8 calls
# only (python M-split): one work-item streams TN weight columns
# K-contiguously, dequantizes in registers (bit-exact f8 exponent
# re-bias, e2m1 value set, e8m0 exp2 group scales), accumulates fp32,
# fixed-order split-K. Larger-M behavior is byte-identical to cert21.
# Gates: r126 GATE-1 correctness (rel < 2e-2 vs torch dequant refs) +
# GATE-2 identical-input A/B + GATE-3 W4A8 dispatch answer; r127 e2e
# five legs (qwen fp8/mxfp8/mxfp4 + gemma mxfp8/mxfp4) read against the
# banked cert21/cert20 controls.
#
# The arc6 wheel is ~352 MB — NEVER tracked (GitHub 100 MB artifact
# law); it lives at rig /root/build/wheels-arc-vxk/ and its sha256 is
# banked in the IMAGE_MANIFEST row + the results README. Build context
# must place it at
# ./vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl.
#
# Inputs (cert7 reproducibility law):
#   - base: certified cert21 (IMAGE_MANIFEST rows carry forward)
#   - rig wheel vllm_xpu_kernels-0.1.14.1+arc6
#     (sha256 filled at close-out)
#   - repo vllm_gaudi/xpu/platform.py + patches/skinny_m.py +
#     vllm_gaudi/envs.py (tracked)
#   - this Dockerfile (tracked)
#
# Build (rig host, context = this file + wheel + the three overlay files):
#   docker build -f cert22.Dockerfile \
#     --build-arg FORK_SHA=<tag commit sha> \
#     --build-arg BUILT=<utc ts> \
#     --build-arg WHEEL_SHA256=<arc6 wheel sha256> \
#     -t vllm-arc:v0.30.0-b70-cert22 .
ARG BASE=vllm-arc:v0.30.0-b70-cert21
FROM ${BASE}
ARG FORK_SHA=unknown
ARG BUILT=unknown
ARG WHEEL_SHA256=unknown
COPY vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl /tmp/vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl
COPY skinny_m.py /tmp/skinny_m.py
COPY platform.py /tmp/platform.py
COPY envs.py /tmp/envs.py
RUN set -euxo pipefail; \
    echo "${WHEEL_SHA256}  /tmp/vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl" | sha256sum -c -; \
    /opt/venv/bin/pip install --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl; \
    cp /tmp/skinny_m.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/skinny_m.py; \
    cp /tmp/platform.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py; \
    cp /tmp/envs.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/platform.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/__pycache__/skinny_m.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/__pycache__/envs.cpython-312.pyc; \
    rm -f /tmp/vllm_xpu_kernels-0.1.14.1+arc6-cp312-cp312-linux_x86_64.whl /tmp/skinny_m.py /tmp/platform.py /tmp/envs.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import importlib.metadata as md; \
       v = md.version('vllm-xpu-kernels'); \
       assert v == '0.1.14.1+arc6', 'vxk version ' + v; \
       import torch; \
       import vllm_xpu_kernels._xpu_C; \
       assert hasattr(torch.ops._xpu_C, 'skinny_fp8_gemm_w8a16'), 'w16 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'skinny_fp8_gemm_w8a16_mxfp8'), 'm816 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'skinny_fp4_gemm_w4a16'), 'g16 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'fp8_gemm_w8a16_mxfp8'), 'p631 op not carried'; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a16'), 'p630 op not carried'; \
       p = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py', encoding='utf-8').read(); \
       assert p.count('skinny_m') >= 1, 'platform registration'; \
       e = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py', encoding='utf-8').read(); \
       assert 'ARC_SKINNY_M' in e, 'envs registration'; \
       m = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/skinny_m.py', encoding='utf-8').read(); \
       assert 'ARC_SKINNY_M' in m and 'skinny_fp8_gemm_w8a16' in m, 'mode gate / route target'; \
       print('CERT22_SKINNY_LIVE wheel+patches+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "cert22=P6.33 skinny-M kernel family: vllm-xpu-kernels 0.1.14.1+arc6 (new _xpu_C.skinny_{fp8_gemm_w8a16,fp8_gemm_w8a16_mxfp8,fp4_gemm_w4a16} GEMV-regime M<=8 kernels + W4A8 dispatch arms + v0.1.17 Option-B backports 611/645/650/600) + patches/skinny_m.py M<=8 route (ARC_SKINNY_M gate, off = A/B control; per-lane fallback) + platform/envs entries; larger-M byte-identical to cert21 (results/2026-11-08-p633-skinny-m/)"; \
      echo "cert22_fork_sha=${FORK_SHA}"; \
      echo "cert22_built=${BUILT}"; \
      echo "cert22_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p633; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p633 /etc/vllm-arc/IMAGE_MANIFEST; \
    cat /etc/vllm-arc/IMAGE_MANIFEST
