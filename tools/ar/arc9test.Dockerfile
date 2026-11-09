# arc9test.Dockerfile — vllm-arc:v0.30.0-b70-arc9test = cert22 + the
# P6.34 TEST layer (arc6test precedent: validated here first; the
# certified cert23 image is rebuilt FROM cert22 via a tracked
# tools/cert23.Dockerfile at close-out with tag-pinned FORK_SHA — this
# file stays rig-side).
# Changes EXACTLY: the vllm-xpu-kernels wheel (arc6 -> arc9; same
# builder toolchain oneAPI 2026.0.0 — the jm_* op trio gains the
# DUAL-VARIANT hot loop selected per call by ARC_JM_VAR (build9,
# r128k4-k7): 14 = SLM-V24 form (A loaded DIRECT from a launcher-built
# k-blocked f16 activation buffer — never-written memory is race-immune;
# B staged store->barrier->load into per-sg SLM slots [4 sg x 256
# halves], no post-load barrier; SLM staging proven EXACT at ~130
# ns/step over the all-direct floor), 5 = the V5 arm byte-unchanged as
# the gate-proven control (global-scratch form). All arc6 skinny_*/fp*
# ops carried unchanged), plus patches/skinny_m.py (ARC_SKINNY_M gains
# mode "jm": lane detect -> _jm_pack repack/scale/bias pad cache ->
# jm op -> [..., :N] slice; per-lane permanent fallback) + envs.py
# (ARC_SKINNY_M doc gains "jm"). platform.py unchanged from cert22.
ARG BASE=vllm-arc:v0.30.0-b70-cert22
FROM ${BASE}
ARG BUILT=unknown
ARG WHEEL_SHA256=unknown
COPY vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl /tmp/vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl
COPY skinny_m.py /tmp/skinny_m.py
COPY envs.py /tmp/envs.py
RUN set -euxo pipefail; \
    echo "${WHEEL_SHA256}  /tmp/vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl" | sha256sum -c -; \
    /opt/venv/bin/pip install --force-reinstall --no-deps /tmp/vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl; \
    cp /tmp/skinny_m.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/skinny_m.py; \
    cp /tmp/envs.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/__pycache__/envs.cpython-312.pyc; \
    rm -f /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/__pycache__/skinny_m.cpython-312.pyc; \
    rm -f /tmp/vllm_xpu_kernels-0.1.14.1+arc9-cp312-cp312-linux_x86_64.whl /tmp/skinny_m.py /tmp/envs.py; \
    export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib; \
    /opt/venv/bin/python -c \
      "import importlib.metadata as md; \
       v = md.version('vllm-xpu-kernels'); \
       assert v == '0.1.14.1+arc9', 'vxk version ' + v; \
       import torch; \
       import vllm_xpu_kernels._xpu_C; \
       assert hasattr(torch.ops._xpu_C, 'jm_fp8_gemm_w8a16'), 'jm w16 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'jm_fp8_gemm_w8a16_mxfp8'), 'jm m816 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'jm_fp4_gemm_w4a16'), 'jm g16 op absent'; \
       assert hasattr(torch.ops._xpu_C, 'skinny_fp8_gemm_w8a16'), 'arc6 w16 op not carried'; \
       assert hasattr(torch.ops._xpu_C, 'skinny_fp4_gemm_w4a16'), 'arc6 g16 op not carried'; \
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a8'), 'w4a8 op not carried'; \
       m = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/skinny_m.py', encoding='utf-8').read(); \
       assert 'ARC_SKINNY_M' in m and 'jm_fp8_gemm_w8a16' in m and '_jm_pack' in m, 'mode gate / jm route'; \
       e = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/envs.py', encoding='utf-8').read(); \
       assert 'jm' in e.split('ARC_SKINNY_M')[-1][:500], 'envs jm doc'; \
       print('ARC9TEST_LIVE wheel+patches OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "arc9test=P6.34 test layer: vxk 0.1.14.1+arc9 (jm_* trio DUAL variant, ARC_JM_VAR 14=SLM-V24 A-direct+B-SLM exact form / 5=V5 byte-unchanged control; r128k4-k7 laws) + patches/skinny_m.py jm mode + envs doc"; \
      echo "arc9test_built=${BUILT}"; \
      echo "arc9test_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p634; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p634 /etc/vllm-arc/IMAGE_MANIFEST; \
    tail -4 /etc/vllm-arc/IMAGE_MANIFEST
