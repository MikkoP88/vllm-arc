# arc5test.Dockerfile — vllm-arc:v0.30.0-b70-arc5test = cert20 + the P6.31
# TEST layer (the arc4test precedent: validated here first; the certified
# cert21 image is rebuilt FROM cert20 via the tracked tools/cert21.Dockerfile
# at close-out with tag-pinned FORK_SHA — this file stays rig-side).
# Changes EXACTLY: the vllm-xpu-kernels wheel (arc4->arc5; same builder
# toolchain oneAPI 2026.0.0, .deps kept — adds _xpu_C.fp8_gemm_w8a16_mxfp8
# + _xpu_C.fp4_gemm_w4a8, carries fp4_gemm_w4a16), adds
# vllm_gaudi/xpu/patches/mxfp8_w8a16.py (ARC_MXFP8_MODE reroute), updates
# patches/mxfp4_w4a16.py (w4a8 probe branch), updates platform.py
# (_XPU_PATCH_MODULES gains mxfp8_w8a16).
# r123 kernel matrix (banked /root/build/p631/kbench/): m816 works
# (rel 2e-3, decode 1.74x / prefill 2.68x faster than stock online-mxfp8)
# but is 1.50x/1.06x SLOWER than the w16pc champion class; w4a8 joint
# (e4m3,f4) REJECTED at oneDNN primitive-create ("Only support int4 and
# fp8 gemm") — clean closed-negative. This image exists for the e2e A/B
# battery that banks the model-level numbers.
ARG BASE=vllm-arc:v0.30.0-b70-cert20
FROM ${BASE}
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
       print('ARC5TEST_LIVE wheel+patches+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "arc5test=P6.31 test layer: vxk 0.1.14.1+arc5 (fp8_gemm_w8a16_mxfp8 + fp4_gemm_w4a8) + patches/mxfp8_w8a16.py (ARC_MXFP8_MODE) + mxfp4_w4a16.py w4a8 probe + platform entry; r123 kernel matrix: m816 green-but-slower-than-w16pc, w4a8 oneDNN-refused"; \
      echo "arc5test_built=${BUILT}"; \
      echo "arc5test_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p631; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p631 /etc/vllm-arc/IMAGE_MANIFEST; \
    tail -4 /etc/vllm-arc/IMAGE_MANIFEST
