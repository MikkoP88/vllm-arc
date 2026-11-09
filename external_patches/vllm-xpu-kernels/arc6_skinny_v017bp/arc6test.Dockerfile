# arc6test.Dockerfile — vllm-arc:v0.30.0-b70-arc6test = cert21 + the P6.33
# TEST layer (the arc5test precedent: validated here first; the certified
# cert22 image is rebuilt FROM cert21 via a tracked tools/cert22.Dockerfile
# at close-out with tag-pinned FORK_SHA — this file stays rig-side).
# Changes EXACTLY: the vllm-xpu-kernels wheel (arc5->arc6; same builder
# toolchain oneAPI 2026.0.0, .deps kept — adds the skinny-M op trio
# _xpu_C.skinny_{fp8_gemm_w8a16,fp8_gemm_w8a16_mxfp8,fp4_gemm_w4a16} +
# W4A8 dispatch arms for the arc5 enum values + backports #611/#645/#650/
# #600), adds vllm_gaudi/xpu/patches/skinny_m.py (ARC_SKINNY_M M<=8
# route), updates platform.py (_XPU_PATCH_MODULES gains skinny_m) +
# envs.py (ARC_SKINNY_M registry entry).
# r123 kernel matrix basis (banked /root/build/p631/kbench/): decode
# M<=8 medians w16pt 0.179 / w16pc 0.218 / m816 0.327 / g16 0.310 ms vs
# bytes floor 0.058/0.029 — the skinny kernel family attacks that gap;
# W4A8 dispatch answer (arc6 arms) is banked by r126 GATE-3.
ARG BASE=vllm-arc:v0.30.0-b70-cert21
FROM ${BASE}
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
       assert hasattr(torch.ops._xpu_C, 'fp4_gemm_w4a8'), 'w4a8 op not carried'; \
       p = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/platform.py', encoding='utf-8').read(); \
       assert p.count('skinny_m') >= 1, 'platform registration'; \
       m = open('/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/skinny_m.py', encoding='utf-8').read(); \
       assert 'ARC_SKINNY_M' in m and 'skinny_fp8_gemm_w8a16' in m, 'mode gate / route target'; \
       print('ARC6TEST_LIVE wheel+patches+registration OK')"; \
    { cat /etc/vllm-arc/IMAGE_MANIFEST; \
      echo "arc6test=P6.33 test layer: vxk 0.1.14.1+arc6 (skinny-M op trio + W4A8 arms + backports 611/645/650/600) + patches/skinny_m.py (ARC_SKINNY_M, M<=8 route) + platform/envs entries; r123 basis: decode medians 3-10x bytes floor"; \
      echo "arc6test_built=${BUILT}"; \
      echo "arc6test_vxk_wheel_sha256=${WHEEL_SHA256}"; \
    } > /etc/vllm-arc/IMAGE_MANIFEST.p633; \
    mv /etc/vllm-arc/IMAGE_MANIFEST.p633 /etc/vllm-arc/IMAGE_MANIFEST; \
    tail -4 /etc/vllm-arc/IMAGE_MANIFEST
