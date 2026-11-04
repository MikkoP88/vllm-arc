#!/bin/bash
# p619d — run MQVAL5 (AoS-pair numeric gate) in the cert6 container.
set -e
D=/root/build/p618/p619d
SP=/opt/venv/lib/python3.12/site-packages
docker run --rm --privileged --network host --ipc host \
  -v "$D/tq_mq_aos.py:$SP/vllm_gaudi/xpu/patches/tq_mq_aos.py:ro" \
  -v "$D/p619d_mqval5.py:/root/mqval5.py:ro" \
  -v /root/build/triton_cache:/root/.triton_cache \
  --entrypoint python3 vllm-arc:v0.30.0-b70-cert6 /root/mqval5.py
