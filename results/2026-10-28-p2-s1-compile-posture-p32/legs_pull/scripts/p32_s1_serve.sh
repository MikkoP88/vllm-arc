#!/bin/bash
# S1/p32 — compile posture leg: W14-transplanted compilation-config
# (FULL_AND_PIECEWISE, capture [4,8,16,32,64], max 64 — the ONLY mode where
# compiled pieces are CAPTURED; defect-b face structurally absent), everything
# else product-verbatim (MTP k4, prefix caching ON). No "mode" key ->
# CompilationMode.VLLM_COMPILE. p10 fastpath + r25 phase timer via
# PYTHONPATH=/root/p25cap (timer ON = gate (d) split; TPOT cells carry the
# +3.8% distortion caveat).
set -eo pipefail
export PYTHONHASHSEED=0
export ZE_AFFINITY_MASK=0,1
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export ARC_TRACE_FLIP=1
export PYTHONPATH=/root/p25cap
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --mamba-ssm-cache-dtype float16 \
  --enable-prefix-caching \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --chat-template /root/chat_template_qwen38_high.jinja \
  --no-disable-hybrid-kv-cache-manager \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[4,8,16,32,64],"max_cudagraph_capture_size":64}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
