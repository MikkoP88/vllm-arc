#!/bin/bash
# r27 profile leg — product posture VERBATIM (r18/p23/r26 line: mode=none +
# FULL_DECODE_ONLY capture 1..320, MTP k4, graphs, prefix caching) + p10
# fastpath ONLY (no r25 timer, no p33 probe — TIMERS OFF, attribution leg)
# + torch profiler (r19 pattern, with_stack=false per aggregation-hang law).
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
export PYTHONPATH=/root/r27cap
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
  --compilation-config '{"mode":"none","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/prof","torch_profiler_with_stack":false}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
