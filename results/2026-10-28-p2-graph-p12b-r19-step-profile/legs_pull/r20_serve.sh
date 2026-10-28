#!/bin/bash
# r20 — eager-decode profile leg: r19 posture with GRAPHS OFF (no
# VLLM_XPU_ENABLE_XPU_GRAPH, cudagraph NONE) so every decode kernel is
# profiler-visible. Diagnostic leg only (eager is not a serving posture —
# the eager-tax law stands); purpose = per-kernel attribution of the c8
# decode step (r18: TPOT 113-185ms, ~10x the D champion step).
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
export ARC_TRACE_FLIP=1
mkdir -p /root/prof2
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
  --compilation-config '{"mode":"none","cudagraph_mode":"NONE"}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/prof2","torch_profiler_with_stack":false}' \
  --port 8000 \
  > /root/serve_full.log 2>&1
