#!/bin/bash
# serve_arc_b70.sh — uniform boot interface for vllm-arc on the dual-B70 rig
# (plan §1 harness law: ONE boot script per image; arc_matrix.sh drives legs
# through this interface).
#
# Posture knobs (environment; defaults = P1 gate posture, plan §10):
#   ARC_IMAGE   image tag            (default vllm-arc:v0.30.0-b70)
#   ARC_NAME    container name       (default arc-test)
#   ARC_GRAPH   0 eager | 1 VLLM_XPU_ENABLE_XPU_GRAPH=1 graph posture
#               (standing rig law: every graph leg runs with it set)
#   ARC_SPEC    0 MTP off | 1 MTP k4 — appends the D-champion spec CLI
#               (--speculative-config method mtp, num_speculative_tokens 4;
#               rig /root/serve_lmc_v133.sh:53 parity). Spec-config law:
#               CLI flag, not env (plan p30). P2-candidate posture — the
#               §10 P2 gate ("FULL graph + MTP boots and survives first
#               request") is proven by the P2 spec legs, not assumed.
#   ARC_LMC     0 no LMCache | 1 D-champion kv-transfer config (P3)
#   ARC_GMU     --gpu-memory-utilization (default 0.8, D-champion certified
#               posture — V1227_GMU lineage, plan Appendix F)
#   ARC_MNBT    --max-num-batched-tokens (default 8192, D champion)
#   ARC_MAXSEQS --max-num-seqs (default 64, D champion)
#   ARC_PORT    serve port (default 8000)
#   VLLM_ARC_XG_COMPACT=1  appends --structured-outputs-config
#               {"backend":"xgrammar","disable_any_whitespace":true}
#               (port of D's V1227_XGCOMPACT; core-0.30 native surface —
#               patch p06)
#
# Model flags are the D-champion surface for qwen3.8-27b-fp8 (rig
# /root/serve_lmc_v133.sh, plan Appendix F): TP2, block 64, fp8 quant +
# fp8_e4m3 KV, float16 mamba SSM cache, prefix caching, hybrid KV manager on.
set -euo pipefail

ARC_IMAGE=${ARC_IMAGE:-vllm-arc:v0.30.0-b70}
ARC_NAME=${ARC_NAME:-arc-test}
ARC_GRAPH=${ARC_GRAPH:-0}
ARC_SPEC=${ARC_SPEC:-0}
ARC_LMC=${ARC_LMC:-0}
ARC_GMU=${ARC_GMU:-0.8}
ARC_MNBT=${ARC_MNBT:-8192}
ARC_MAXSEQS=${ARC_MAXSEQS:-64}
ARC_PORT=${ARC_PORT:-8000}
RIG=/root/build

# ARC_SPEC=1 is a P2-candidate posture (p30): allowed by default now — its
# validation IS the §10 P2 spec-leg row. Only the still-future cache posture
# stays hard-refused.
if [ "$ARC_SPEC" = 1 ] && [ "$ARC_GRAPH" = 0 ]; then
  echo "NOTE: ARC_SPEC=1 with ARC_GRAPH=0 — non-champion combo (D v133 runs MTP"
  echo "with FULL_DECODE_ONLY graphs); allowed for spec-eager A/B isolation only."
fi
[ "$ARC_LMC" = 0 ] || { echo "REFUSE: ARC_LMC=1 before p40..p43 land (P3) — cache posture not yet validated on this image"; exit 3; }

# --- container: create if absent (W14 pattern: persistent container, serve
# via exec — fast re-boots between legs; same device/ipc/net posture as the
# validated radiance-test container). privileged + oneCCL env = the universal
# validated TP2 posture on this rig: both donor lines boot it identically
# (radiance-test W14 env == lsv-test iexperimental v1.0.0 env, byte-compared
# 2026-10-27; CCL_ZE_IPC_EXCHANGE=pidfd is what keeps oneCCL's ze_fd_manager
# off the opendir device walk that kills non-privileged TP2 init). ---
if ! docker inspect "$ARC_NAME" >/dev/null 2>&1; then
  docker run -d --name "$ARC_NAME" \
    --privileged \
    --device /dev/dri --ipc host --network host \
    --shm-size 64m \
    -v /root/build/neo_cache:/root/.neo_cache \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v /root/build/inductor_cache:/root/.inductor_cache \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v "$RIG/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro" \
    --entrypoint /bin/bash "$ARC_IMAGE" -c 'sleep infinity'
fi
docker exec "$ARC_NAME" pkill -f "vllm serve" 2>/dev/null || true
# Wait out the old engine: /health can answer for seconds after pkill, and a
# fresh leg's health probe would race it and "pass" against a corpse.
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$ARC_PORT/health" >/dev/null 2>&1 || break
  sleep 1
done
sleep 2

GRAPH_ENV=""
COMPILATION_FLAG=""
if [ "$ARC_GRAPH" = 1 ]; then
  # Standing rig law (plan Appendix F): VLLM_XPU_ENABLE_XPU_GRAPH=1 on all
  # graph legs. P1 posture = FULL_DECODE_ONLY pool identical to the
  # D-champion certified set (54/64-size discipline lands with p12/p32).
  GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"
  COMPILATION_FLAG="--compilation-config '{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}'"
fi

XG_FLAG=""
if [ "${VLLM_ARC_XG_COMPACT:-0}" = 1 ]; then
  XG_FLAG="--structured-outputs-config '{\"backend\":\"xgrammar\",\"disable_any_whitespace\":true}'"
fi

# MTP k4 (p30): D-champion spec config, CLI flag not env — rig
# /root/serve_lmc_v133.sh:53 parity. k=4 is the certified D value (plan
# Appendix F champion chain). The draft side resolves entirely inside
# upstream 0.30: qwen3_5 model_type -> Qwen3_5MTP arch, n_predict from
# text_config.mtp_num_hidden_layers (p30 README assessment) — no draft-model
# path, the head ships inside the target checkpoint (mtp.* weights).
SPEC_FLAG=""
if [ "$ARC_SPEC" = 1 ]; then
  SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":4}'"
fi

docker exec -d "$ARC_NAME" bash -c "
export PYTHONHASHSEED=0
export ZE_AFFINITY_MASK=0,1
export LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib
# oneCCL TP2 block — identical in W14 incumbent and D lsv-test (validated
# posture, cited above); provenance: rig container env inspect 2026-10-27.
export CCL_ENABLE_SYCL_KERNELS=1
export CCL_ZE_IPC_EXCHANGE=pidfd
export CCL_TOPO_P2P_ACCESS=1
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
export CCL_SYCL_ALLGATHERV_SMALL_THRESHOLD=131072
export CCL_SYCL_ALLGATHERV_SCALEOUT_THRESHOLD=1048576
export CCL_SYCL_ALLGATHERV_TMP_BUF=0
export CCL_SYCL_ALLREDUCE_TMP_BUF=0
$GRAPH_ENV
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization $ARC_GMU \
  --max-model-len 262144 \
  --max-num-batched-tokens $ARC_MNBT \
  --max-num-seqs $ARC_MAXSEQS \
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
  $COMPILATION_FLAG \
  $SPEC_FLAG \
  $XG_FLAG \
  --port $ARC_PORT \
  > /root/serve_full.log 2>&1"

echo "ARC_SERVE_LAUNCHED image=$ARC_IMAGE name=$ARC_NAME graph=$ARC_GRAPH spec=$ARC_SPEC gmu=$ARC_GMU mnbt=$ARC_MNBT seqs=$ARC_MAXSEQS xgc=${VLLM_ARC_XG_COMPACT:-0} port=$ARC_PORT $(date +%H:%M:%S)"
