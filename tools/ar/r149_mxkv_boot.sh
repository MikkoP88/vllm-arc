#!/bin/bash
# r149_mxkv_boot.sh -- P6.47 leg B2: first MX KV engine boot (serve) on
# the dual-B70 rig. Boots qwen3.8-27b-fp8 with --kv-cache-dtype
# $FMT through the mounted /work tree (PYTHONPATH shadow of the
# installed vllm_gaudi), champion env chain minus LMCache (the MX lane
# is isolated first; kv-transfer interop is a later leg), then probes
# models + a 3-prompt generation battery and greps the serve log for
# the MX routing/impl lines.
# Knobs: FMT (mxfp8|mxfp4|fp8), SPEC (0|3), GRAPH (0|1), MAXLEN, TRACE (0|1).
# W14 quiesce before, trap restore after (rig law).
# Artifacts -> /root/build/p149/{out,logs}. Exit 0 = boot + probes PASS.
set -uo pipefail
R=/root/build
P=$R/p149
W=$R/p148/work
IMG=vllm-arc:v0.30.0-b70-cert23
FMT=${FMT:-mxfp8}
SPEC=${SPEC:-0}
GRAPH=${GRAPH:-0}
MAXLEN=${MAXLEN:-32768}
MODEL=${MODEL:-/models/qwen3.8-27b-fp8}
TRACE=${TRACE:-0}
PC=${PC:-1}
ALIGN=${ALIGN:-1}
# DUMP=1: after the battery, touch the ARC_MXKV_DUMP sentinel and fire a
# 1-token request — the engine saves its low pool region to
# $P_OUT/mxkv_pool.pt (the spec-vs-eager ground-truth byte differential).
DUMP=${DUMP:-0}
# Per-run bank dir (trace9/10 evidence was overwritten by the next boot's
# fixed out/ paths — banking law: every boot banks to its OWN dir).
TAG=${TAG:-out}
P_OUT=$P/$TAG
CNAME=arc-p149

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p149.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P_OUT

echo "=== [r149:legB2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5
docker rm -f $CNAME >/dev/null 2>&1 || true

GRAPH_ENV=""
COMPILATION_FLAG=""
if [ "$GRAPH" = 1 ]; then
  GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"
  COMPILATION_FLAG="--compilation-config '{\"mode\":\"none\",\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,30,35,40,45,48,50,55,60,64,65,70,75,80,85,90,95,96,100,105,110,112,115,120,125,128,130,135,140,145,150,155,160,165,170,175,180,185,190,192,195,200,205,210,215,220,224,225,230,235,240,245,250,255,256,260,265,270,275,280,285,290,295,300,305,310,315,320]}'"
fi
TRACE_ENV=""
if [ "$TRACE" = 1 ]; then
  TRACE_ENV="export ARC_MXKV_TRACE=1;"
fi
DUMP_ENV=""
if [ "$DUMP" = 1 ]; then
  # Layer pin + p1 shape filter: only the first attention impl consumes
  # sentinels, and only for battery-prompt-1-shaped forwards (warmup/
  # other-request forwards eat unfiltered sentinels — r152 trace13).
  DUMP_ENV="export ARC_MXKV_DUMP=1 ARC_MXKV_DUMP_ORD=0 ARC_MXKV_DUMP_FILTER=p1;"
fi
SPEC_FLAG=""
if [ "$SPEC" != 0 ]; then
  SPEC_FLAG="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":$SPEC,\"draft_sample_method\":\"probabilistic\"}'"
fi
PC_FLAG="--enable-prefix-caching"
if [ "$PC" = 0 ]; then
  PC_FLAG="--no-enable-prefix-caching"
fi
ALIGN_FLAG="--mamba-cache-mode=align"
if [ "$ALIGN" = 0 ]; then
  ALIGN_FLAG=""
fi

echo "=== [r149:legB2] boot $FMT spec=$SPEC graph=$GRAPH maxlen=$MAXLEN $(date +%H:%M:%S) ==="
docker run -d --name $CNAME \
  --privileged --network host --ipc host \
  --shm-size 16g \
  -v "$MODEL":/models/target:ro \
  -v "$W":/work \
  -v /root/build/triton_cache:/root/.triton_cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTHONPATH=/work \
  --entrypoint /bin/bash "$IMG" -c 'sleep infinity' \
  > $P/logs/boot.out 2>&1

docker exec -d $CNAME bash -c "
export PYTHONHASHSEED=0
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=300
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
$GRAPH_ENV
$TRACE_ENV
$DUMP_ENV
exec /opt/venv/bin/vllm serve --model /models/target \
  --served-model-name mx-boot \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --max-model-len $MAXLEN \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 64 \
  --block-size 64 \
  --dtype float16 \
  --quantization fp8 \
  --kv-cache-dtype $FMT \
  --mamba-ssm-cache-dtype float16 \
  $ALIGN_FLAG \
  $PC_FLAG \
  --trust-remote-code \
  --no-disable-hybrid-kv-cache-manager \
  $COMPILATION_FLAG \
  $SPEC_FLAG \
  --port 8000 \
  > /root/serve_mx.log 2>&1"

# readiness poll (fresh boot incl. weight load + capture: allow 25 min)
READY=0
for i in $(seq 1 1500); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then READY=1; break; fi
  # fail fast on a dead engine
  if ! docker exec $CNAME pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "ENGINE_PROC_GONE at poll $i"; break
  fi
  sleep 1
done
echo "READY=$READY after ~${i}s $(date +%H:%M:%S)"
if [ "$READY" != 1 ]; then
  docker cp $CNAME:/root/serve_mx.log $P_OUT/serve_fail_full.log >/dev/null 2>&1 || true
  grep -B4 -A25 "Traceback" $P_OUT/serve_fail_full.log | tail -80 || true
  echo "R149_FAIL_NO_READY"; exit 1
fi

echo "=== [r149:legB2] generation battery $(date +%H:%M:%S) ==="
if [ "$DUMP" = 1 ]; then
  # Dump mode: prompt 1 runs in the background and the sentinel is
  # touched MID-GENERATION — the next verify/decode forward banks with
  # bt = prompt 1's OWN block table (a post-hoc trigger re-allocates
  # fresh blocks and banks the wrong request; r151 pass 2).
  # REQ_JSON (rig path to a full /v1/completions body): overrides the
  # hardcoded p1 prompt (trace22 same-tokens store check banks a
  # custom P+suffix prefill; the body ships verbatim — no shell
  # quoting of model-generated text).
  if [ -n "${REQ_JSON:-}" ]; then
    docker cp "$REQ_JSON" $CNAME:/tmp/p1_req.json >/dev/null
    P1_CURL='curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" --data @/tmp/p1_req.json'
  else
    P1_CURL='curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d "{\"model\":\"mx-boot\",\"prompt\":\"What is the capital of France? Answer with just the city name.\",\"max_tokens\":64,\"temperature\":0}"'
  fi
  docker exec $CNAME sh -c "rm -f /tmp/mxkv_dump /tmp/mxkv_dump.done /tmp/mxkv_dump.*.pt /tmp/p1_resp.json /tmp/p1_req.json"
  [ -n "${REQ_JSON:-}" ] && docker cp "$REQ_JSON" $CNAME:/tmp/p1_req.json >/dev/null
  # Pre-place one sentinel BEFORE p1: its prefill (guaranteed p1-shaped,
  # never a dummy) consumes it. Then a dense touch fan — the async
  # scheduler interleaves idle-dummy forwards that eat sentinels (all 4
  # eager payloads were dummy-shaped in r151 pass 4); 20x0.2s over the
  # whole run guarantees several mid-p1 captures per side.
  docker exec $CNAME touch /tmp/mxkv_dump
  docker exec -d $CNAME bash -c "
$P1_CURL \
  > /tmp/p1_resp.json 2>&1"
  # Touch fan: keep a sentinel pending UNTIL p1 COMPLETES. The async
  # scheduler interleaves the startup-warmup request's spec steps with
  # p1's (r153 trace17 diag: warmup verifies at sl0=99/8199 legally
  # filter and hold the sentinel); a fixed 20x0.2s fan exhausts during
  # the warmup tail and banks ONLY the prefill (1 payload/boot across
  # trace14-17). Cap 300 taps = 60 s.
  T=0
  until docker exec $CNAME sh -c "test -s /tmp/p1_resp.json" || [ $T -ge 300 ]; do
    sleep 0.2
    T=$((T+1))
    docker exec $CNAME touch /tmp/mxkv_dump
  done
  docker exec $CNAME bash -c '
for i in $(seq 1 120); do [ -s /tmp/p1_resp.json ] && break; sleep 0.5; done
/opt/venv/bin/python -c "import json; r=json.load(open(\"/tmp/p1_resp.json\")); print(\"PROMPT_OK\", len(r[\"choices\"][0][\"text\"]), repr(r[\"choices\"][0][\"text\"][:120]))" 2>&1
for q in "Write a Python function that returns the n-th Fibonacci number using iteration." \
         "Name three chemical elements that are gases at room temperature."; do
  curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"mx-boot\",\"prompt\":\"$q\",\"max_tokens\":64,\"temperature\":0}" \
    | /opt/venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print(\"PROMPT_OK\", len(r[\"choices\"][0][\"text\"]), repr(r[\"choices\"][0][\"text\"][:120]))" 2>&1
done' > $P_OUT/probe.out 2>&1
else
  docker exec $CNAME bash -c '
for q in "What is the capital of France? Answer with just the city name." \
         "Write a Python function that returns the n-th Fibonacci number using iteration." \
         "Name three chemical elements that are gases at room temperature."; do
  curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
    -d "{\"model\":\"mx-boot\",\"prompt\":\"$q\",\"max_tokens\":64,\"temperature\":0}" \
    | /opt/venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print(\"PROMPT_OK\", len(r[\"choices\"][0][\"text\"]), repr(r[\"choices\"][0][\"text\"][:120]))" 2>&1
done' > $P_OUT/probe.out 2>&1
fi
cat $P_OUT/probe.out

docker exec $CNAME grep -E "Using MX KV attention backend|MXKV impl|Traceback|ERROR|Setting attention block size|GPU KV cache size|Maximum concurrency" /root/serve_mx.log \
  > $P_OUT/serve_grep.txt 2>&1 || true
echo "--- serve log greps ---"
head -25 $P_OUT/serve_grep.txt || true
docker exec $CNAME tail -5 /root/serve_mx.log > $P_OUT/serve_tail.txt 2>&1 || true
# P619I forward-instrument lines (verify/prefill shapes + qlens) and the
# FULL serve log always bank on PASS too (telemetry law — the spec-path
# corruption hunt needs the real call shapes, 2026-11-11).
docker exec $CNAME grep -c "P619I" /root/serve_mx.log > $P_OUT/p619i_count.txt 2>&1 || true
docker exec $CNAME grep "P619I" /root/serve_mx.log | head -400 > $P_OUT/p619i_head.txt 2>&1 || true
docker exec $CNAME grep "MXKVTRACE" /root/serve_mx.log | head -50000 > $P_OUT/mxkv_trace.txt 2>&1 || true
# Full arbiter population (unlimited; xcheck3 only): the corruption hunt
# compares store smap vs walk slot ids across the WHOLE battery.
docker exec $CNAME grep "xcheck3" /root/serve_mx.log > $P_OUT/xcheck3_full.txt 2>&1 || true
docker cp $CNAME:/root/serve_mx.log $P_OUT/serve_full.log >/dev/null 2>&1 || true

# Pool ground-truth dump fetch (DUMP=1): bank ALL collected payloads
# (tar) + the post-battery serve tail (self-diagnosing).
if [ "$DUMP" = 1 ]; then
  docker exec $CNAME sh -c "cd /tmp && ls mxkv_dump.*.pt >/dev/null 2>&1 && tar -cf dumps.tar mxkv_dump.*.pt"
  if docker exec $CNAME sh -c "test -f /tmp/dumps.tar"; then
    mkdir -p $P_OUT/dumps
    docker cp $CNAME:/tmp/dumps.tar $P_OUT/dumps/ >/dev/null 2>&1 || true
    tar -xf $P_OUT/dumps/dumps.tar -C $P_OUT/dumps 2>/dev/null || true
    rm -f $P_OUT/dumps/dumps.tar
    echo "POOL_DUMP $(ls $P_OUT/dumps/ 2>/dev/null | wc -l) payloads -> $P_OUT/dumps/"
  else
    echo "POOL_DUMP_MISSING (no payloads mid-p1)"
  fi
  docker exec $CNAME sh -c "ls -la /tmp/mxkv_dump* 2>&1 | head -30" > $P_OUT/sentinel_state.txt 2>&1 || true
  docker exec $CNAME tail -40 /root/serve_mx.log > $P_OUT/serve_post_dump.log 2>&1 || true
fi

if grep -q "PROMPT_OK" $P_OUT/probe.out && [ "$(grep -c PROMPT_OK $P_OUT/probe.out)" = 3 ]; then
  echo "R149_BOOT_PASS fmt=$FMT spec=$SPEC graph=$GRAPH maxlen=$MAXLEN $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cd $P_OUT && sha256sum probe.out serve_grep.txt serve_tail.txt serve_full.log > SHA256SUMS || true
  exit 0
fi
echo "R149_FAIL_PROBES"
exit 1
