#!/bin/bash
# p635_gate2b.sh -- P6.35 leg F: full draft-path/replay/offloader attribution.
# Probe image p135fprobe from patch_p135f (13 files). Boot 1: certified
# posture (GRAPH=1 SPEC=3 LMC=1); boot 2: ARC_SPEC=0. Faces: U8k solo 600.
# Fixes over gate2: verify-removed container teardown between boots (gate2's
# silent rm failure left a stale stopped container that the serve script's
# freshness logic reused -> "is not running"), race-free PROF_ON arming.
set -uo pipefail
R=/root/build
CT=$R/p635
SP=/opt/venv/lib/python3.12/site-packages
IMG=vllm-arc:v0.30.0-b70-cert22
PROBE=vllm-arc:v0.30.0-b70-cert22-p135fprobe
NAME=arc-telf
LEG=$CT/legF-$(date +%Y%m%d-%H%M%S)
mkdir -p "$LEG"

w14_restore() {
  docker stop "$NAME" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {  # $1 = container name; fails the gate if it lingers
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2
    i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $1"
  docker ps -a --filter name=^/"$1"$ --format '{{.Names}} {{.Status}}'
  return 1
}

docker stop radiance-test >/dev/null 2>&1
docker rm -f "$NAME" arc-tel2 >/dev/null 2>&1
verify_removed "$NAME" || exit 1
verify_removed arc-tel2 || exit 1
sleep 2

# ---- 1. extract 13 sources ----
rm -rf $CT/probe_src $CT/probe_out
mkdir -p $CT/probe_src $CT/probe_out
docker run -d --name p135x --entrypoint sleep "$IMG" 180 >/dev/null
declare -A REL=(
  [core.py]=vllm/v1/engine/core.py
  [scheduler.py]=vllm/v1/core/sched/scheduler.py
  [model_runner.py]=vllm/v1/worker/gpu/model_runner.py
  [mamba_hybrid.py]=vllm/v1/worker/gpu/model_states/mamba_hybrid.py
  [cudagraph_utils.py]=vllm/v1/worker/gpu/cudagraph_utils.py
  [gpu_worker.py]=vllm/v1/worker/gpu_worker.py
  [llm_base_proposer.py]=vllm/v1/spec_decode/llm_base_proposer.py
  [speculator.py]=vllm/v1/worker/gpu/spec_decode/speculator.py
  [sd_utils.py]=vllm/v1/worker/gpu/spec_decode/utils.py
  [ar_speculator.py]=vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py
  [off_base.py]=vllm/model_executor/offloader/base.py
  [off_prefetch.py]=vllm/model_executor/offloader/prefetch.py
  [batch_shard.py]=vllm/v1/worker/gpu/sample/batch_shard.py
)
for f in "${!REL[@]}"; do
  docker cp "p135x:$SP/${REL[$f]}" "$CT/probe_src/$f" || { echo "EXTRACT_FAIL $f"; exit 1; }
done
docker rm -f p135x >/dev/null

# ---- 2. patch + image ----
md5sum $CT/patch_p135f.py
python3 $CT/patch_p135f.py > "$LEG/patch.txt" 2>&1 || { cat "$LEG/patch.txt"; echo GATE_FAIL_PATCH; exit 1; }
cat "$LEG/patch.txt"
docker run -d --name p135x --entrypoint sleep "$IMG" 180 >/dev/null
for f in "${!REL[@]}"; do
  docker cp "$CT/probe_out/$f" "p135x:$SP/${REL[$f]}" || { echo CP_FAIL $f";" exit 1; }
done
docker exec p135x find $SP/vllm -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
SYN=0
for f in "${!REL[@]}"; do
  docker exec p135x /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/${REL[$f]}').read())" || { echo "GATE_FAIL_SYNTAX $f"; SYN=1; }
done
[ $SYN -eq 0 ] || exit 1
echo SYNTAX_OK
docker commit p135x "$PROBE" >/dev/null
docker rm -f p135x >/dev/null
echo "PROBE_IMAGE_OK $PROBE"

boot_and_face() {  # $1=spec $2=tag
  local SPEC=$1 TAG=$2
  nohup env ARC_IMAGE="$PROBE" ARC_NAME=$NAME ARC_GRAPH=1 ARC_SPEC=$SPEC ARC_LMC=1 \
    bash $R/serve_arc_b70.sh > "$LEG/boot_$TAG.txt" 2>&1 &
  local i=0
  while [ $i -lt 150 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { echo "HEALTH_OK_$TAG after ${i}0s"; break; }
    local d
    d=$(docker logs "$NAME" 2>&1 | grep -cE "EngineCore.*(died|exception)|RuntimeError|DEVICE_LOST" || true)
    [ "${d:-0}" -gt 0 ] && { docker logs "$NAME" 2>&1 | tail -40; echo "GATE_FAIL_ENGINE_DEAD_$TAG"; return 1; }
    i=$((i + 1)); sleep 10
  done
  [ $i -lt 150 ] || { echo "GATE_FAIL_HEALTH_TIMEOUT_$TAG"; docker logs "$NAME" 2>&1 | tail -40; return 1; }
  sleep 5
  docker exec "$NAME" sh -c 'rm -f /root/p135_steps.log /root/PROF_ON /root/p135_face_*.json /root/p135_prof_rank*.json'
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  local L
  L=$(docker exec "$NAME" sh -c 'wc -l < /root/p135_steps.log 2>/dev/null' || echo 0)
  [ "${L:-0}" -ge 5 ] || { echo "GATE_FAIL_NO_TELEMETRY_$TAG lines=$L"; return 1; }
  echo "TELEMETRY_ALIVE_$TAG lines=$L"
  docker cp $CT/p135_face.py "$NAME":/root/p135_face.py
  docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 600 "$TAG"
  sleep 8
  docker exec "$NAME" sh -c 'echo 24 > /root/PROF_ON'
  local i2=0
  while [ $i2 -lt 200 ]; do docker exec "$NAME" test -f /root/p135_face_$TAG.json && break; sleep 2; i2=$((i2+1)); done
  echo "FACE_${TAG}_DONE waited=$((i2*2))s"
  sleep 3
  local PD
  PD=$(docker exec "$NAME" sh -c 'grep -c PROF_DONE /root/p135_steps.log' || echo 0)
  echo "prof_done_count=${PD:-0}"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$LEG/metrics_$TAG.txt" || true
  docker cp "$NAME":/root/p135_steps.log "$LEG/p135_steps_$TAG.log" >/dev/null 2>&1 || echo "steps MISSING $TAG"
  for r in 0 1; do
    docker exec "$NAME" test -f /root/p135_prof_rank$r.json && docker cp "$NAME":/root/p135_prof_rank$r.json "$LEG/p135_prof_${TAG}_rank$r.json" >/dev/null 2>&1
  done
  docker cp "$NAME":/root/p135_face_$TAG.json "$LEG/" >/dev/null 2>&1 || echo "face missing $TAG"
  return 0
}

boot_and_face 3 spec3 || exit 1
docker stop "$NAME" >/dev/null 2>&1
docker rm -f "$NAME"
verify_removed "$NAME" || exit 1
sleep 5
boot_and_face 0 spec0 || exit 1

# ---- tokenizer bench (live spec0 boot) ----
docker cp $CT/p135_tok.py "$NAME":/root/p135_tok.py
docker exec "$NAME" /opt/venv/bin/python /root/p135_tok.py > "$LEG/tok.txt" 2>&1 || echo "TOK_BENCH_FAIL"
cat "$LEG/tok.txt"
docker cp "$NAME":/root/p135_tok.json "$LEG/" >/dev/null 2>&1 || true

cd "$LEG" && sha256sum p135_steps_*.log p135_prof_*.json p135_face_*.json p135_tok.json > SHA256SUMS 2>/dev/null
ls -la "$LEG"
echo "LEG_ARTIFACTS $LEG"
echo "P635_GATE2B_CAPTURE_COMPLETE -- analysis next"
