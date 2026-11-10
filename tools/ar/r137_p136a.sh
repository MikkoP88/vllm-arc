#!/bin/bash
# r137_p136a.sh -- P6.36 leg A: prefill-window decomposition on cert23.
# Metrics proved the U8k TTFT lives INSIDE the engine prefill window
# (request_prefill_time 1.5-5 s vs ~1.0 s device EXEC; queue sub-ms), so
# this leg brackets the window per rank with epoch ts (ARR/SCHE/ESTEP/
# WEXE/REXE) to name the chunk-boundary host cost and the rank skew.
#   L1 fp8 k3 (certified posture): U8k 600 + U32k 300 faces
#   L2 mx4-W4A16 k5 (winner posture, TTFT tax): U8k 600 + warmup TTFT
# Probe image = cert23 + patch_p136a.py wrappers (never ships).
# Hardened teardown (r136b law: rm retried to 90 s). W14 trap.
set -uo pipefail
R=/root/build
CT=$R/p636
SP=/opt/venv/lib/python3.12/site-packages
IMG=vllm-arc:v0.30.0-b70-cert23
PROBE=vllm-arc:v0.30.0-b70-cert23-p136probe
NAME=arc-p136
mkdir -p "$CT"

w14_restore() {
  docker rm -f "$NAME" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p636.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$NAME"$ -q)" ] && return 0
    docker rm -f "$NAME" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "P136_FAIL_STALE_CONTAINER $NAME"; return 1
}

teardown() {
  docker exec "$NAME" bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f "$NAME" >/dev/null 2>&1
  verify_removed "$NAME"
}

gcount() {
  local c
  c=$(docker exec "$NAME" grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect "$NAME" >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[p136:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec "$NAME" pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] 2>/dev/null && { echo "[p136:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[p136:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

run_leg() {
  # $1=legname $2=served-model $3..=env
  local LEGN=$1; SERVED=$2; shift 2
  local CD=$CT/$LEGN
  mkdir -p "$CD"
  echo "=== [$LEGN] boot $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$PROBE" ARC_NAME="$NAME" "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEGN"; then
    echo "P136_BOOT_FAIL_$LEGN"
    docker cp "$NAME":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec "$NAME" tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f "$NAME" >/dev/null 2>&1; verify_removed "$NAME" || true
    return 1
  fi
  echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  local L; L=$(docker exec "$NAME" sh -c 'wc -l < /root/p136_prefill.log 2>/dev/null' || echo 0)
  echo "telemetry lines after warmup: ${L:-0}"
  if [ "${L:-0}" -lt 5 ]; then echo "P136_FAIL_NO_TELEMETRY_$LEGN"; return 1; fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics_warm.txt" || true

  docker cp $R/p635/p135_face.py "$NAME":/root/p135_face.py
  # U8k face
  docker exec "$NAME" sh -c ': > /root/p136_prefill.log'
  docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 8192 600 "$LEGN"
  local i=0
  while [ $i -lt 250 ]; do docker exec "$NAME" test -f /root/p135_face_$LEGN.json && break; sleep 2; i=$((i+1)); done
  echo "FACE_U8K_DONE_$LEGN waited=$((i*2))s"
  docker cp "$NAME":/root/p136_prefill.log "$CD/prefill_u8k.log" >/dev/null 2>&1
  docker cp "$NAME":/root/p135_face_$LEGN.json "$CD/" >/dev/null 2>&1
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics_u8k.txt" || true
  rm -f "$CD/face_u8k.json"; cp "$CD/p135_face_$LEGN.json" "$CD/face_u8k.json" 2>/dev/null || true

  if [ "$LEGN" = "l1fp8" ]; then
    # U32k face (chunkier prefill) on the same boot; distinct tag so the
    # container JSON cannot be confused with the U8k pull (r137 defect)
    docker exec "$NAME" sh -c "rm -f /root/p135_face_${LEGN}32.json; : > /root/p136_prefill.log"
    docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py 32768 300 "${LEGN}32"
    i=0
    while [ $i -lt 250 ]; do docker exec "$NAME" test -f /root/p135_face_${LEGN}32.json && break; sleep 2; i=$((i+1)); done
    echo "FACE_U32K_DONE_$LEGN waited=$((i*2))s"
    docker cp "$NAME":/root/p136_prefill.log "$CD/prefill_u32k.log" >/dev/null 2>&1
    docker cp "$NAME":/root/p135_face_${LEGN}32.json "$CD/" >/dev/null 2>&1
    cp "$CD/p135_face_${LEGN}32.json" "$CD/face_u32k.json" 2>/dev/null || true
    curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics_u32k.txt" || true
  fi

  docker cp "$NAME":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  teardown || return 1
  sleep 3
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

# ---- 1. probe image build (cert23 + p136 wrappers) ----
docker stop radiance-test >/dev/null 2>&1
sleep 3
docker rm -f p136x >/dev/null 2>&1
mkdir -p $CT/probe_src $CT/probe_out
docker run -d --name p136x --entrypoint sleep "$IMG" 120 >/dev/null
declare -A REL=(
  [core.py]=vllm/v1/engine/core.py
  [scheduler.py]=vllm/v1/core/sched/scheduler.py
  [model_runner.py]=vllm/v1/worker/gpu/model_runner.py
  [gpu_worker.py]=vllm/v1/worker/gpu_worker.py
)
for f in "${!REL[@]}"; do
  docker cp "p136x:$SP/${REL[$f]}" "$CT/probe_src/$f" || { echo "EXTRACT_FAIL $f"; exit 1; }
done
python3 $CT/patch_p136a.py > $CT/patch.txt 2>&1 || { cat $CT/patch.txt; echo GATE_FAIL_PATCH; exit 1; }
cat $CT/patch.txt
for f in "${!REL[@]}"; do
  docker cp "$CT/probe_out/$f" "p136x:$SP/${REL[$f]}" || { echo CP_FAIL $f; exit 1; }
done
docker exec p136x find $SP/vllm/v1 -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
SYN=0
for f in "${!REL[@]}"; do
  docker exec p136x /opt/venv/bin/python -c "import ast; ast.parse(open('$SP/${REL[$f]}').read())" || { echo "GATE_FAIL_SYNTAX $f"; SYN=1; }
done
[ $SYN -eq 0 ] || exit 1
echo SYNTAX_OK
docker commit p136x "$PROBE" >/dev/null
docker rm -f p136x >/dev/null
echo "PROBE_IMAGE_OK $PROBE"

# ---- 2. legs ----
run_leg l1fp8 qwen3.8-27b-fp8 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 || exit 1

run_leg l2mx4 qwen3.8-27b \
  ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
  ARC_QUANT=mxfp4 'ARC_QUANTCFG={"ignore":["*visual*","*linear_attn*","mtp.fc"]}' \
  ARC_MXFP4_MODE=w4a16 ARC_GMU=0.85 ARC_GRAPH=1 ARC_SPEC=5 ARC_LMC=1 \
  ARC_FP8=0 || exit 1

echo "P636_LEG_A_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- timeline analysis next"
