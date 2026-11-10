#!/bin/bash
# r137b_p136b.sh -- P6.36 leg B: prefill device-lever discriminators on the
# p136 probe image (built by r137_p136a.sh; must exist).
#
# Leg A verdict (results/2026-11-10-p636-legA/ANALYSIS.md): the prefill
# window is device drain -- 80% of the U32k window is engine-core
# future.result() waiting on chunk-forward device completion; ranks in
# lockstep (zero skew); scheduler/queue sub-ms. So the levers are
# device-side:
#   b1m16  fp8-k3 + ARC_MNBT=16384 -- fewer chunk boundaries; U8k becomes
#          a single-chunk prefill. Splits boundary cost vs device-proportional.
#   b2bf16 /models/qwen3.8-27b bf16, no quant, spec3, MAXLEN 32768 --
#          wide-math discriminator. Xe2 has no native FP8 dot (counter-
#          operand conversion); if bf16 chunk drain < fp8 drain, a wide-math
#          prefill lane is worth a patch.
#   b3m4k  fp8-k3 + ARC_MNBT=4096 -- shorter bursts; probes the 2.5x
#          identical-chunk variance (power? boundary?).
# Faces: U8k 600 + U32k 300 per leg, distinct tags (r137 defect fix:
# docker cp the fresh container JSON, never cp a stale local copy).
# Metric of record: per-chunk enqueue wall + gap from p136 telemetry
# (p136_decompose.py), not client TTFT (B2 differs in MAXLEN too).
set -uo pipefail
R=/root/build
CT=$R/p636
SP=/opt/venv/lib/python3.12/site-packages
PROBE=vllm-arc:v0.30.0-b70-cert23-p136probe
NAME=arc-p136
mkdir -p "$CT"

docker image inspect "$PROBE" >/dev/null 2>&1 || { echo "P136B_FAIL_NO_PROBE_IMAGE $PROBE"; exit 1; }

# Entry-time W14 quiesce (r137 law): a live W14 (host-network container)
# holds the B70s at 0.955 GMU and answers rig :8000 -- arc legs started
# after a W14 restore OOM at worker init and pass bogus health checks.
docker stop radiance-test >/dev/null 2>&1
sleep 15
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  echo "P136B_WARN port 8000 still held after W14 stop -- waiting"
  for i in 1 2 3 4 5 6; do sleep 10; ss -tln 2>/dev/null | grep -q ':8000 ' || break; done
fi

w14_restore() {
  docker rm -f "$NAME" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p636b.log 2>&1 &
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
  echo "P136B_FAIL_STALE_CONTAINER $NAME"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[p136b:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec "$NAME" pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] 2>/dev/null && { echo "[p136b:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[p136b:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

face() {
  # $1=tag $2=ctx $3=max_tokens $4=outdir $5=stem $6=model
  local TAG=$1 CX=$2 MX=$3 OD=$4 STEM=$5 MODEL=$6
  docker exec "$NAME" sh -c "rm -f /root/p135_face_$TAG.json; : > /root/p136_prefill.log"
  docker exec -d "$NAME" /opt/venv/bin/python /root/p135_face.py "$CX" "$MX" "$TAG" "$MODEL"
  local i=0
  while [ $i -lt 250 ]; do docker exec "$NAME" test -f /root/p135_face_$TAG.json && break; sleep 2; i=$((i+1)); done
  docker cp "$NAME":/root/p136_prefill.log "$OD/prefill_$STEM.log" >/dev/null 2>&1
  docker cp "$NAME":/root/p135_face_$TAG.json "$OD/face_$STEM.json" >/dev/null 2>&1
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$OD/metrics_$STEM.txt" || true
  echo "FACE_DONE_$TAG waited=$((i*2))s"
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
    echo "P136B_BOOT_FAIL_$LEGN (continuing)"
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
  if [ "${L:-0}" -lt 5 ]; then echo "P136B_FAIL_NO_TELEMETRY_$LEGN"; teardown >/dev/null 2>&1; return 1; fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics_warm.txt" || true

  docker cp $R/p635/p135_face.py "$NAME":/root/p135_face.py
  face "$LEGN" 8192 600 "$CD" u8k "$SERVED"
  face "${LEGN}32" 32768 300 "$CD" u32k "$SERVED"

  docker cp "$NAME":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  teardown || return 1
  sleep 3
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

# ---- legs (order: highest expected value first) ----
run_leg b1m16 qwen3.8-27b-fp8 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 ARC_MNBT=16384 || true

run_leg b2bf16 qwen3.8-27b \
  ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=0 ARC_GMU=0.9 \
  ARC_MAXLEN=32768 || true

run_leg b3m4k qwen3.8-27b-fp8 \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1 ARC_MNBT=4096 || true

echo "P636_LEG_B_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- decompose + compare vs l1fp8"
