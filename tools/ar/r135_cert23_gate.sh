#!/bin/bash
# r135_cert23_gate.sh -- cert23 (P6.35 D1 graph-exec double-buffer) §10 gate.
# Image vllm-arc:v0.30.0-b70-cert23 = cert22 + patches/graph_exec_doublebuffer.py
# (VLLM_ARC_GRAPH_EXEC_DB gate, default ON). Three legs on the rig:
#   L1 qwen fp8 spec3 db-default(ON)  -- expect db markers (x2-capture info
#      line per worker + "arc-db" second-capture passes), ITL p50 ~= cert22
#      baseline 40.5 ms (leg K: step wall is the fp8 pipeline floor, the db
#      fix removes the replay block INSIDE the step, engine serial 40.8->12.7),
#      battery decode sane, 0 Tracebacks.
#   L2 qwen fp8 spec3 VLLM_ARC_GRAPH_EXEC_DB=0 -- the stock control: NO db
#      markers, face ~= cert22 byte-identical posture.
#   L3 gemma4-31b mxfp8 spec0 (cert22 gemma posture, r127g line) -- db is
#      model-agnostic; boot + markers + battery + face.
# trap restores W14. Artifacts -> /root/build/p635/cert23/<leg>/.
set -uo pipefail
R=/root/build
P=$R/p635/cert23
IMG=vllm-arc:v0.30.0-b70-cert23
mkdir -p "$P"

w14_restore() {
  docker rm -f arc-c23 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635c23.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "GATE_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-c23 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-c23 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-c23 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

run_leg() {
  # $1=legname $2..=env assignments (KEY=VAL words, passed to env)
  local LEGN=$1; shift
  local CD=$P/$LEGN
  mkdir -p "$CD"
  echo "=== [$LEGN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-c23 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEGN"; then
    echo "GATE_FAIL_BOOT_$LEGN"
    docker cp arc-c23:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-c23 tail -40 /root/serve_full.log 2>/dev/null
    docker rm -f arc-c23 >/dev/null 2>&1; verify_removed arc-c23 || true
    return 1
  fi
  echo "HEALTH_OK_$LEGN $(date +%H:%M:%S)"
  sleep 5
  # db evidence
  echo "db_info_lines=$(gcount 'graph-exec double-buffer captured')" > "$CD/db_markers.txt"
  echo "arcdb_capture_passes=$(gcount 'arc-db')" >> "$CD/db_markers.txt"
  echo "db_optout_seen=$(gcount 'VLLM_ARC_GRAPH_EXEC_DB=0')" >> "$CD/db_markers.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/db_markers.txt"
  cat "$CD/db_markers.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  # battery (decode class)
  python3 $R/r113_battery.py decode "$SERVED" > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT $LEGN"
  tail -6 "$CD/battery.txt"
  # face
  docker cp $R/p635/p135_face.py arc-c23:/root/p135_face.py
  docker exec -d arc-c23 /opt/venv/bin/python /root/p135_face.py 8192 600 "$LEGN" "$SERVED"
  local i2=0
  while [ $i2 -lt 250 ]; do docker exec arc-c23 test -f /root/p135_face_$LEGN.json && break; sleep 2; i2=$((i2+1)); done
  echo "FACE_DONE_$LEGN waited=$((i2*2))s"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-c23:/root/p135_face_$LEGN.json "$CD/" >/dev/null 2>&1 || echo "face MISSING $LEGN"
  docker cp arc-c23:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-c23 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-c23 >/dev/null 2>&1
  verify_removed arc-c23 || return 1
  sleep 3
  echo "=== [$LEGN] done $(date +%H:%M:%S) ==="
}

docker stop radiance-test >/dev/null 2>&1
sleep 5

QWEN_FP8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1"

# L1: db default ON (no env)
SERVED=qwen3.8-27b-fp8
run_leg dbon $QWEN_FP8 || exit 1

# L2: stock control
SERVED=qwen3.8-27b-fp8
run_leg dboff $QWEN_FP8 VLLM_ARC_GRAPH_EXEC_DB=0 || exit 1

# L3: gemma mxfp8 spec0 (cert22 gemma line)
SERVED=gemma4-31b
run_leg gemmam8 ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_QUANT=mxfp8 ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 \
  ARC_MAXLEN=40960 ARC_FP8=0 || exit 1

cd "$P" && find . -name "p135_face_*.json" -exec sha256sum {} \; > SHA256SUMS
echo "LEG_ARTIFACTS $P"
echo "P635_CERT23_GATE_COMPLETE -- analysis next"
