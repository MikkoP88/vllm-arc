#!/bin/bash
# r136b_makeup.sh -- re-run the two cells the r136 matrix lost to
# stale-teardown cascades (q4_0 boot-failed on the stale record from
# q8_0's slow rm; q8_3 same after qf5's). Identical protocol/postures;
# teardown hardened: rm retried up to 90 s before declaring stale.
set -uo pipefail
R=/root/build
P=$R/p635/vmatrix
IMG=vllm-arc:v0.30.0-b70-cert23

w14_restore() {
  docker rm -f arc-vm >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635vm2.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/arc-vm$ -q)" ] && return 0
    docker rm -f arc-vm >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "VM2_FAIL_STALE_CONTAINER arc-vm"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-vm grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-vm >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[vm2:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-vm pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[vm2:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[vm2:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

run_cell() {
  local CELLN=$1; shift
  local CD=$P/$CELLN
  rm -rf "$CD"; mkdir -p "$CD"
  echo "=== [vm2:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-vm "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "VM2_BOOT_FAIL_$CELLN"
    docker cp arc-vm:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-vm tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f arc-vm >/dev/null 2>&1; verify_removed arc-vm || true
    return 0
  fi
  echo "HEALTH_OK_vm2_$CELLN $(date +%H:%M:%S)"
  sleep 5
  echo "db_info_lines=$(gcount 'graph-exec double-buffer captured')" > "$CD/db_markers.txt"
  echo "arcdb_capture_passes=$(gcount 'arc-db')" >> "$CD/db_markers.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/db_markers.txt"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  cat "$CD/db_markers.txt"; cat "$CD/posture.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  python3 $R/r113_battery.py decode "$SERVED" > "$CD/battery.txt" 2>&1 || echo "BATTERY_NONZERO_EXIT $CELLN"
  tail -6 "$CD/battery.txt"
  docker cp $R/p635/p135_face.py arc-vm:/root/p135_face.py
  docker exec -d arc-vm /opt/venv/bin/python /root/p135_face.py 8192 600 "$CELLN" "$SERVED"
  local i2=0
  while [ $i2 -lt 250 ]; do docker exec arc-vm test -f /root/p135_face_$CELLN.json && break; sleep 2; i2=$((i2+1)); done
  echo "FACE_DONE_vm2_$CELLN waited=$((i2*2))s"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-vm:/root/p135_face_$CELLN.json "$CD/" >/dev/null 2>&1 || echo "face MISSING $CELLN"
  docker cp arc-vm:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-vm bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-vm >/dev/null 2>&1
  verify_removed arc-vm || return 1
  sleep 3
  echo "=== [vm2:$CELLN] done $(date +%H:%M:%S) ==="
}

docker stop radiance-test >/dev/null 2>&1
sleep 5

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"

# make-up 1: qwen mxfp4-W4A16 spec0
SERVED=qwen3.8-27b
run_cell q4_0 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0

# make-up 2: qwen mxfp8 spec3
SERVED=qwen3.8-27b
run_cell q8_3 $QMX ARC_QUANT=mxfp8 ARC_SPEC=3

# make-up 3: gemma mxfp8 (stale casualty 3, after q4_5's teardown)
SERVED=gemma4-31b
run_cell gm8 ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
  ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 \
  ARC_MAXLEN=40960 ARC_FP8=0 ARC_QUANT=mxfp8

echo "P635_VM2_MAKEUP_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
