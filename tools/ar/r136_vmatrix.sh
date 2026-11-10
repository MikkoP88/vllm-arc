#!/bin/bash
# r136_vmatrix.sh -- P6.35-V full validation matrix on cert23 (the shipping
# image, db default ON). Owner-directed matrix: qwen3.8-27b x {fp8, mxfp8,
# mxfp4-W4A16} x {spec0, k3, k5} + gemma-4-31b x {fp8, mxfp8, mxfp4-W4A16}
# spec0 (gemma has no MTP head; LMC=0 posture per gemma4-lmc memory).
# Cells already double-covered on cert23 (qwen fp8 k3: dbon 94.5 + dboff
# replicate) are NOT re-run; every other cell runs the uniform protocol:
# boot-health, db markers, r113 battery decode, p135_face 8192 600.
# Postures byte-exact from the r135 sweep boot echoes:
#   fp8  -> model qwen3.8-27b-fp8, quant fp8 (default), no QCFG, GMU 0.8, FP8=1
#   mx8  -> model qwen3.8-27b BASE, quant mxfp8 + QCFG ignore, GMU 0.85, FP8=0
#   mx4  -> model qwen3.8-27b BASE, quant mxfp4 w4a16 + QCFG ignore, GMU 0.85, FP8=0
#   gemma-> gemma4-31b, GMU 0.8, LMC 0, RPC 1200, MAXLEN 40960, FP8=0, SPEC 0
# Boot failure does NOT abort the matrix: the failing cell's serve.log is
# banked as that cell's result (a refusal is a valid matrix outcome).
# trap restores W14. Artifacts -> /root/build/p635/vmatrix/<cell>/.
set -uo pipefail
R=/root/build
P=$R/p635/vmatrix
IMG=vllm-arc:v0.30.0-b70-cert23
mkdir -p "$P"
FAILED=""

w14_restore() {
  docker rm -f arc-vm >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p635vm.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 15 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "VM_FAIL_STALE_CONTAINER $1"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[vm:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-vm pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[vm:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[vm:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

run_cell() {
  # $1=cellname $2..=env assignments (KEY=VAL words, passed to env)
  local CELLN=$1; shift
  local CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [vm:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-vm "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "VM_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-vm:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-vm tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f arc-vm >/dev/null 2>&1; verify_removed arc-vm || true
    FAILED="$FAILED $CELLN"
    sleep 3
    return 0
  fi
  echo "HEALTH_OK_vm_$CELLN $(date +%H:%M:%S)"
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
  echo "FACE_DONE_vm_$CELLN waited=$((i2*2))s"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-vm:/root/p135_face_$CELLN.json "$CD/" >/dev/null 2>&1 || echo "face MISSING $CELLN"
  docker cp arc-vm:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-vm bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-vm >/dev/null 2>&1
  verify_removed arc-vm || return 1
  sleep 3
  echo "=== [vm:$CELLN] done $(date +%H:%M:%S) ==="
}

docker stop radiance-test >/dev/null 2>&1
sleep 5

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"
GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

# -- new cells first (the matrix gaps), then cert23 re-runs for single-batch
#    uniformity (references: sweep cert22 / gate cert23) --

# 1. qwen fp8 spec0
SERVED=qwen3.8-27b-fp8
run_cell qf0 $QF8 ARC_SPEC=0

# 2. qwen mxfp8 spec0
SERVED=qwen3.8-27b
run_cell q8_0 $QMX ARC_QUANT=mxfp8 ARC_SPEC=0

# 3. qwen mxfp4 spec0
SERVED=qwen3.8-27b
run_cell q4_0 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0

# 4. qwen mxfp8 spec5
SERVED=qwen3.8-27b
run_cell q8_5 $QMX ARC_QUANT=mxfp8 ARC_SPEC=5

# 5. gemma fp8 (online fp8 on the bf16 base -- never run; refusal = result)
SERVED=gemma4-31b
run_cell gf8 $GM ARC_QUANT=fp8

# 6. gemma mxfp4 W4A16
SERVED=gemma4-31b
run_cell gm4 $GM ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

# 7. qwen fp8 spec5 (ref: sweep cert22 115.2)
SERVED=qwen3.8-27b-fp8
run_cell qf5 $QF8 ARC_SPEC=5

# 8. qwen mxfp8 spec3 (ref: sweep cert22 83.8)
SERVED=qwen3.8-27b
run_cell q8_3 $QMX ARC_QUANT=mxfp8 ARC_SPEC=3

# 9. qwen mxfp4 spec3 (ref: sweep cert22 110.4)
SERVED=qwen3.8-27b
run_cell q4_3 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3

# 10. qwen mxfp4 spec5 (ref: gate cert23 mx4k5db 121.0)
SERVED=qwen3.8-27b
run_cell q4_5 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=5

# 11. gemma mxfp8 (ref: gate cert23 gemmam8 26.0)
SERVED=gemma4-31b
run_cell gm8 $GM ARC_QUANT=mxfp8

cd "$P"
find . -name "p135_face_*.json" -exec sha256sum {} \; > SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "P635_VMATRIX_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- analysis next"
