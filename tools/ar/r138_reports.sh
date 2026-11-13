#!/bin/bash
# r138_reports.sh -- P6.37 close-out phases E+G: per-config performance
# report matrix on cert23 (shipping image, stock, no probe).
# Close-out requirement: individual reports per model x quant x MTP,
# decode/TTFT/prefill at ctx 1K/8K/32K/64K/128K (if fit), MTP k3 with
# temperature 0.0..1.0, all supported --kv-cache-dtype values.
# Cells (12 boots, ~2.5 h):
#   E  qf3/qf0  qwen fp8-ckpt        k3(+temp sweep)/k0  ctx 1k..128k
#      q83/q80  qwen mxfp8 online    k3(+temp sweep)/k0  ctx 1k..128k
#      q43/q40  qwen mxfp4 W4A16     k3(+temp sweep)/k0  ctx 1k..128k
#      gf8/gm8/gm4 gemma fp8/mxfp8/mxfp4 spec0 (no MTP head; LMC=0 law)
#                ctx 1k/8k/32k (MAXLEN 40960)
#   G  kvauto/kvfp8/kve5m2 qwen fp8 k3, ARC_KVDTYPE in
#      {auto, fp8, fp8_e5m2} @ 8k+32k (e4m3 = the E qf3 cells)
# ALL legs ride the close-out F defaults (shm 16g, LMCache CPU 16 GB) --
# the run itself is the law-6 validating leg for those defaults.
# Faces: p138_face.py (per-tag prompt seed -- same-ctx faces on one boot
# must NOT share a prompt or LMCache serves them from cache).
# Boot failure does NOT abort: the cell's serve.log is its banked result
# (a refusal is a valid matrix outcome -- vmatrix law).
# W14 entry quiesce + trap restore. Artifacts -> /root/build/p637/<cell>/.
set -uo pipefail
R=/root/build
P=$R/p637
IMG=vllm-arc:v0.30.0-b70-cert23
mkdir -p "$P"
FAILED=""

w14_restore() {
  docker rm -f arc-r138 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p637.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$1"$ -q)" ] && return 0
    docker rm -f "$1" >/dev/null 2>&1
    sleep 3; i=$((i + 1))
  done
  echo "R138_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r138 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r138 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r138:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r138 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r138:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r138:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

# face <ctx> <maxtok> <tag> <temp>  (model = $SERVED, always passed --
# the p135 default-model 404 bit twice; r137b law)
face() {
  local CTX=$1 MT=$2 TAG=$3 TP=$4
  docker exec arc-r138 rm -f /root/p138_face_$TAG.json
  docker exec -d arc-r138 /opt/venv/bin/python /root/p138_face.py \
    "$CTX" "$MT" "$TAG" "$SERVED" "$TP"
  local i=0
  while [ $i -lt 250 ]; do
    docker exec arc-r138 test -f /root/p138_face_$TAG.json && break
    sleep 2; i=$((i+1))
  done
  if [ $i -ge 250 ]; then
    echo "FACE_TIMEOUT_$TAG"
    echo "FACE_TIMEOUT_$TAG" >> "$CD/faces.log"
    return 0
  fi
  docker cp arc-r138:/root/p138_face_$TAG.json "$CD/" >/dev/null 2>&1
  echo "FACE_OK_$TAG waited=$((i*2))s" | tee -a "$CD/faces.log"
}

run_cell() {
  # $1=cellname $2=served-model $3=plan(qwen3|qwen0|gemma|kv) $4..=env
  local CELLN=$1; SERVED=$2; local PLAN=$3; shift 3
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r138:$CELLN] boot $(date +%H:%M:%S) plan=$PLAN env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r138 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R138_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-r138:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-r138 tail -30 /root/serve_full.log 2>/dev/null
    docker rm -f arc-r138 >/dev/null 2>&1; verify_removed arc-r138 || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3
    SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r138_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p637/p138_face.py arc-r138:/root/p138_face.py

  case "$PLAN" in
  qwen3)
    for C in 1024 8192 32768 65536 131072; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done
    for T in 0.0 0.2 0.4 0.6 0.8 1.0; do
      face 8192 600 "${CELLN}_s8192_t${T}" "$T"
    done ;;
  qwen0)
    for C in 1024 8192 32768 65536 131072; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  gemma)
    for C in 1024 8192 32768; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  kv)
    for C in 8192 32768; do
      face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
    done ;;
  esac

  curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
  docker cp arc-r138:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r138 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r138 >/dev/null 2>&1
  verify_removed arc-r138 || return 1
  echo "FACES_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r138:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law -- a live W14 holds the B70s and
#      answers :8000 with bogus HEALTH_OK). Long wait: a mid-boot W14
#      restore can hold :8000 for a while after docker start. ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R138_WARN_PORT8000_STILL_HELD"
sleep 10

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"
GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

mkdir -p $R/p637
cp -f $R/p138_face.py $R/p637/p138_face.py 2>/dev/null || true

# ---- E cells: qwen 3 quants x {k3, k0} ----
run_cell qf3 qwen3.8-27b-fp8   qwen3 $QF8 ARC_SPEC=3
run_cell qf0 qwen3.8-27b-fp8   qwen0 $QF8 ARC_SPEC=0
run_cell q83 qwen3.8-27b       qwen3 $QMX ARC_QUANT=mxfp8 ARC_SPEC=3
run_cell q80 qwen3.8-27b       qwen0 $QMX ARC_QUANT=mxfp8 ARC_SPEC=0
run_cell q43 qwen3.8-27b       qwen3 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3
run_cell q40 qwen3.8-27b       qwen0 $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0

# ---- E cells: gemma 3 quants, spec0 ----
run_cell gf8 gemma4-31b gemma $GM ARC_QUANT=fp8
run_cell gm8 gemma4-31b gemma $GM ARC_QUANT=mxfp8
run_cell gm4 gemma4-31b gemma $GM ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

# ---- G cells: kv-cache-dtype sweep on qwen fp8 k3 ----
run_cell kvauto qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=auto
run_cell kvfp8 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=fp8
run_cell kve5m2 qwen3.8-27b-fp8 kv $QF8 ARC_SPEC=3 ARC_KVDTYPE=fp8_e5m2

cd "$P"
find . -name "p138_face_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "P637_REPORTS_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
