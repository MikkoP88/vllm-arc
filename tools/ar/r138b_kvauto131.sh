#!/bin/bash
# r138b_kvauto131.sh -- rig-side supplement to r138 (P6.37 close-out, G).
# The kvauto cell (kv-cache-dtype=auto, i.e. fp16 KV) banked a BOOT_FAIL:
# one 262144-token request needs 8.73 GiB KV > 6.78 GiB available at GMU 0.8.
# This leg re-measures auto at ARC_MAXLEN=131072 (the fp16-KV capacity
# ceiling vLLM itself reported: est. max ~202048; 131072 still covers the
# full 1K-128K report sweep), so the kv report carries a WORKING fp16-KV
# row plus the documented refusal at 262144. Same machinery as r138:
# W14 entry quiesce + trap restore, per-tag-seeded p138_face, kv plan
# (8k + 32k faces). Artifacts -> /root/build/p637/kvauto131/.
#   r138b law: run ONLY after the r138 gate exited (single arc container).
set -uo pipefail
R=/root/build
P=$R/p637
IMG=vllm-arc:v0.30.0-b70-cert23
mkdir -p "$P"

w14_restore() {
  docker rm -f arc-r138 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p637b.log 2>&1 &
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
  echo "R138B_FAIL_STALE_CONTAINER $1"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r138b] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r138 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r138b] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r138b] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

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

# ---- entry: W14 quiesce (r137b law). The r138 gate's trap re-started
#      radiance-test; stop it again and wait :8000 free. ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R138B_WARN_PORT8000_STILL_HELD"
sleep 10

QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
CELLN=kvauto131
SERVED=qwen3.8-27b-fp8
CD=$P/$CELLN
mkdir -p "$CD"
echo "=== [r138b:$CELLN] boot $(date +%H:%M:%S) ARC_KVDTYPE=auto ARC_MAXLEN=131072 ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r138 $QF8 ARC_SPEC=3 \
  ARC_KVDTYPE=auto ARC_MAXLEN=131072 \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &

if ! boot_wait; then
  echo "R138B_BOOT_FAIL_$CELLN -- banking log"
  docker cp arc-r138:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r138 tail -30 /root/serve_full.log 2>/dev/null
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  exit 1
fi
echo "HEALTH_OK_r138b_$CELLN $(date +%H:%M:%S)"
grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
cat "$CD/posture.txt"
sleep 5
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
sleep 3
docker cp $R/p637/p138_face.py arc-r138:/root/p138_face.py

for C in 8192 32768; do
  face "$C" 600 "${CELLN}_c${C}_t0.0" 0.0
done

curl -s -m 10 http://127.0.0.1:8000/metrics > "$CD/metrics.txt" || true
docker cp arc-r138:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec arc-r138 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f arc-r138 >/dev/null 2>&1
verify_removed arc-r138 || exit 1
echo "FACES_OK" > "$CD/outcome.txt"
sleep 3
echo "=== [r138b:$CELLN] done $(date +%H:%M:%S) ==="

cd "$P"
find "$CELLN" -name "p138_face_*.json" -exec sha256sum {} \; > SHA256SUMS.b
find "$CELLN" -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS.b
echo "P637B_KVAUTO131_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
