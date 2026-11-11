#!/bin/bash
# r140b_gm4x131k.sh -- rig-side surgical re-run of the three gm4x 131K
# faces (P6.39). Root cause of the r140 HTTP 400s, measured on the rig
# (throwaway cert23 container, 2026-11-11): the harness builds ctx-8
# token ids and sends decoded TEXT; the server re-tokenizes, and gemma's
# sentencepiece inflates 1.026x -- 131064 ids became 134479 server
# tokens > max_model_len 131072 -> HTTP 400 on every 131K face (the
# <=32K faces passed under the boot's 131072 len; qwen BPE round-trips
# exact, proven by its banked 131K faces passing on the 8-id margin).
# The fixed p139_conc.py (verify-and-trim; no-op for exact round-trips)
# ships with this gate. The <=32K gm4x faces and every other r139/r140
# face are untouched -- banks stand. Boot posture byte-exact with the
# r140 gm4x cell. Faces re-run:
#   gm4x_c131072_n1, gm4x_c131072_n2, gm4x_c131072_n1_t1.0
# Original 400 faces preserved rig-side under r140/gm4x/http400_orig/.
# Bank -> /root/build/r140b/gm4x/. W14 entry quiesce + trap restore.
# ETA ~15-25 min.
set -uo pipefail
R=/root/build
P=$R/r140b
IMG=vllm-arc:v0.30.0-b70-cert23
CN=arc-r140b
MT=600
CELLN=gm4x
CD=$P/gm4x
mkdir -p "$CD"

w14_restore() {
  docker rm -f "$CN" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r140b.log 2>&1 &
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
  echo "R140B_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec "$CN" grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect "$CN" >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r140b:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec "$CN" pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r140b:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r140b:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec "$CN" grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

group() {  # ctx conc tag [temp] -- runs one concurrency group, banks the JSON
  local CTX=$1 NC=$2 TAG=$3 TP=${4:-0.0}
  if [ -n "$CELL_DEAD" ]; then
    echo "KV_SKIP_ENGINE_DEAD $CELLN $TAG" | tee -a "$CD/conc.log"
    return 0
  fi
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  docker exec "$CN" rm -f /root/p139_conc_$TAG.json
  docker exec -d "$CN" /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec "$CN" test -f /root/p139_conc_$TAG.json && break
    docker inspect "$CN" >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec "$CN" test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec "$CN" tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp "$CN":/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

# ---- preserve the banked 400 faces rig-side before anything moves ----
mkdir -p $R/r140/gm4x/http400_orig
for f in gm4x_c131072_n1 gm4x_c131072_n2 gm4x_c131072_n1_t1.0; do
  if [ -f "$R/r140/gm4x/p139_conc_$f.json" ] && \
     [ ! -f "$R/r140/gm4x/http400_orig/p139_conc_$f.json" ]; then
    mv "$R/r140/gm4x/p139_conc_$f.json" "$R/r140/gm4x/http400_orig/"
  fi
done

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R140B_WARN_PORT8000_STILL_HELD"
sleep 10

SERVED=gemma4-31b
GM128="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0"

echo "=== [r140b:gm4x] boot $(date +%H:%M:%S) env=[$GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16] ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME="$CN" $GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
if ! boot_wait gm4x; then
  echo "R140B_BOOT_FAIL_gm4x -- banking log, gate aborts (banked r140 faces stand)"
  docker cp "$CN":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  exit 1
fi
echo "HEALTH_OK_r140b_gm4x $(date +%H:%M:%S)"
grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
POOL=$(kv_pool)
echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
[ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
cat "$CD/posture.txt"
sleep 5
curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
sleep 3
docker cp $R/p139_conc.py "$CN":/root/p139_conc.py
CELL_DEAD=""

group 131072 1 gm4x_c131072_n1
group 131072 2 gm4x_c131072_n2
group 131072 1 gm4x_c131072_n1_t1.0 1.0

docker cp "$CN":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec "$CN" bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f "$CN" >/dev/null 2>&1
verify_removed "$CN" || true
echo "CONC_OK" > "$CD/outcome.txt"
sleep 3
cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "R140B_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + face overwrite + report gen next"
