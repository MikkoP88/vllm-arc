#!/bin/bash
# r140c_gm4x131k.sh -- rig-side probe + conditional re-run of the gm4x
# 131K faces (P6.39). Chain so far: r140 banked HTTP 400 on every 131K
# face; the r140b trim to <= max_model_len STILL 400'd (and the 400s
# return in seconds -- validation-layer rejection, no prefill). Working
# rule: the API server enforces prompt_len + max_tokens <= max_model_len
# (qwen's BPE round-trip shrinks slightly, so its banked 131K faces
# passed; gemma's sentencepiece inflates 1.026x and busts the wall).
# This gate makes the server state its own math verbatim: two boundary
# probes (131000/600 -> expect 400 with the rule in the body;
# 130300/600 -> expect 200), banked as p140_probe_*.json. Only on a
# 200 at the 130300 class do the three faces re-run under the v2
# harness (server-side prompt budget = ctx - max_tokens - 64); else the
# faces are skipped and the banked r140 400s stand. Boot posture
# byte-exact with the r140/r140b gm4x cell. Bank -> /root/build/r140c/.
# W14 entry quiesce + trap restore. ETA ~20-30 min.
set -uo pipefail
R=/root/build
P=$R/r140c
IMG=vllm-arc:v0.30.0-b70-cert23
CN=arc-r140c
MT=600
CELLN=gm4x
CD=$P/gm4x
mkdir -p "$CD"

w14_restore() {
  docker rm -f "$CN" >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r140c.log 2>&1 &
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
  echo "R140C_FAIL_STALE_CONTAINER $1"; return 1
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r140c:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec "$CN" pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r140c:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r140c:$1] ENGINE_DEAD poll=$i"; break; }
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

probe() {  # budget tag -- one boundary probe, banks status+body json
  local B=$1 TAG=$2
  docker exec "$CN" rm -f /root/p140_probe_$TAG.json
  docker exec -d "$CN" /opt/venv/bin/python /root/p140_probe.py "$B" 600 "$TAG"
  local i=0
  while [ $i -lt 270 ]; do
    docker exec "$CN" test -f /root/p140_probe_$TAG.json && break
    docker inspect "$CN" >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  if docker exec "$CN" test -f /root/p140_probe_$TAG.json 2>/dev/null; then
    docker cp "$CN":/root/p140_probe_$TAG.json "$CD/" >/dev/null 2>&1
    cat "$CD/p140_probe_$TAG.json" | tee -a "$CD/probe.log"
    echo "" >> "$CD/probe.log"
    echo "PROBE_OK_$TAG" | tee -a "$CD/probe.log"
  else
    echo "PROBE_TIMEOUT_$TAG" | tee -a "$CD/probe.log"
  fi
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

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R140C_WARN_PORT8000_STILL_HELD"
sleep 10

SERVED=gemma4-31b
GM128="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0"

echo "=== [r140c:gm4x] boot $(date +%H:%M:%S) env=[$GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16] ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME="$CN" $GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
if ! boot_wait gm4x; then
  echo "R140C_BOOT_FAIL_gm4x -- banking log, gate aborts (banked r140 faces stand)"
  docker cp "$CN":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  exit 1
fi
echo "HEALTH_OK_r140c_gm4x $(date +%H:%M:%S)"
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
docker cp $R/p140_probe.py "$CN":/root/p140_probe.py
docker cp $R/p139_conc.py "$CN":/root/p139_conc.py
CELL_DEAD=""

# ---- boundary probes: the server states its own rule ----
probe 131000 gm4x_probe_b131000
probe 130300 gm4x_probe_b130300

if grep -q '"status": 200' "$CD/p140_probe_gm4x_probe_b130300.json" 2>/dev/null; then
  echo "PROBE_GATE_PASS -- 130300+600 fits; faces run with ctx-mt-64 budget" | tee -a "$CD/probe.log"
  group 131072 1 gm4x_c131072_n1
  group 131072 2 gm4x_c131072_n2
  group 131072 1 gm4x_c131072_n1_t1.0 1.0
else
  echo "PROBE_GATE_FAIL -- faces skipped; the r140 banked 400s stand; body above states the rule" | tee -a "$CD/probe.log"
fi

docker cp "$CN":/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec "$CN" bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 8
docker rm -f "$CN" >/dev/null 2>&1
verify_removed "$CN" || true
if grep -q '"status": 200' "$CD/p140_probe_gm4x_probe_b130300.json" 2>/dev/null; then
  echo "CONC_OK" > "$CD/outcome.txt"
else
  echo "PROBE_GATE_FAIL" > "$CD/outcome.txt"
fi
sleep 3
cd "$P"
find . \( -name "p139_conc_*.json" -o -name "p140_probe_*.json" -o -name "serve.log" \) \
  -exec sha256sum {} \; > SHA256SUMS
echo "R140C_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
