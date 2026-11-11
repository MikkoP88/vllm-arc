#!/bin/bash
# r140_gemma128k.sh -- rig-side gemma 128K annex gate (P6.39).
# Owner directive (2026-11-11): individual gemma reports to 128K when
# using MXFP4 -- the 4-bit weight savings are what buy the KV headroom
# for max_model_len 131072 on the 31B two-head-size-GDN geometry. Two
# boots, postures byte-exact as the r139 gemma cells except ARC_MAXLEN:
#   gm8x -- gemma MXFP8  @ ARC_MAXLEN=131072 (the 8-bit control: boot
#          attempt; a refusal banks the "128K needs MXFP4" boundary; if
#          it boots, two probe faces prove it servable)
#   gm4x -- gemma MXFP4 W4A16 @ ARC_MAXLEN=131072: full ctx sweep
#          {1K, 8K, 32K, 64K, 128K} (the canonical rung ladder,
#          standing directive docs/TESTING-arc.md) x conc
#          {1,2,3,4,8} (t0.0) + a t1.0 solo face per ctx (same
#          convention as r139)
# Feasibility from the booted KV pool exactly as r139: need =
# conc*(ctx+600+64); infeasible points bank as KV_CAP rows.
# Faces: /root/p139_conc.py (already on the rig from r139). Groups
# sequential per boot. Boot failure does NOT abort (vmatrix law). W14
# entry quiesce + trap restore. Artifacts -> /root/build/r140/<cell>/.
# ETA ~3-4 h (2 boots; 64K/131K legs are long).
set -uo pipefail
R=/root/build
P=$R/r140
IMG=vllm-arc:v0.30.0-b70-cert23
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

w14_restore() {
  docker rm -f arc-r140 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r140.log 2>&1 &
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
  echo "R140_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r140 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r140 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r140:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r140 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r140:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r140:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec arc-r140 grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
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
  docker exec arc-r140 rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r140 /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r140 test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r140 >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r140 test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r140 tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r140:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3=family(bootprobe|gemma128) $4..=env (byte-exact)
  CELLN=$1; SERVED=$2; local FAM=$3; shift 3
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r140:$CELLN] boot $(date +%H:%M:%S) fam=$FAM env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r140 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R140_BOOT_FAIL_$CELLN -- banking log, gate continues"
    docker cp arc-r140:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r140 >/dev/null 2>&1; verify_removed arc-r140 || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r140_$CELLN $(date +%H:%M:%S)"
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
  docker cp $R/p139_conc.py arc-r140:/root/p139_conc.py
  CELL_DEAD=""

  if [ "$FAM" = "bootprobe" ]; then
    # the 8-bit control: prove servable at 8K + the biggest ctx that fits
    group 8192 1 "${CELLN}_c8192_n1"
    group 131072 1 "${CELLN}_c131072_n1"
    group 131072 1 "${CELLN}_c131072_n1_t1.0" 1.0
  else
    local CTX
    # standing directive: the canonical rung ladder incl. 64K — every
    # rung the boot's maxlen/KV pool permits, swept to the wall
    # (docs/TESTING-arc.md context-ladder law).
    for CTX in 1024 8192 32768 65536 131072; do
      local ran="" c
      local maxf=0
      for c in 1 2 3 4 5 6 7 8; do
        [ $(( c * (CTX + MT + 64) )) -le "$POOL" ] && maxf=$c
      done
      for c in 1 2 3 4 8 $maxf; do
        [ "$c" = "0" ] && continue
        case " $ran " in *" $c "*) continue ;; esac
        ran="$ran $c"
        group "$CTX" "$c" "${CELLN}_c${CTX}_n${c}"
      done
      group "$CTX" 1 "${CELLN}_c${CTX}_n1_t1.0" 1.0
    done
  fi

  docker cp arc-r140:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r140 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r140 >/dev/null 2>&1
  verify_removed arc-r140 || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r140:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R140_WARN_PORT8000_STILL_HELD"
sleep 10

GM128="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0"

mkdir -p $R/r140
cp -f $R/p139_conc.py $R/r140/p139_conc.py 2>/dev/null || true

# -- the 8-bit control first (short cell either way), then the MXFP4 sweep --
run_cell gm8x gemma4-31b bootprobe $GM128 ARC_QUANT=mxfp8
run_cell gm4x gemma4-31b gemma128 $GM128 ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "R140_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
