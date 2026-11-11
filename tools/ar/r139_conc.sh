#!/bin/bash
# r139_conc.sh -- rig-side P6.37 concurrency-sweep gate (close-out E addendum).
# Owner directive (2026-11-11): every individual report carries measured
# concurrency 1/2/3/4/8 series (tables + charts) + temperature 0.0 AND 1.0
# values, the KV report gains the gemma comparison and the turboquant_*
# presets, all world-class per the millstone example. Banked data covers
# conc 1 (r138 faces) and conc 8 (r113 battery); this leg measures the full
# curve per config on cert23, postures byte-exact as r138:
#   15 boots = 9 config cells (qf3 qf0 q83 q80 q43 q40 gf8 gm8 gm4)
#            + 4 turboquant cells (tq_k8v4 tq_4bit tq_k3v4 tq_3bit,
#              qwen-fp8 posture, ARC_KVDTYPE=turboquant_*)
#            + 2 gemma KV cells (kvauto @40960 expected refusal,
#              kvauto16384 working fp16-KV supplement).
#   per config cell: ctx in {1K, 8K, 32K} (+128K qwen-only) x conc in
#   {1,2,3,4,8} (t0.0) + a t1.0 solo face per ctx -- the temp 0.0/1.0
#   pair lands in every report row.
#   feasibility from the booted KV pool (serve.log "KV cache size: N
#   tokens"): need = conc*(ctx+max_tokens+64). Infeasible points bank as
#   KV_CAP rows -- the measured capacity wall itself; max feasible conc
#   runs when 8 does not fit.
# Faces: p139_conc.py (per-slot unique prompts -- the prompt-seed law
# applies per REQUEST; common submit barrier; per-request ttft so queue
# wait is measurable). Groups sequential per boot. Boot failure does NOT
# abort (vmatrix law). W14 entry quiesce + trap restore.
# Artifacts -> /root/build/p637b/<cell>/. ETA ~11-13 h (15 boots).
set -uo pipefail
R=/root/build
P=$R/p637b
IMG=vllm-arc:v0.30.0-b70-cert23
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

w14_restore() {
  docker rm -f arc-r139 >/dev/null 2>&1
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
  echo "R139_FAIL_STALE_CONTAINER $1"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r139 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r139 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r139:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r139 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r139:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r139:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {  # parse "KV cache size: N tokens" from the booted engine
  local v
  v=$(docker exec arc-r139 grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
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
  docker exec arc-r139 rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r139 /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r139 test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r139 >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r139 test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r139 tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r139:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3=family(qwen|gemma|kvface) $4..=env (byte-exact r138)
  # family kvface: $3="kvface:<ctx:conc ...>" -- explicit face list, no sweep.
  CELLN=$1; SERVED=$2; local SPEC=$3; shift 3
  local FAM=${SPEC%%:*} KVFACES=${SPEC#*:}
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r139:$CELLN] boot $(date +%H:%M:%S) fam=$FAM env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r139 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R139_BOOT_FAIL_$CELLN -- banking log, matrix continues"
    docker cp arc-r139:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f arc-r139 >/dev/null 2>&1; verify_removed arc-r139 || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r139_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  echo "tracebacks=$(gcount 'Traceback (most recent call last)')" >> "$CD/posture.txt"
  echo "tq_hook_lines=$(gcount 'tq_spec_fix')" >> "$CD/posture.txt"
  POOL=$(kv_pool)
  echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p139_conc.py arc-r139:/root/p139_conc.py
  CELL_DEAD=""

  if [ "$FAM" = "kvface" ]; then
    # explicit face list "ctx:conc[:temp] ..." (gemma KV supplement)
    for f in $KVFACES; do
      local fx=${f%%:*} rest=${f#*:} fn=${rest%%:*} ft=${rest#*:}
      [ "$ft" = "$rest" ] && ft=0.0
      group "$fx" "$fn" "${CELLN}_c${fx}_n${fn}_t${ft}" "$ft"
    done
  else
    local CTXS="1024 8192 32768 131072"
    [ "$FAM" = "gemma" ] && CTXS="1024 8192 32768"
    for CTX in $CTXS; do
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
      # the t1.0 solo face: the temp 0.0/1.0 pair per ctx row (n1 conc,
      # unique-prompt seed derives from the t-tag so no cache collision)
      group "$CTX" 1 "${CELLN}_c${CTX}_n1_t1.0" 1.0
    done
  fi

  docker cp arc-r139:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r139 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r139 >/dev/null 2>&1
  verify_removed arc-r139 || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r139:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R139_WARN_PORT8000_STILL_HELD"
sleep 10

QCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'
QF8="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=1"
QMX="ARC_MODEL=/models/qwen3.8-27b ARC_SERVED=qwen3.8-27b ARC_QUANTCFG=$QCFG ARC_GMU=0.85 ARC_GRAPH=1 ARC_LMC=1 ARC_FP8=0"
GM="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=40960 ARC_FP8=0"

mkdir -p $R/p637b
cp -f $R/p139_conc.py $R/p637b/p139_conc.py 2>/dev/null || true

# -- the 9 report cells (order = report priority; qwen spec first) --
run_cell qf3 qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3
run_cell qf0 qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=0
run_cell q83 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp8 ARC_SPEC=3
run_cell q80 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp8 ARC_SPEC=0
run_cell q43 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=3
run_cell q40 qwen3.8-27b      qwen $QMX ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16 ARC_SPEC=0
run_cell gf8 gemma4-31b gemma $GM ARC_QUANT=fp8
run_cell gm8 gemma4-31b gemma $GM ARC_QUANT=mxfp8
run_cell gm4 gemma4-31b gemma $GM ARC_QUANT=mxfp4 ARC_MXFP4_MODE=w4a16

# -- the 4 turboquant cells (qwen-fp8 certified posture, KV preset knob) --
run_cell tq_k8v4  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_k8v4
run_cell tq_4bit  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_4bit_nc
run_cell tq_k3v4  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_k3v4_nc
run_cell tq_3bit  qwen3.8-27b-fp8 qwen $QF8 ARC_SPEC=3 ARC_KVDTYPE=turboquant_3bit_nc

# -- gemma KV-cache-dtype comparison (auto = fp16 KV; expect 40960 refusal,
#    16384 working -- the same bank-the-cliff pattern as qwen kvauto) --
run_cell gkv_auto   gemma4-31b "kvface:8192:1:0.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=auto
run_cell gkv_auto16 gemma4-31b "kvface:8192:1:0.0 16384:1:0.0" $GM ARC_QUANT=fp8 ARC_KVDTYPE=auto ARC_MAXLEN=16384

cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "P637B_CONC_COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
