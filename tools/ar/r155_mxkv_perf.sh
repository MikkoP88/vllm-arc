#!/bin/bash
# r155_mxkv_perf.sh -- P6.47 leg D: P6.46 context-ladder perf on cert24 MX
# KV lanes (standing directive docs/TESTING-arc.md: rungs {1K,8K,32K,64K,
# 128K} x conc {1,2,3,4,8,+pool-wall} t0.0 + one t1.0 solo face per rung;
# infeasible points bank as KV_CAP rows; feasibility from the booted pool:
# need = conc*(ctx+600+64)).
#
# Cells (argv subset; rig-window law: qwen pair and gemma pair run as two
# windows, never leave the rig on an arc image overnight):
#   q8m8  qwen3.8-27b-fp8 champion chain (LMC=1 spec3) + ARC_KVDTYPE=mxfp8
#   q8m4  same + ARC_KVDTYPE=mxfp4
#   g8m4  gemma4-31b W8A16-mxfp8 weights @131072 + ARC_KVDTYPE=mxfp4 --
#         THE boundary cell: r140 gm8x (fp8-class KV) BOOT_FAILED at
#         131072; the 4-bit KV pool (~0.52x bytes) may buy the boot
#   g4m4  gemma4-31b W4A16-mxfp4 @131072 + ARC_KVDTYPE=mxfp4 (max-savings)
#
# Native-lane baselines: r139 banked qwen fp8-KV ladders (same posture,
# cert23; the cert24 overlay engages ONLY on mxfp4/mxfp8 dtypes so the
# native lane is byte-identical). W14 entry quiesce + trap restore.
# Artifacts -> /root/build/p647/r155/<cell>/.
set -uo pipefail
R=/root/build
P=$R/p647/r155
IMG=vllm-arc:v0.30.0-b70-cert24
MT=600
mkdir -p "$P"
FAILED=""
CELL_DEAD=""

w14_restore() {
  docker rm -f arc-r155 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r155.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/arc-r155$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "STALE_CONTAINER arc-r155"; return 1
}

gcount() {
  local c
  c=$(docker exec arc-r155 grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect arc-r155 >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r155:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec arc-r155 pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r155:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r155:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec arc-r155 grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
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
  docker exec arc-r155 rm -f /root/p139_conc_$TAG.json
  docker exec -d arc-r155 /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" "$TP"
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec arc-r155 test -f /root/p139_conc_$TAG.json && break
    docker inspect arc-r155 >/dev/null 2>&1 || { CELL_DEAD=1; break; }
    sleep 4; i=$((i+1))
  done
  if ! docker exec arc-r155 test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s dead=$CELL_DEAD)" | tee -a "$CD/conc.log"
    [ -n "$CELL_DEAD" ] && docker exec arc-r155 tail -5 /root/serve_full.log 2>/dev/null | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp arc-r155:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3..=env (byte-exact); full ladder (P6.46 law)
  CELLN=$1; SERVED=$2; shift 2
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r155:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=arc-r155 "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R155_BOOT_FAIL_$CELLN -- banking log, run continues"
    docker cp arc-r155:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec arc-r155 tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f arc-r155 >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    sleep 3; SERVED=""
    return 0
  fi
  echo "HEALTH_OK_r155_$CELLN $(date +%H:%M:%S)"
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  {
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    docker exec arc-r155 grep -E "GPU KV cache size|Maximum concurrency" /root/serve_full.log 2>/dev/null | head -4
  } >> "$CD/posture.txt"
  POOL=$(kv_pool)
  echo "kv_pool_tokens=$POOL" >> "$CD/posture.txt"
  [ "$POOL" = "0" ] && echo "POOL_PARSE_FAIL -- sweep runs ungated" >> "$CD/posture.txt"
  cat "$CD/posture.txt"
  sleep 5
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  docker cp $R/p139_conc.py arc-r155:/root/p139_conc.py
  CELL_DEAD=""
  # canonical rung ladder incl. 64K/128K, swept to the pool wall
  local CTX c maxf ran
  for CTX in 1024 8192 32768 65536 131072; do
    ran=""; maxf=0
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
  docker cp arc-r155:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec arc-r155 bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-r155 >/dev/null 2>&1
  verify_removed || return 1
  echo "CONC_OK" > "$CD/outcome.txt"
  sleep 3
  echo "=== [r155:$CELLN] done $(date +%H:%M:%S) ==="
  SERVED=""
}

# ---- entry: W14 quiesce (r137b law) ----
docker stop radiance-test >/dev/null 2>&1
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R155_WARN_PORT8000_STILL_HELD"
sleep 10

QWEN="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_FP8=1"
GM131="ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b ARC_GMU=0.8 ARC_GRAPH=1 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 ARC_MAXLEN=131072 ARC_FP8=0"
GMQCFG='{"ignore":["*visual*","*linear_attn*","mtp.fc"]}'

for CELL in "$@"; do
  case $CELL in
    q8m8) run_cell q8m8 qwen3.8-27b-fp8 $QWEN ARC_KVDTYPE=mxfp8 ;;
    q8m4) run_cell q8m4 qwen3.8-27b-fp8 $QWEN ARC_KVDTYPE=mxfp4 ;;
    g8m4) run_cell g8m4 gemma4-31b $GM131 ARC_QUANT=mxfp8 ARC_KVDTYPE=mxfp4 ;;
    g4m4) run_cell g4m4 gemma4-31b $GM131 ARC_QUANT=mxfp4 "ARC_QUANTCFG=$GMQCFG" ARC_MXFP4_MODE=w4a16 ARC_GMU=0.85 ARC_KVDTYPE=mxfp4 ;;
    *) echo "UNKNOWN_CELL $CELL (q8m8|q8m4|g8m4|g4m4)";;
  esac
done

cd "$P"
find . -name "p139_conc_*.json" -exec sha256sum {} \; > SHA256SUMS
find . -name "serve.log" -exec sha256sum {} \; >> SHA256SUMS
echo "BOOT_FAILS:${FAILED:-none}"
echo "R155_COMPLETE cells=[$*] $(date -u +%Y-%m-%dT%H:%M:%SZ) -- pull + report gen next"
