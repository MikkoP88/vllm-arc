#!/bin/bash
# r162_telemetry_ladder.sh -- P6.47 A10: compact end-to-end decode/TTFT
# telemetry ladder (2026-11-12 requirement: comprehensive telemetry
# capture points end-to-end to locate the root issue).
#
# Cells q8 (native fp8_e4m3 KV) / m8 (mxfp8) / m4 (mxfp4) on $IMG, qwen
# champion chain byte-exact from r155. Streaming capture via p139_conc.py
# docker-cp'd into each container (the images do NOT bake it — maiden-run
# lesson 2026-11-12: exec -d of a missing script exits instantly, no JSON,
# every cell burns its cap in CONC_TIMEOUT): per-slot ttft_s + ITL
# p50/p90/p99 (inter-token latency =
# the decode step time at growing ctx) -- ITL deltas MX-vs-ctrl at conc1
# isolate the attn-path share (non-attn layers identical), TTFT carries
# the prefill axis. Rungs {1024, 8192, 32768} x conc {1, 8} minus the
# 8192/n8 cell (curve shape kept, wall-clock trimmed). Report:
# p147_telemetry_report.py over the banked JSONs.
# W14 quiesce before, trap restore after. Artifacts ->
# /root/build/p647/telemA10/<cell>/. Exit 0 = all cells banked.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/telemA10}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25x}
CNAME=arc-r162
MT=600
PROBE=$R/p139_conc.py  # canonical copy (API length-budget fix, r140c law)
[ -f "$PROBE" ] || { echo "R162_NO_PROBE $PROBE"; exit 1; }
mkdir -p "$P"
FAILED=""

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r162.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

verify_removed() {
  local i=0
  while [ $i -lt 30 ]; do
    [ -z "$(docker ps -a --filter name=^/"$CNAME"$ -q)" ] && return 0
    sleep 2; i=$((i + 1))
  done
  echo "STALE_CONTAINER $CNAME"; return 1
}

gcount() {
  local c
  c=$(docker exec $CNAME grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r162:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r162:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r162:$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

kv_pool() {
  local v
  v=$(docker exec $CNAME grep -o "KV cache size: [0-9,]* tokens" /root/serve_full.log 2>/dev/null | head -1)
  v=${v#KV cache size: }; v=${v% tokens}; v=${v//,/}
  echo "${v:-0}"
}

group() {  # ctx conc tag -- one streaming group, banks the JSON
  local CTX=$1 NC=$2 TAG=$3
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)" | tee -a "$CD/conc.log"
    return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=served $3..=env (byte-exact from r155)
  CELLN=$1; SERVED=$2; shift 2
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r162:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R162_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r162_$CELLN $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R162_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true; return 0; }
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
    /root/serve_full.log 2>/dev/null | head -3 > "$CD/markers.txt" || true
  POOL=$(kv_pool); echo "POOL=$POOL" | tee -a "$CD/markers.txt"
  # warm the engine (first-hit compiles/pages) before timing cells
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  group 1024 1  c1024_n1
  group 1024 8  c1024_n8
  group 8192 1  c8192_n1
  group 32768 1 c32768_n1
  group 32768 8 c32768_n8
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "R162_CELL_DONE_$CELLN $(date +%H:%M:%S)"
}

# P6.47 r168 posture: ARC_LMC is no longer pinned here — the serve
# script's default applies per lane (1 for q8/m8; auto-off for the m4
# e2m1-V lane, the r168-discriminated LMCache double-quant garble
# driver). The battery measures the shipping posture of each cell.
QWEN="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1"

echo "=== [r162:A10] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R162_WARN_PORT8000_STILL_HELD"
sleep 10

for CELL in ${CELLS:-q8 m8 m4}; do
  case $CELL in
    q8) run_cell q8 qwen3.8-27b-fp8 $QWEN ;;
    m8) run_cell m8 qwen3.8-27b-fp8 $QWEN ARC_KVDTYPE=mxfp8 ;;
    m4) run_cell m4 qwen3.8-27b-fp8 $QWEN ARC_KVDTYPE=mxfp4 ;;
    *) echo "UNKNOWN_CELL $CELL (q8|m8|m4)"; FAILED="$FAILED $CELL" ;;
  esac
done

# attribution report over the banked JSONs (host python3, stdlib)
python3 $R/p647/p147_telemetry_report.py "$P" \
  > "$P/REPORT.md" 2>&1 || echo "REPORT_NONZERO (bank intact)"
cat "$P/REPORT.md"

cd "$P"
find . -name 'p139_conc_*.json' -o -name 'posture.txt' -o -name 'markers.txt' \
  | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
echo "R162_TELEMETRY_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -z "$FAILED" ]
