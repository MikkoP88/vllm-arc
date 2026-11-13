#!/bin/bash
# r172_deepcapture.sh -- P6.47 A14: deep end-to-end latency attribution
# capture (2026-11-13 goal: comprehensive
# capturing using capture points to get root delays/latencies of
# --kv-cache-dtype mxfp4 and mxfp8 vs fp8, and fix all).
#
# The r162 e2e ladder isolated the WHAT (conc-1 ITL 32K: m8 172.9 /
# m4 112.0 vs q8 42.6 ms; per-layer deltas 8.15/4.34 ms sitting ABOVE
# the known kernel times; m8 TTFT +2.95 s at 8K) but not the WHERE.
# This campaign banks every phase split, on cert25z:
#
#   phase 0  kernel/aux/store/dequant/plan micro-bench — p172_deepprobe
#            in a bare container, loading the INSTALLED shipping
#            kernels (v5 default m8/mkv4, v2 batch m4). A+B+C+D+E
#            faces cover: verify kernel x16 layers (the graph-replay
#            attn share), the index_select/index_copy_/cast aux per
#            layer, the once-per-step plan chain, mx_store prefill
#            write-quant, mx_dequant_cache continuation re-dequant.
#   phase 1-3  e2e legs q8/m8/m4 (champion chain byte-exact r162),
#            per leg conc-1 rungs ctx {1024, 8192, 32768} via
#            p139_conc (TTFT/ITL JSON) AND the serve-log window per
#            rung sliced for spec-metrics acceptance lines — the r161
#            signal (acceptance 0.0% at 32K on e2m1-V lanes) vs the
#            r162 TPS hint (q8 ~1.13 tok/step at 32K looks dead too)
#            must be settled per dtype per ctx on the SHIPPING image.
#
# Measurement campaign, not a gate: cells bank regardless; exit 0 =
# every cell produced its JSON + the probe ran. The FIX loop iterates
# on r171 fastcycle after analysis. W14 quiesce before, trap restore
# after. Artifacts -> /root/build/p647/deepA14/.
set -uo pipefail
R=/root/build
P=${P647_BANK:-$R/p647/deepA14}
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25z}
CNAME=arc-r172
MT=600
PROBE=$R/p139_conc.py  # canonical copy (API length-budget fix, r140c law)
DEEPPROBE=$R/p147/p172_deepprobe.py
[ -f "$PROBE" ] || { echo "R172_NO_PROBE $PROBE"; exit 1; }
[ -f "$DEEPPROBE" ] || { echo "R172_NO_DEEPPROBE $DEEPPROBE"; exit 1; }
mkdir -p "$P"
FAILED=""

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  docker rm -f arc-p172 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r172.log 2>&1 &
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r172:$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[r172:$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r172:$1] ENGINE_DEAD poll=$i"; break; }
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

# ctx conc tag -- one streaming rung + its serve-log acceptance window
group() {
  local CTX=$1 NC=$2 TAG=$3
  local need=$(( NC * (CTX + MT + 64) ))
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN c$CTX n$NC POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    return 0
  fi
  local l0
  l0=$(docker exec $CNAME wc -l < /root/serve_full.log 2>/dev/null || echo 0)
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
  # acceptance window: the serve-log lines this rung produced, spec
  # metrics sliced raw (never parsed at capture time — analysis-side law)
  docker exec $CNAME sh -c \
    "tail -n +$((l0 + 1)) /root/serve_full.log | grep -iE 'acceptance|draft|spec' | tail -40" \
    > "$CD/acceptance_$TAG.txt" 2>/dev/null || true
  echo "CONC_OK_$TAG (acceptance lines: $(wc -l < "$CD/acceptance_$TAG.txt" 2>/dev/null || echo 0))" \
    | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2..=env (champion chain byte-exact r162)
  CELLN=$1; shift
  CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r172:$CELLN] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R172_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r172_$CELLN $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R172_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true; return 0; }
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  {
    docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
      /root/serve_full.log 2>/dev/null | head -3
    echo "mx_backend_lines=$(gcount 'Using MX KV attention backend')"
    echo "lmcache_lines=$(gcount 'LMCacheConnector')"
    echo "tracebacks=$(gcount 'Traceback (most recent call last)')"
  } > "$CD/markers.txt" || true
  cat "$CD/markers.txt"
  POOL=$(kv_pool); echo "POOL=$POOL" | tee -a "$CD/markers.txt"
  # warm the engine (first-hit compiles/pages) before timing rungs
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  group 1024 1  c1024_n1
  group 8192 1  c8192_n1
  group 32768 1 c32768_n1
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "R172_CELL_DONE_$CELLN $(date +%H:%M:%S)"
}

echo "=== [r172:A14] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R172_WARN_PORT8000_STILL_HELD"
sleep 10

# ---- phase 0: kernel/aux/store/dequant/plan micro-bench (bare
# container, INSTALLED shipping kernels; r167 run pattern)
mkdir -p "$P/kbench"
echo "=== [r172:A14] kernel probe in $IMG $(date +%H:%M:%S) ==="
docker rm -f arc-p172 >/dev/null 2>&1 || true
docker run --rm --name arc-p172 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147 -v "$P/kbench":/out \
  --entrypoint /bin/bash \
  $IMG -c "cp /p147/p172_deepprobe.py /tmp/ && cd /tmp && /opt/venv/bin/python p172_deepprobe.py" \
  > "$P/kbench/bench_kernel.txt" 2>&1
KRC=$?
echo "=== [r172:A14] kernel probe exit rc=$KRC $(date +%H:%M:%S) ==="
tail -5 "$P/kbench/bench_kernel.txt"
[ "$KRC" = 0 ] || FAILED="$FAILED kbench"

# ---- phases 1-3: e2e acceptance-rung legs (champion chain r162)
QWEN="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1"
SERVED=qwen3.8-27b-fp8

for CELL in ${CELLS:-q8 m8 m4}; do
  case $CELL in
    q8) run_cell q8 $QWEN ;;
    m8) run_cell m8 $QWEN ARC_KVDTYPE=mxfp8 ;;
    m4) run_cell m4 $QWEN ARC_KVDTYPE=mxfp4 ;;
    *) echo "UNKNOWN_CELL $CELL (q8|m8|m4)"; FAILED="$FAILED $CELL" ;;
  esac
done

# attribution report over the banked JSONs (host python3, stdlib)
python3 $R/p647/p147_telemetry_report.py "$P" \
  > "$P/REPORT.md" 2>&1 || echo "REPORT_NONZERO (bank intact)"
cat "$P/REPORT.md"

cd "$P"
find . -name 'p139_conc_*.json' -o -name 'acceptance_*.txt' -o -name 'markers.txt' \
  -o -name 'posture.txt' -o -name 'bench_kernel.txt' | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
echo "R172_DEEPCAPTURE_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -z "$FAILED" ]
