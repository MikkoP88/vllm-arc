#!/bin/bash
# r185_acceptknee.sh -- P6.47 S1 rung 1: the e2m1-V acceptance ctx-knee
# map on cert25zb. r161 (cert25y, PRE the r168 LMC auto-off fix) showed
# MTP verify acceptance exactly 0.0% in all four 32K longprobe windows
# on m4q/mkq while short-ctx held 50-80%; the current curve must be
# re-established on the shipping image before the adaptive ARC_SPEC
# off-switch is designed. Three qwen boots (m4q, mkq, m8q control —
# the e4m3/e4m3 lane that never died), each cell = one p139_conc
# streaming group bracketed by /metrics snapshots (spec_decode counter
# deltas -> acceptance% per window). Rungs {4K,8K,16K,32K} x {n1,n8}
# (the plan's S1 curve). Report: p181_acceptknee.py over the bank.
#
# w14_restore carries the arc-in-flight guard. Artifacts ->
# /root/build/p647/acceptA23/. Exit 0 = every cell banked with a valid
# window (ACCEPT_KNEE_PASS).
set -uo pipefail
R=/root/build
P=$R/p647/acceptA23
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
CNAME=arc-r185
PROBE=$R/p139_conc.py   # canonical copy (API length-budget fix, r140c law)
REPORT=$R/p147/p181_acceptknee.py
FAILED=""

[ -f "$PROBE" ] || { echo "R185_NO_PROBE $PROBE"; exit 1; }
[ -f "$REPORT" ] || { echo "R185_NO_REPORT $REPORT -- ship first"; exit 1; }
grep -q "p181_acceptknee" "$REPORT" \
  || { echo "R185_STALE_TREE (report identity marker missing)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "R185_NO_IMAGE $IMG"; exit 1; }
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r185.log 2>&1 &
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb; tb=$(gcount "Traceback (most recent call last)")
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
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

group() {  # ctx conc -- metrics-bracketed streaming cell -> $CD/c{ctx}_n{nc}/
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  local need=$(( NC * (CTX + MT + 64) ))
  mkdir -p "$GD"
  if [ "$POOL" -gt 0 ] && [ "$need" -gt "$POOL" ]; then
    echo "KV_CAP $CELLN $TAG POOL=$POOL NEED=$need" | tee -a "$CD/conc.log"
    echo "CAP" > "$GD/outcome.txt"; return 0
  fi
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" "$SERVED" 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 + (NC == 8 ? 240 : 0) ))
  local i=0
  while [ $i -lt $((cap / 4)) ]; do
    docker exec $CNAME test -f /root/p139_conc_$TAG.json && break
    docker inspect $CNAME >/dev/null 2>&1 || break
    sleep 4; i=$((i+1))
  done
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_after.txt" || true
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)" | tee -a "$CD/conc.log"
    echo "TIMEOUT" > "$GD/outcome.txt"; return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$GD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  # $1=cell $2=kvdtype (qwen champion chain byte-exact from r184)
  CELLN=$1; SERVED=qwen3.8-27b-fp8; CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r185:$CELLN] boot $(date +%H:%M:%S) ARC_KVDTYPE=$2 ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$2 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R185_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r185_$CELLN $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R185_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true; return 0; }
  docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
    /root/serve_full.log 2>/dev/null | head -3 > "$CD/markers.txt" || true
  POOL=$(kv_pool); echo "POOL=$POOL" | tee -a "$CD/markers.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  group  4096 1; group  4096 8
  group  8192 1; group  8192 8
  group 16384 1; group 16384 8
  group 32768 1; group 32768 8
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r185:$CELLN] done $(date +%H:%M:%S) ==="
}

echo "=== [r185:acceptknee] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R185_WARN_PORT8000_STILL_HELD"
sleep 5

# order: the lanes under test first (time-box protects the primary),
# the never-died e4m3 control last
run_cell m4q mxfp4
run_cell mkq mxk8v4
run_cell m8q mxfp8

echo "=== [r185:acceptknee] report $(date +%H:%M:%S) ==="
python3 "$REPORT" "$P" 2>&1 | tee "$P/acceptknee.txt"
RC=${PIPESTATUS[0]}
echo "=== [r185:acceptknee] report rc=$RC $(date +%H:%M:%S) ==="
grep -q "ACCEPT_KNEE_PASS" "$P/acceptknee.txt" || RC=1
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.json" -o -name "serve.log" \) -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
[ -z "$FAILED" ] || echo "R185_FAILED_CELLS:$FAILED"
echo "R185_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
