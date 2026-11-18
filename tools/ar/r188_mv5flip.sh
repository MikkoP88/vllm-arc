#!/bin/bash
# r188_mv5flip.sh -- P6.47 decode-parity fast-cycle 1 (2026-11-14): flip the MTP VERIFY decoder to v5 dot on the mxfp4 lane
# (ARC_MXKV_VERIFY_DECODER=v5 -- new serve_arc_b70.sh pass-through) and
# re-run the r185 acceptance ladder cell-for-cell. Evidence chain: r185
# closed the r161 e2m1-V quality investigation (KNEE=none on cert25zb,
# acceptance healthy through 32K -- the r168 LMC auto-off fix was the
# cure), voiding engine.py's v2-freeze rationale for mxfp4; the A13
# "v5 is a 0.94-1.05x wash on mxfp4" verdict PREDATES K3 (r179 k3cycleA18:
# v5+K3 m4 kernel 7.618 ms vs v2's ~8.2-9.2 = 1.1-1.2x win). This leg is
# the A/B: identical chain to r185b's mkq cell except the decoder knob.
# Quality gates: v5's offline d<=2e-3 (banked A13) + this ladder's
# ACCEPT_KNEE_PASS (acceptance must stay r185-level through 32K).
# Bank flipA25/m4qv5; report p181 cells=m4qv5 -> flipv5.txt. Exit 0 =
# ACCEPT_KNEE_PASS. Manual leg -- fires only after the r187 chain lands.
set -uo pipefail
R=/root/build
P=$R/p647/flipA25
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
CNAME=arc-r188
PROBE=$R/p139_conc.py
REPORT=$R/p147/p181_acceptknee.py
FAILED=""

[ -f "$PROBE" ] || { echo "R188_NO_PROBE"; exit 1; }
[ -f "$REPORT" ] || { echo "R188_NO_REPORT -- ship first"; exit 1; }
grep -q "p181_acceptknee" "$REPORT" || { echo "R188_STALE_TREE"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R188_NO_IMAGE $IMG"; exit 1; }
grep -q "ARC_MXKV_VERIFY_DECODER" "$R/serve_arc_b70.sh" \
  || { echo "R188_NO_PASSTHROUGH -- serve_arc_b70.sh lacks the MXKV slot"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R188_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  echo "R188_PORT_HELD"; exit 3
fi
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (the actor removed radiance-test 2026-11-13;
  # docker start + serve_only can never restore it -- see r185b/r186)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r188.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
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

group() {  # ctx conc -- metrics-bracketed streaming cell; serve.log snapshot per group
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
  docker cp $CNAME:/root/serve_full.log "$GD/serve_snapshot.log" >/dev/null 2>&1 || true
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_after.txt" || true
  if ! docker exec $CNAME test -f /root/p139_conc_$TAG.json 2>/dev/null; then
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)" | tee -a "$CD/conc.log"
    echo "TIMEOUT" > "$GD/outcome.txt"; return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$GD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG" | tee -a "$CD/conc.log"
}

run_cell() {
  CELLN=m4qv5; SERVED=qwen3.8-27b-fp8; CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r188:$CELLN] boot $(date +%H:%M:%S) ARC_KVDTYPE=mxfp4 VERIFY_DECODER=v5 ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_VERIFY_DECODER=v5 \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R188_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r188_$CELLN $(date +%H:%M:%S)"
  docker exec $CNAME grep -c "ARC_MXKV_VERIFY_DECODER\|verify decoder" \
    /root/serve_full.log 2>/dev/null || true
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R188_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
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
  echo "=== [r188:$CELLN] done $(date +%H:%M:%S) ==="
}

echo "=== [r188] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell

echo "=== [r188] report $(date +%H:%M:%S) ==="
python3 "$REPORT" "$P" m4qv5 2>&1 | tee "$P/flipv5.txt"
RC=${PIPESTATUS[0]}
echo "=== [r188] report rc=$RC $(date +%H:%M:%S) ==="
grep -q "ACCEPT_KNEE_PASS" "$P/flipv5.txt" || RC=1
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.json" -o -name "serve.log" -o -name "serve_snapshot.log" \) -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
[ -z "$FAILED" ] || echo "R188_FAILED_CELLS:$FAILED"
echo "R188_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
