#!/bin/bash
# r196b_fastcycle.sh -- P6.47 cert25zc validation fastcycle: the three MX
# chains on the NEW image with NO env deltas -- the image defaults carry
# both closing-rung changes (TILE_FLOOR launcher default 2; G-adaptive
# mxfp4 dispatch, ARC_MXKV_V5_G_MIN=8). m4q is the G-adaptive proof cell
# (n1 must stay on v2 = r185b baseline byte-class; n8 must flip to v5 =
# the r188-B1 win, 32K ss 1978->~1700); m8q/mkq must reproduce the r195
# tf2 numbers (the knob now baked in). The one-time decoder dispatch log
# line is banked per cell (decoder.txt) -- m4q must show BOTH (mxfp4,v2)
# and (mxfp4,v5) source=g-adaptive lines. Quality gates:
# ACCEPT_KNEE_PASS per cell. Bank zcA27/{m4qzc,m8qzc,mkqzc}; report ->
# zc.txt. Manual leg -- fires only when the rig is clear.
set -uo pipefail
R=/root/build
P=$R/p647/zcA27
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r196b
PROBE=$R/p139_conc.py
REPORT=$R/p147/p181_acceptknee.py
FAILED=""

[ -f "$PROBE" ] || { echo "R196B_NO_PROBE"; exit 1; }
[ -f "$REPORT" ] || { echo "R196B_NO_REPORT -- ship first"; exit 1; }
grep -q "p181_acceptknee" "$REPORT" || { echo "R196B_STALE_TREE"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R196B_NO_IMAGE $IMG"; exit 1; }
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -q "cert25zc_fork_sha=dd74715079dc6c23fd4ad2feb1a700294515d423" /etc/vllm-arc/IMAGE_MANIFEST' \
  || { echo "R196B_WRONG_IMAGE_CONTENT (fork_sha mismatch)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R196B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R196B_PORT_HELD (unknown holder)"; exit 3
  fi
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
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r196b.log 2>&1 &
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

run_cell() {  # cellname kvdtype
  CELLN=$1; KVD=$2; SERVED=qwen3.8-27b-fp8; CD=$P/$CELLN
  mkdir -p "$CD"
  echo "=== [r196b:$CELLN] boot $(date +%H:%M:%S) ARC_KVDTYPE=$KVD image-defaults (tf2 + G-adaptive) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KVD \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R196B_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r196b_$CELLN $(date +%H:%M:%S)"
  docker exec $CNAME grep "MXKV verify decoder" /root/serve_full.log 2>/dev/null \
    > "$CD/decoder.txt" || true
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R196B_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
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
  docker exec $CNAME grep "MXKV verify decoder" /root/serve_full.log 2>/dev/null \
    > "$CD/decoder.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r196b:$CELLN] done $(date +%H:%M:%S) ==="
}

echo "=== [r196b] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell m4qzc mxfp4
run_cell m8qzc mxfp8
run_cell mkqzc mxk8v4

echo "=== [r196b] report $(date +%H:%M:%S) ==="
python3 "$REPORT" "$P" m4qzc,m8qzc,mkqzc 2>&1 | tee "$P/zc.txt"
RC=${PIPESTATUS[0]}
echo "=== [r196b] report rc=$RC $(date +%H:%M:%S) ==="
grep -q "ACCEPT_KNEE_PASS" "$P/zc.txt" || RC=1
# G-adaptive proof gate: the m4q cell must show BOTH decoder faces
if ! grep -q "decoder=v5" "$P/m4qzc/decoder.txt" 2>/dev/null \
   || ! grep -q "decoder=v2" "$P/m4qzc/decoder.txt" 2>/dev/null; then
  echo "R196B_GADAPTIVE_NOT_OBSERVED (m4q decoder.txt lacks a v2/v5 face)"; RC=1
fi
[ -z "$FAILED" ] || echo "R196B_FAILED_CELLS:$FAILED"
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.json" -o -name "serve.log" -o -name "serve_snapshot.log" \) -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
echo "R196B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
