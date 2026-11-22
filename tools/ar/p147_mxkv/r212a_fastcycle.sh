#!/bin/bash
# r212a_fastcycle.sh -- arc7 Phase-8 r212a: FIRST native e2e with the
# ctx-adaptive dispatch (ARC_MXKV_NATIVE=1 + per-lane E5b thresholds in
# the overlay engine.py, BUCKET-CAP estimator: rows x bt_width x block
# stride -- the r212a_ctl control proved metadata.max_seq_len freezes
# triton on every captured graph via the diluted dummy, so the branch
# now rides the bucket token class the runtime step maps to by
# sum(num_blocks)). The cert25zc image predates the bridge, so serve
# boots with the mx_kv overlay mount + arc7mx wheel pre-install
# (serve_arc_b70.sh inert hooks, banked serve_patch.txt).
# Full r196b grid (1024 n1 + 4096/8192/16384/32768 x n1/n8) per format:
# long-ctx n1 + the n8 faces push walked_cap over every per-lane
# threshold (mxfp8's 262144 = n8-8K exactly), so each format must show
# BOTH decoder=native and its certified triton lane. Gates: p181
# ACCEPT_KNEE_PASS (full grid) + per-format side checks + zero
# tracebacks. Rates A/B vs r212a_ctl (dispatch-inert control) and
# zcA27 (certified) in the bank write-up.
# Bank -> /root/build/p647/r212a/{m4qa,m8qa,mkqa}
set -uo pipefail
R=/root/build
P=$R/p647/r212a
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212a
PROBE=$R/p139_conc.py
REPORT=$R/p147/p181_acceptknee.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv
FAILED=""

[ -f "$PROBE" ] || { echo "R212A_NO_PROBE"; exit 1; }
[ -f "$REPORT" ] || { echo "R212A_NO_REPORT -- ship first"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R212A_NO_WHEEL"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212A_NO_OVERLAY"; exit 1; }
grep -q "ARC_MXKV_NATIVE:" /root/build/serve_arc_b70.sh \
  || { echo "R212A_SERVE_UNPATCHED (run p212a_patch_serve.py first)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212A_NO_IMAGE $IMG"; exit 1; }
docker run --rm --entrypoint /bin/bash "$IMG" -c \
  'grep -q "cert25zc_fork_sha=dd74715079dc6c23fd4ad2feb1a700294515d423" /etc/vllm-arc/IMAGE_MANIFEST' \
  || { echo "R212A_WRONG_IMAGE_CONTENT (fork_sha mismatch)"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212A_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212A_PORT_HELD (unknown holder)"; exit 3
  fi
fi

mkdir -p "$P"
# provenance: the exact serve patch + overlay bytes this leg rides
diff /root/build/serve_arc_b70.sh.pre_r212a /root/build/serve_arc_b70.sh \
  > "$P/serve_patch.txt" 2>/dev/null || true
md5sum "$OVERLAY"/*.py > "$P/overlay_md5.txt" 2>/dev/null || true

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212A_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212a.log 2>&1 &
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

group() {  # ctx conc -- metrics-bracketed streaming cell
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
  echo "=== [r212a:$CELLN] boot $(date +%H:%M:%S) KVDTYPE=$KVD NATIVE=1 dispatch=overlay ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KVD \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$CELLN"; then
    echo "R212A_BOOT_FAIL_$CELLN -- banking log, continuing"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    FAILED="$FAILED $CELLN"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r212a_$CELLN $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R212A_PROBE_CP_FAIL_$CELLN"; FAILED="$FAILED $CELLN"; \
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true; return 0; }
  docker exec $CNAME grep -E "GPU KV cache size|Maximum concurrency" \
    /root/serve_full.log 2>/dev/null | head -3 > "$CD/markers.txt" || true
  POOL=$(kv_pool); echo "POOL=$POOL" | tee -a "$CD/markers.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  group  1024 1
  group  4096 1; group  4096 8
  group  8192 1; group  8192 8
  group 16384 1; group 16384 8
  group 32768 1; group 32768 8
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV verify decoder" /root/serve_full.log 2>/dev/null \
    > "$CD/decoder.txt" || true
  docker exec $CNAME grep "MXKV native dispatch" /root/serve_full.log 2>/dev/null \
    > "$CD/dispatch.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r212a:$CELLN] done $(date +%H:%M:%S) ==="
}

echo "=== [r212a] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell m4qa mxfp4
run_cell m8qa mxfp8
run_cell mkqa mxk8v4

echo "=== [r212a] report $(date +%H:%M:%S) ==="
python3 "$REPORT" "$P" m4qa,m8qa,mkqa 2>&1 | tee "$P/r212a.txt"
RC=${PIPESTATUS[0]}
echo "=== [r212a] report rc=$RC $(date +%H:%M:%S) ==="
grep -q "ACCEPT_KNEE_PASS" "$P/r212a.txt" || RC=1
# dispatch-side structural gates: the bucket-cap estimator must freeze
# native for the long-ctx/n8 bucket classes in every format (the n8
# faces push walked_cap over mxfp8's 262144 too), and the certified
# triton lanes must still appear (short/warmup classes).
for C in m4qa m8qa mkqa; do
  if ! grep -q "decoder=native" "$P/$C/decoder.txt" 2>/dev/null; then
    echo "R212A_${C^^}_NATIVE_NOT_OBSERVED"; RC=1
  fi
  if ! grep -qE "decoder=v[25]" "$P/$C/decoder.txt" 2>/dev/null; then
    echo "R212A_${C^^}_TRITON_NOT_OBSERVED"; RC=1
  fi
done
[ -z "$FAILED" ] || echo "R212A_FAILED_CELLS:$FAILED"
cd "$P"
find . -type f \( -name "*.txt" -o -name "*.json" -o -name "serve.log" -o -name "serve_snapshot.log" -o -name "boot.txt" \) -exec sha256sum {} \; > SHA256SUMS 2>/dev/null || true
echo "R212A_DONE rc=$RC $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $RC
