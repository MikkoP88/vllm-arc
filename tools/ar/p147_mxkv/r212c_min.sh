#!/bin/bash
# r212c_min.sh -- arc7 Phase-8 r212b: same-stack lane isolator after the
# launch4 verdict (numerics PASS; dispatch froze native on EVERY
# captured bucket because the estimator prices bt CAPACITY -- width
# 4096 = max_model_len/block, constant across buckets -- so walked_cap
# 38.8M > every threshold; native-frozen ran 3.0-8.6x the control at
# matched faces and ~100x the offline per-call map at matched walked
# rows). Two boots, m4qa only, everything else identical (build21
# wheel + overlay + image):
#   A  ARC_MXKV_NATIVE_MINLEN=mxfp4:999999999 -- walked_cap < minlen
#      on every bucket -> ALL-TRITON on the treatment stack (replaces
#      the stale r212a_ctl control, whose 4K->32K slope sat ~4x BELOW
#      the bandwidth floor)
#   B  default thresholds -- native-frozen (launch4 posture)
# Faces c16384 n1 / c32768 n1 / c16384 n8 (the anomaly cells).
# Also banks the r212b capture telemetry (native_decode.py one line
# per (fmt, rows, max_seqlen_k, bt_width) class) -- the frozen walk
# extent is the prime suspect for the 100x.
# Bank -> /root/build/p647/r212c/{A_triton,B_native}
set -uo pipefail
R=/root/build
P=$R/p647/r212c
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212b
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R212C_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R212C_NO_WHEEL"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212C_NO_OVERLAY"; exit 1; }
grep -q "r212c-e2e capacity-priced walk" "$OVERLAY/native_decode.py" \
  || { echo "R212C_STALE_OVERLAY (telemetry edit not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212C_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212C_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212C_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212C_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212b.log 2>&1 &
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

boot_wait() {
  local i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[$1] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb
      tb=$(docker exec $CNAME grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null)
      [ "${tb:-0}" -gt 0 ] && { echo "[$1] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[$1] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

group() {  # ctx conc -- streaming cell (lifted from r212a_fastcycle)
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  mkdir -p "$GD"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" qwen3.8-27b-fp8 0.0
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
    echo "CONC_TIMEOUT_$TAG (cap=${cap}s)"; echo "TIMEOUT" > "$GD/outcome.txt"; return 0
  fi
  docker cp $CNAME:/root/p139_conc_$TAG.json "$GD/" >/dev/null 2>&1
  echo "CONC_OK_$TAG"
}

run_side() {  # side minlen
  SIDE=$1; MINLEN=${2:-}
  CD=$P/$SIDE
  mkdir -p "$CD"
  echo "=== [r212c:$SIDE] boot $(date +%H:%M:%S) minlen='${MINLEN:-default}' ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ${MINLEN:+ARC_MXKV_NATIVE_MINLEN=$MINLEN} \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$SIDE"; then
    echo "R212C_BOOT_FAIL_$SIDE"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r212b_$SIDE $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R212C_PROBE_CP_FAIL_$SIDE"; docker rm -f $CNAME >/dev/null 2>&1; \
    verify_removed || true; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  group 16384 1
  group 32768 1
  group 16384 8
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV verify decoder" /root/serve_full.log 2>/dev/null \
    > "$CD/decoder.txt" || true
  docker exec $CNAME grep "MXKV native dispatch" /root/serve_full.log 2>/dev/null \
    > "$CD/dispatch.txt" || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r212c:$SIDE] done $(date +%H:%M:%S) ==="
}

echo "=== [r212b] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

mkdir -p "$P"
md5sum "$OVERLAY"/*.py > "$P/overlay_md5.txt" 2>/dev/null || true
sha256sum "$WHEELDIR/$WHEELNAME" > "$P/wheel_sha256.txt"

# A side banked in r212b (same stack, triton numbers citable)
run_side B_native ""

echo "=== [r212b] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, sys, os
p = sys.argv[1]
print("r212b A/B -- ITL p50 ms/token (itl_ms_p50 mean over slots)")
for side in ("A_triton", "B_native"):
    d = os.path.join(p, side)
    for tag in ("c16384_n1", "c32768_n1", "c16384_n8"):
        f = os.path.join(d, tag, f"p139_conc_{tag}.json")
        if not os.path.exists(f):
            print(f" {side:9s} {tag:10s} MISSING")
            continue
        j = json.load(open(f))
        itls = [r["itl_ms_p50"] for r in j["requests"] if r["error"] is None]
        accs = [r for r in j["requests"] if r["error"] is None]
        itl = sum(itls) / max(1, len(itls))
        print(f" {side:9s} {tag:10s} itl {itl:8.1f} ms  n_ok {j['n_ok']}")
PY
echo "=== [r212b] telemetry (B) ==="
cat "$P/B_native/telemetry.txt" 2>/dev/null | head -8
echo "=== [r212b] dispatch (A/B) ==="
head -3 "$P/A_triton/dispatch.txt" 2>/dev/null
head -3 "$P/B_native/dispatch.txt" 2>/dev/null
echo "R212C_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
