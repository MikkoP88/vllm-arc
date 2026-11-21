#!/bin/bash
# r212m4_e2e.sh -- arc7 Phase-8 r212m4: the CONVERGENCE CHECKPOINT of the
# r212m3 wheel (e2m1-gated direct-bf16). Offline the wheel moved mxfp4
# -27/-32% (399.8/306.8/749.3 us at 1K/8K/32K) with both e2m1 decode
# lanes nearly halved, mxfp8 -3% (byte-equivalent float path); this leg
# measures what that buys END-TO-END: ITL p50 faces on the real engine,
# two boots (one per format), faces c8192_n1 / c16384_n1 / c32768_n1.
# Bars (banked, n1 ITL p50 ms/token):
#   fp8   ctrlA24 cert25zc:  40.4 / 41.1 / 42.5
#   mxfp4 pre  (r212g-class): 57.4 / 74.1 / (c32768 was unbanked)
#   mxfp8 pre  (r212h):       50.9 / 61.4 / 84.3
# Wheel is pinned by MD5 (constant wheel name law): the r212m3 build
# e50f1c3aa38729931c0d6ccc9c57a370. DIRECTBF16 defaults ON (shipping
# state) -- no env override, e2e measures what would ship.
# Bank -> /root/build/p647/r212m4/{m4qa,m8qa}
set -uo pipefail
R=/root/build
P=$R/p647/r212m4
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212m4
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=e50f1c3aa38729931c0d6ccc9c57a370
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R212M4_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R212M4_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R212M4_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212M4_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R212M4_STALE_OVERLAY (argument fix not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212M4_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212M4_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212M4_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212M4_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212m4.log 2>&1 &
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
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[boot] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb
      tb=$(docker exec $CNAME grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null)
      [ "${tb:-0}" -gt 0 ] && { echo "[boot] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[boot] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

group() {  # ctx conc
  local CTX=$1 NC=$2
  local TAG="c${1}_n${2}"
  local GD="$CD/$TAG"
  local MT; [ "$NC" = 1 ] && MT=192 || MT=128
  mkdir -p "$GD"
  curl -s -m 10 http://127.0.0.1:8000/metrics > "$GD/metrics_before.txt" || true
  docker exec $CNAME rm -f /root/p139_conc_$TAG.json
  docker exec -d $CNAME /opt/venv/bin/python /root/p139_conc.py \
    "$CTX" "$MT" "$NC" "$TAG" qwen3.8-27b-fp8 0.0
  local cap=$(( 300 + CTX / 8192 * 90 + 120 ))
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

run_fmt() {  # kvdtype tag
  local KV=$1 FT=$2
  CD=$P/$FT
  mkdir -p "$CD"
  echo "=== [r212m4:$FT] boot kv=$KV $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KV \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R212M4_BOOT_FAIL_$FT"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r212m4_$FT $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R212M4_PROBE_CP_FAIL_$FT"; docker rm -f $CNAME >/dev/null 2>&1; \
    verify_removed || true; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  group 8192 1
  group 16384 1
  group 32768 1
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME grep "MXKV native dispatch" /root/serve_full.log 2>/dev/null \
    > "$CD/dispatch.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r212m4:$FT] done $(date +%H:%M:%S) ==="
}

echo "=== [r212m4] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

mkdir -p "$P"
md5sum "$OVERLAY"/*.py > "$P/overlay_md5.txt" 2>/dev/null || true
md5sum "$WHEELDIR/$WHEELNAME" > "$P/wheel_md5.txt"
echo "wheel md5 $GOTMD5 (pinned $WHEELMD5)"

run_fmt mxfp4 m4qa
run_fmt mxfp8 m8qa

echo "=== [r212m4] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# Banked n1 ITL p50 bars (ms/token). fp8 = ctrlA24 cert25zc control;
# pre = the pre-r212m3-wheel MX faces (r212g-class m4, r212h m8).
FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}
PRE = {"m4qa": {8192: 57.4, 16384: 74.1, 32768: None},
       "m8qa": {8192: 50.9, 16384: 61.4, 32768: 84.3}}

def face(ft, ctx):
    f = os.path.join(P, ft, f"c{ctx}_n1", f"p139_conc_c{ctx}_n1.json")
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    errs = len([r for r in j["requests"] if r["error"] is not None])
    if not rs:
        return None, errs
    itls = [r["itl_ms_p50"] for r in rs]
    return (sum(itls) / len(itls)), errs

ok = True
print("r212m4 convergence checkpoint -- ITL p50 ms/token (n1 faces, r212m3 wheel)")
print(f" {'fmt':>5} {'ctx':>6} | {'r212m4':>7} | {'pre':>7} | {'delta':>7} | {'fp8 bar':>8} | {'vs fp8':>7} | errs")
for fmt in ("m4qa", "m8qa"):
    for ctx in (8192, 16384, 32768):
        new, errs = face(fmt, ctx)
        pre = PRE[fmt][ctx]
        if new is None:
            print(f" {fmt:>5} {ctx:>6} | MISSING (errs={errs})")
            ok = False
            continue
        ds = f"{100.0*(new-pre)/pre:+6.1f}%" if pre else "  n/a  "
        vs = f"{new/FP8[ctx]:6.2f}x" if FP8[ctx] else ""
        print(f" {fmt:>5} {ctx:>6} | {new:7.1f} | {pre if pre else '   n/a'} | {ds} | {FP8[ctx]:8.1f} | {vs} | {errs}")
        if errs:
            ok = False
        if fmt == "m4qa" and pre and new >= pre:
            print(f"  FAIL_m4qa_{ctx}: mxfp4 did not improve vs pre face"); ok = False
        if fmt == "m8qa" and pre and new > pre * 1.10:
            print(f"  FAIL_m8qa_{ctx}: mxfp8 regressed >10% vs pre face"); ok = False
print("R212M4_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R212M4_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
