#!/bin/bash
# r213_e2e.sh -- arc7 Phase-8 r213: the E2E TRUTH FACE on the converged
# r212aa wheel (template-split: shipping body compiles with ZERO probe
# branches). Probe-level deltas vs the r212m4-banked wheel (e50f1c3a) at
# 32K: m0 mxfp4 749.5 -> 971.6 (+29.6%, the attractor tax the split only
# partially recovered) and m8 mxfp8 1320.7 -> 1193.7 (-9.6%, NEW BEST).
# This leg measures what those probe deltas buy END-TO-END: ITL p50 faces
# on the real engine, two boots (one per format), c8192_n1/c16384_n1/
# c32768_n1. Transfer coefficient ~87% (r212m4) says expect m8 ~-8% at
# c32768 and m4 ~+25%; the face decides whether SWAR decode / attractor
# recovery re-prioritizes. Bars (banked, n1 ITL p50 ms/token):
#   fp8    ctrlA24 cert25zc:   40.4 / 41.1 / 42.5
#   mxfp4  r212m4 bank:        48.9 / 56.7 / 73.1
#   mxfp8  r212m4 bank:        50.5 / 60.3 / 81.7
# Wheel is pinned by MD5 (constant wheel name law): the r212aa build
# 9a01790b5748cb4d7e1eac9c099a997e. DIRECTBF16 defaults ON (shipping
# state) -- no env override, e2e measures what would ship.
# Bank -> /root/build/p647/r213/{m4qa,m8qa}
set -uo pipefail
R=/root/build
P=$R/p647/r213
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r213
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R213_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R213_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R213_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R213_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R213_STALE_OVERLAY (argument fix not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R213_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R213_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R213_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R213_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r213.log 2>&1 &
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
  echo "=== [r213:$FT] boot kv=$KV $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KV \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R213_BOOT_FAIL_$FT"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r213_$FT $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R213_PROBE_CP_FAIL_$FT"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r213:$FT] done $(date +%H:%M:%S) ==="
}

echo "=== [r213] W14 quiesce $(date +%H:%M:%S) ==="
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

echo "=== [r213] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# Banked n1 ITL p50 bars (ms/token). fp8 = ctrlA24 cert25zc control;
# pre = the r212m4 bank (the e50f1c3a wheel faces this leg replaces).
FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}
PRE = {"m4qa": {8192: 48.9, 16384: 56.7, 32768: 73.1},
       "m8qa": {8192: 50.5, 16384: 60.3, 32768: 81.7}}
# Probe-oracle deltas at 32K on this wheel vs the m4-banked wheel:
# m0 mxfp4 +29.6% (749.5->971.6, attractor tax), m8 mxfp8 -9.6%
# (1320.7->1193.7, new best). Transfer ~87% -> expect m8 ~-8%, m4 ~+25%.
PROBE_DEV = {"m4qa": 0.296, "m8qa": -0.096}

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
print("r213 e2e truth face -- ITL p50 ms/token (n1 faces, r212aa wheel)")
print(f" {'fmt':>5} {'ctx':>6} | {'r213':>7} | {'r212m4':>7} | {'delta':>7} | {'fp8 bar':>8} | {'vs fp8':>7} | errs")
for fmt in ("m4qa", "m8qa"):
    for ctx in (8192, 16384, 32768):
        new, errs = face(fmt, ctx)
        pre = PRE[fmt][ctx]
        if new is None:
            print(f" {fmt:>5} {ctx:>6} | MISSING (errs={errs})")
            ok = False
            continue
        ds = f"{100.0*(new-pre)/pre:+6.1f}%"
        vs = f"{new/FP8[ctx]:6.2f}x"
        print(f" {fmt:>5} {ctx:>6} | {new:7.1f} | {pre:7.1f} | {ds} | {FP8[ctx]:8.1f} | {vs} | {errs}")
        if errs:
            ok = False
        if ctx == 32768 and fmt == "m8qa" and new > pre * 1.05:
            print(f"  FAIL_m8qa_{ctx}: mxfp8 regressed >5% vs the r212m4 bank"
                  " (probe oracle said -9.6% -- transfer broke)"); ok = False
        if ctx == 32768 and fmt == "m4qa":
            dev = (new - pre) / pre
            if dev > 0.35:
                print(f"  FLAG_m4qa_{ctx}: regression {dev*100:+.1f}% exceeds the"
                      " probe-predicted +29.6% band (transfer >100% -- look for"
                      " a second mover)")
            elif dev < 0.0:
                print(f"  FLAG_m4qa_{ctx}: mxfp4 IMPROVED e2e while the probe cell"
                      " regressed +29.6% -- probe/e2e divergence, investigate")
# Telemetry presence: the dispatch posture + native-call lines are the
# proof the faces actually rode the native bridge on THIS wheel.
for fmt in ("m4qa", "m8qa"):
    tf = os.path.join(P, fmt, "telemetry.txt")
    df = os.path.join(P, fmt, "dispatch.txt")
    nt = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    nd = sum(1 for _ in open(df)) if os.path.exists(df) else 0
    print(f" telemetry[{fmt}]: native_call_lines={nt} dispatch_lines={nd}")
    if nt < 50 or nd < 1:
        print(f"  FAIL_telemetry_{fmt}: faces did not ride the native bridge")
        ok = False
print("R213_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R213_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
