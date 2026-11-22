#!/bin/bash
# r213b_e2e.sh -- arc7 Phase-8 r213b: the A/A CONFIRMATION of the r213
# e2e truth face (same r212aa wheel 9a01790b, fresh boots). r213 found
# BOTH lanes diverged from the probe prediction (m4 e2e -4.0% at 32K
# while its probe cell regressed +29.6%; m8 e2e +3.8% while its probe
# oracle improved -9.6%) -- per the single-sample law, the signs need a
# second same-state sample before banking conclusions. Gates here are
# DATA-QUALITY only (faces complete, zero errors, native telemetry); the
# report reads r213 vs r213b consistency per face and the combined mean
# vs the r212m4 bank.
# Bank -> /root/build/p647/r213b/{m4qa,m8qa}
set -uo pipefail
R=/root/build
P=$R/p647/r213b
A=$R/p647/r213
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r213b
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R213B_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R213B_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R213B_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R213B_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R213B_STALE_OVERLAY (argument fix not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R213B_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R213B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R213B_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R213B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r213b.log 2>&1 &
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
  echo "=== [r213b:$FT] boot kv=$KV $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KV \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R213B_BOOT_FAIL_$FT"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r213b_$FT $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R213B_PROBE_CP_FAIL_$FT"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r213b:$FT] done $(date +%H:%M:%S) ==="
}

echo "=== [r213b] W14 quiesce $(date +%H:%M:%S) ==="
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

echo "=== [r213b] report $(date +%H:%M:%S) ==="
python3 - "$P" "$A" <<'PY'
import json, os, sys
P, A = sys.argv[1], sys.argv[2]
FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}
PRE = {"m4qa": {8192: 48.9, 16384: 56.7, 32768: 73.1},
       "m8qa": {8192: 50.5, 16384: 60.3, 32768: 81.7}}

def face(base, ft, ctx):
    f = os.path.join(base, ft, f"c{ctx}_n1", f"p139_conc_c{ctx}_n1.json")
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
print("r213b A/A confirmation -- ITL p50 ms/token (same wheel, fresh boots)")
print(f" {'fmt':>5} {'ctx':>6} | {'r213':>7} | {'r213b':>7} | {'spread':>7} | {'mean':>7} | {'r212m4':>7} | {'mean dev':>8} | errs")
for fmt in ("m4qa", "m8qa"):
    for ctx in (8192, 16384, 32768):
        a, ea = face(A, fmt, ctx)
        b, eb = face(P, fmt, ctx)
        if b is None:
            print(f" {fmt:>5} {ctx:>6} | {a if a else 'MISS':>7} | MISSING (errs={eb})")
            ok = False
            continue
        errs = (ea or 0) + (eb or 0)
        if a is None:
            print(f" {fmt:>5} {ctx:>6} | r213 file missing -- cannot compare")
            ok = False
            continue
        spread = 100.0 * abs(b - a) / a
        mean = (a + b) / 2.0
        dev = 100.0 * (mean - PRE[fmt][ctx]) / PRE[fmt][ctx]
        print(f" {fmt:>5} {ctx:>6} | {a:7.1f} | {b:7.1f} | {spread:6.1f}% | {mean:7.1f} | {PRE[fmt][ctx]:7.1f} | {dev:+7.1f}% | {errs}")
        if errs:
            ok = False
        if spread > 6.0:
            print(f"  FLAG_spread_{fmt}_{ctx}: boot-to-boot spread {spread:.1f}%"
                  " exceeds 6% -- face unstable, treat means cautiously")
for fmt in ("m4qa", "m8qa"):
    tf = os.path.join(P, fmt, "telemetry.txt")
    df = os.path.join(P, fmt, "dispatch.txt")
    nt = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    nd = sum(1 for _ in open(df)) if os.path.exists(df) else 0
    print(f" telemetry[{fmt}]: native_call_lines={nt} dispatch_lines={nd}")
    if nt < 50 or nd < 1:
        print(f"  FAIL_telemetry_{fmt}: faces did not ride the native bridge")
        ok = False
print("R213B_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R213B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
