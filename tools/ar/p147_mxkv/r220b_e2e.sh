#!/bin/bash
# r220b_e2e.sh -- arc7 Phase-8 r220b: e2e A/A of the TRUE _64 policy
# (ARC_MXKV_P64NATIVE=1 env route, r220a3 frame laws, wheel 62f234c8) vs the shipping
# _16 sub-tile frame on the SAME wheel, against the banked ns128/auto
# posture. Cells (one boot each, faces c8192/c16384/c32768 n1):
#   p64 : ARC_MXKV_P64NATIVE=1 -- the native 64-row tile
#   ctl : env unset -- same-wheel control (the band edits must be
#         behavior-identical on the shipping _16 route)
# QUALITY CROSS-CHECK (r216b law): per_user_tps + chunks per cell;
# chunks > 80 = INVALID_FOR_SPEED. Telemetry must print splits= on
# both cells (the host policy is unchanged; the kernel maps the splits
# through the 4-SG workgroups).
# Bank -> /root/build/p647/r220b/{p64,ctl}
set -uo pipefail
R=/root/build
P=$R/p647/r220b
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r220b
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R220B_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R220B_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R220B_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
[ -f "$OVERLAY/engine.py" ] || { echo "R220B_NO_OVERLAY"; exit 1; }
grep -q "P64NATIVE" $R/serve_arc_b70.sh \
  || { echo "R220B_SERVE_HOOK_MISSING (P64NATIVE forward)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R220B_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R220B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R220B_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R220B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r220b.log 2>&1 &
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
  local cap=$(( 300 + CTX / 8192 * 90 + 240 ))
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

run_cell() {  # tag p64native
  local CELL=$1 NATIVE=$2
  CD=$P/$CELL
  mkdir -p "$CD"
  local PEN=""
  [ "$NATIVE" = "1" ] && PEN="ARC_MXKV_P64NATIVE=1"
  echo "=== [r220b:$CELL] boot p64=${NATIVE} $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    ${PEN:+$PEN} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R220B_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r220b_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R220B_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r220b:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r220b] W14 quiesce $(date +%H:%M:%S) ==="
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
echo "wheel md5 $GOTMD5"

run_cell p64 1
run_cell ctl 0

echo "=== [r220b] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
POSTURE = {8192: 48.5, 16384: 51.2, 32768: 56.8}
FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}

def cell(name, ctx):
    f = os.path.join(P, name, "c%d_n1" % ctx, "p139_conc_c%d_n1.json" % ctx)
    if not os.path.exists(f):
        return None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    if not rs:
        return None
    itl = sum(r["itl_ms_p50"] for r in rs) / len(rs)
    chunks = sum(r["n_stream_chunks"] for r in rs) / len(rs)
    tps = j.get("per_user_tps")
    tps = tps[0] if isinstance(tps, list) and tps else None
    errs = len([r for r in j["requests"] if r["error"] is not None])
    return itl, chunks, tps, errs

def splits_logged(name):
    tf = os.path.join(P, name, "telemetry.txt")
    if not os.path.exists(tf):
        return None
    vals = set()
    for line in open(tf):
        m = re.search(r"splits=(\d+|auto)", line)
        if m:
            vals.add(m.group(1))
    return sorted(vals) if vals else None

ok = True
print("r220b native p64 tile vs _16 control -- mxfp4 n1")
print(f" {'cell':>4} | {'ctx':>6} | {'itl':>7} | {'vs ctl':>7} | {'vs posture':>10} | {'tps':>6} | {'chunks':>7} | qual | errs")
for name in ("p64", "ctl"):
    for ctx in (8192, 16384, 32768):
        r = cell(name, ctx)
        if r is None:
            print(f" {name:>4} | {ctx:>6} | MISSING")
            ok = False
            continue
        itl, chunks, tps, errs = r
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        ts = "%6.1f" % tps if isinstance(tps, (int, float)) else "   n/a"
        print(f" {name:>4} | {ctx:>6} | {itl:7.1f} |         | {'%+9.2f%%' % ((itl - POSTURE[ctx]) / POSTURE[ctx] * 100.0)} | {ts} | ~{chunks:5.0f} | {qual} | {errs}")
    print(f"   telemetry splits={splits_logged(name)}")
for ctx in (8192, 16384, 32768):
    p, c = cell("p64", ctx), cell("ctl", ctx)
    if p and c:
        d = (p[0] - c[0]) / c[0] * 100.0
        print(f" p64 vs ctl @c{ctx}: {d:+.2f}%")
a = cell("p64", 32768)
if a:
    print(f"  p64 @32K vs fp8: {a[0]/FP8[32768]:.2f}x (posture 1.34x)")
print("R220B_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R220B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
