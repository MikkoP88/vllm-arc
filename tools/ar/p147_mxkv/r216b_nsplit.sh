#!/bin/bash
# r216b_nsplit.sh -- arc7 Phase-8 r216b: the ns64 A/A CONFIRMATION +
# deeper probe (ns96) + multi-ctx posture faces. r216 found ns64 =
# 62.2 = -11.45 pct vs the 70.2 reference (anchor ns40 +0.85 pct,
# knob live) -- the biggest single e2e win of the arc7 MX line, and a
# THIRD instance of the sign-inversion class (the r209/r210 "reduce
# cap 40 optimal" was probe-priced). Per the r213 law the sign needs a
# fresh-boot A/A before banking, the posture must not regress the
# other faces (48.5/55.9 at 8K/16K), and the sweep stopped AT the
# heuristic's 64 cap without hitting any kernel limit -- so ns96
# probes deeper (the r208 fork-owned reduce handles deep S).
#   boot1 ns64b: faces c8192/c16384/c32768 -> A/A at 32K + regression
#                check at 8K/16K
#   boot2 ns96 : same faces -> deeper-split posture candidate
# Bank -> /root/build/p647/r216b/{ns64b,ns96}
set -uo pipefail
R=/root/build
P=$R/p647/r216b
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r216b
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R216B_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R216B_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R216B_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R216B_NO_OVERLAY"; exit 1; }
grep -q "num_splits_kv=_ns" "$OVERLAY/native_decode.py" \
  || { echo "R216B_OVERLAY_HOOK_MISSING (NSPLIT pin not shipped)"; exit 1; }
grep -q "ARC_MXKV_NSPLIT" $R/serve_arc_b70.sh \
  || { echo "R216B_SERVE_HOOK_MISSING (passthrough not inserted)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R216B_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R216B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R216B_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R216B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r216b.log 2>&1 &
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

run_cell() {  # tag nsplit
  local CELL=$1 NS=$2
  CD=$P/$CELL
  mkdir -p "$CD"
  echo "=== [r216b:$CELL] boot ARC_MXKV_NSPLIT=$NS $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    ARC_MXKV_NSPLIT=$NS \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R216B_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r216b_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R216B_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r216b:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r216b] W14 quiesce $(date +%H:%M:%S) ==="
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

run_cell ns64b 64
run_cell ns96 96

echo "=== [r216b] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
# r213 A/A-confirmed auto-split means (posture baseline) + r216 ns64.
M4 = {8192: 48.5, 16384: 55.9, 32768: 70.2}
NS64 = {32768: 62.2}  # r216 single boot (this leg's A/A subject)

def cell(name, ctx):
    f = os.path.join(P, name, f"c{ctx}_n1", f"p139_conc_c{ctx}_n1.json")
    if not os.path.exists(f):
        return None, None, None, None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    errs = len([r for r in j["requests"] if r["error"] is not None])
    if not rs:
        return None, errs, None, None
    itl = sum(r["itl_ms_p50"] for r in rs) / len(rs)
    chunks = sum(r["n_stream_chunks"] for r in rs) / len(rs)
    rows = None
    df = os.path.join(P, name, "dispatch.txt")
    if os.path.exists(df):
        m = re.search(r"rows=(\d+)", open(df).read())
        rows = int(m.group(1)) if m else None
    return itl, errs, chunks, rows

ok = True
vals = {}
print("r216b ns64 A/A + ns96 deeper probe -- mxfp4 n1 (r212aa wheel)")
print(f" {'cell':>5} | {'ctx':>6} | {'itl':>7} | {'vs auto':>8} | {'chunks':>7} | errs")
for name in ("ns64b", "ns96"):
    for ctx in (8192, 16384, 32768):
        itl, errs, chunks, rows = cell(name, ctx)
        if itl is None:
            print(f" {name:>5} | {ctx:>6} | MISSING (errs={errs})")
            ok = False
            continue
        vals[(name, ctx)] = itl
        dev = (itl - M4[ctx]) / M4[ctx] * 100.0
        print(f" {name:>5} | {ctx:>6} | {itl:7.1f} | {dev:+7.2f}% | ~{chunks:5.0f} | {errs}"
              + (f" rows={rows}" if rows not in (None, 148) else ""))
        if errs:
            ok = False
if (32768,) and ("ns64b", 32768) in vals and 32768 in NS64:
    a = vals[("ns64b", 32768)]
    spread = abs(a - NS64[32768]) / NS64[32768] * 100.0
    if spread <= 2.0:
        print(f" AA_OK ns64 @32K: {NS64[32768]} -> {a} (spread {spread:.2f}%)"
              " -- the -11 pct sign is CONFIRMED")
    else:
        print(f" AA_FLAG ns64 @32K: {NS64[32768]} -> {a} (spread {spread:.2f}%)"
              " -- boot variance exceeds the delta, investigate")
        ok = False
for ctx in (8192, 16384, 32768):
    a = vals.get(("ns64b", ctx))
    if a is not None:
        d = (a - M4[ctx]) / M4[ctx] * 100.0
        tag = "REGRESSION" if d > 2.0 else "ok"
        print(f"  ns64 posture @ {ctx}: {a} vs auto {M4[ctx]} ({d:+.2f}%) [{tag}]")
if ("ns96", 32768) in vals and ("ns64b", 32768) in vals:
    d = (vals[("ns96", 32768)] - vals[("ns64b", 32768)]) / vals[("ns64b", 32768)] * 100.0
    print(f"  ns96 vs ns64 @32K: {d:+.2f}%"
          + (" -- DEEPER WINS, keep pushing" if d < -1.5
             else " -- ns64 region is the knee (or past it)"))
print("R216B_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R216B_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
