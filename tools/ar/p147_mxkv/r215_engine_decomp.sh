#!/bin/bash
# r215_engine_decomp.sh -- arc7 Phase-8 r215: the ENGINE-SHAPE LANE
# DECOMPOSITION on the converged r212aa wheel (9a01790b), mxfp4 lane,
# c32768 n1 -- the r213-lawful replacement for the offline probe
# decomposition (probe/e2e sign inversion: offline lanes are
# shape-scoped; only engine faces arbitrate). The C++ config run()
# reads ARC_MXKV_VPROBE + ARC_MXKV_DIRECTBF16 from the env, and
# serve_arc_b70.sh now passes both through (r215 inert additions) --
# zero code changes. Cells (one boot each):
#   dbf0 : ARC_MXKV_DIRECTBF16=0 -- full float decode tail (CORRECT
#          output; acceptance normal); tail pool = dbf0 - full
#   v6   : ARC_MXKV_VPROBE=6  -- K chain severed (const-fill): V+P+base
#   v2   : ARC_MXKV_VPROBE=2  -- V severed (staging ON): K+P+base
#   v8   : ARC_MXKV_VPROBE=8  -- K+V+P all severed: base floor
#   full : r213 A/A mean 70.2 ms/tok (same wheel) -- not re-run
# Lane arithmetic: K = v2 - v8; V = v6 - v8; P+resid = full - v2 - v6
# + v8. Severed cells produce GARBAGE quality -> MTP acceptance
# collapses, but itl_ms_p50 is the inter-CHUNK gap = per-STEP time and
# the verify call runs every step regardless of acceptance, so the
# bucket shape holds; bucket drift is caught by the dispatch telemetry
# rows= (r213 reference rows=148) and chunk counts land in the report.
# Bank -> /root/build/p647/r215/{dbf0,v6,v2,v8}
set -uo pipefail
R=/root/build
P=$R/p647/r215
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r215
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R215_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R215_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R215_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R215_NO_OVERLAY"; exit 1; }
grep -q "ARC_MXKV_VPROBE" $R/serve_arc_b70.sh \
  || { echo "R215_SERVE_HOOK_MISSING (passthrough not inserted?)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R215_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R215_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R215_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R215_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r215.log 2>&1 &
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

run_cell() {  # tag kv extra-env-name extra-env-value
  local CELL=$1 KV=$2 EN=$3 EV=$4
  CD=$P/$CELL
  mkdir -p "$CD"
  echo "=== [r215:$CELL] boot kv=$KV $EN=$EV $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KV \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    $EN="$EV" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R215_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r215_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R215_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
    verify_removed || true; return 0; }
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
  sleep 3
  group 32768 1
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
    > "$CD/telemetry.txt" || true
  docker exec $CNAME grep "MXKV native dispatch" /root/serve_full.log 2>/dev/null \
    > "$CD/dispatch.txt" || true
  docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$CD/erraudit.txt" 2>/dev/null || true
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "=== [r215:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r215] W14 quiesce $(date +%H:%M:%S) ==="
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

# order: dbf0 first (correct output -- the cleanest anchor), then the
# severance ladder v6 (K out), v2 (V out), v8 (floor).
run_cell dbf0 mxfp4 ARC_MXKV_DIRECTBF16 0
run_cell v6   mxfp4 ARC_MXKV_VPROBE 6
run_cell v2   mxfp4 ARC_MXKV_VPROBE 2
run_cell v8   mxfp4 ARC_MXKV_VPROBE 8

echo "=== [r215] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
FULL = 70.2  # r213 A/A mean, mxfp4 c32768 n1, same wheel
FP8 = 42.5

def cell(name):
    f = os.path.join(P, name, "c32768_n1", "p139_conc_c32768_n1.json")
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
print("r215 engine-shape lane decomposition -- mxfp4 c32768 n1 (r212aa wheel)")
print(f" full (r213 mean) {FULL:.1f} ms/tok-step; fp8 bar {FP8}")
vals = {}
for name, desc in (
    ("dbf0", "float decode tail (correct output)"),
    ("v6", "K severed (V+P+base)"),
    ("v2", "V severed (K+P+base)"),
    ("v8", "floor (K+V+P severed)"),
):
    itl, errs, chunks, rows = cell(name)
    if itl is None:
        print(f" {name:>4} | MISSING (errs={errs})")
        ok = False
        continue
    vals[name] = itl
    rnote = f"rows={rows}" + (" [DRIFT? r213 ref 148]" if rows not in (None, 148) else "")
    print(f" {name:>4} | {itl:7.1f} | chunks~{chunks:5.0f} | {rnote} | errs={errs}"
          f"  ({desc})")
    if errs:
        ok = False
if "v2" in vals and "v8" in vals and "v6" in vals:
    k_lane = vals["v2"] - vals["v8"]
    v_lane = vals["v6"] - vals["v8"]
    p_resid = FULL - vals["v2"] - vals["v6"] + vals["v8"]
    mx_tot = FULL - FP8
    print(f" ENGINE LANE TABLE (ms/tok-step at c32768):")
    print(f"   K lane  (v2-v8)      {k_lane:+7.1f}")
    print(f"   V lane  (v6-v8)      {v_lane:+7.1f}")
    print(f"   P+resid (full-v2-v6+v8) {p_resid:+7.1f}")
    print(f"   MX total (full-fp8)  {mx_tot:+7.1f}")
    if "dbf0" in vals:
        print(f"   decode-tail pool (dbf0-full) {vals['dbf0']-FULL:+7.1f}")
    if k_lane > v_lane * 1.5:
        print("   TARGET: the K lane dominates -- attack the K walk/decode")
    elif v_lane > k_lane * 1.5:
        print("   TARGET: the V lane dominates -- attack the V gather/decode")
    else:
        print("   TARGET: balanced K/V -- shared/floor levers")
if "dbf0" in vals:
    d = vals["dbf0"] - FULL
    if d < 1.0:
        print("   NOTE: decode tail is e2e-CHEAP (<1 ms/tok) -- SWAR class"
              " confirmed dead at engine shape too")
for name in ("dbf0", "v6", "v2", "v8"):
    tf = os.path.join(P, name, "telemetry.txt")
    nt = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    if nt < 50:
        print(f"  WARN_telemetry_{name}: native_call_lines={nt} (<50)")
print("R215_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R215_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
