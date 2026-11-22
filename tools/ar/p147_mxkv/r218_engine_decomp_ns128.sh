#!/bin/bash
# r218_engine_decomp_ns128.sh -- arc7 Phase-8 r218: the r215 lane
# decomposition RE-PRICED at the banked ns128 posture (r217 wheel
# 9ded4fb6). The r215 table (floor 50.8, K +1.6, V +5.0 at c32768) was
# priced at auto~40 splits; the split-depth win (62.2 -> 56.8) reshaped
# the pools, so the next-surgery decision (kv_tile 16->64 vs deeper
# ceiling vs auto-formula) needs the ns128-conditioned table. Same env
# severance (ARC_MXKV_VPROBE through serve passthrough; C++ reads it
# per call), zero code changes; every boot pins ARC_MXKV_NSPLIT=128.
# Cells (one boot each, c32768 n1):
#   full : no probe env -- the in-batch reference (3rd ns128 full
#          sample; r217 56.7 / r217b 56.8 must reproduce)
#   v6   : ARC_MXKV_VPROBE=6  -- K chain severed: V+P+base
#   v2   : ARC_MXKV_VPROBE=2  -- V severed: K+P+base
#   v8   : ARC_MXKV_VPROBE=8  -- all severed: base floor
# Lane arithmetic: K = v2 - v8; V = v6 - v8; P+resid = full - v2 - v6
# + v8. Severed cells stream garbage (acceptance collapse) but
# itl_ms_p50 stays per-STEP; bucket shape caught by rows= (ref 148).
# Bank -> /root/build/p647/r218/{full,v6,v2,v8}
set -uo pipefail
R=/root/build
P=$R/p647/r218
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r218
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9ded4fb67e15ba092aac607d58db1b72
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R218_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R218_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R218_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R218_NO_OVERLAY"; exit 1; }
grep -q "ARC_MXKV_VPROBE" $R/serve_arc_b70.sh \
  || { echo "R218_SERVE_HOOK_MISSING (passthrough not inserted?)"; exit 1; }
grep -q "ARC_MXKV_NSPLIT" $R/serve_arc_b70.sh \
  || { echo "R218_SERVE_HOOK_NSPLIT_MISSING"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R218_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R218_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R218_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R218_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r218.log 2>&1 &
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

run_cell() {  # tag extra-env-name extra-env-value ("" = none)
  local CELL=$1 EN=$2 EV=$3
  # single unquoted word "NAME=VALUE" appended to env when set; the
  # r215 form $EN="$EV" leaves a literal ="" arg when EN is empty
  local XENV=""
  [ -n "$EN" ] && XENV="$EN=$EV"
  CD=$P/$CELL
  mkdir -p "$CD"
  echo "=== [r218:$CELL] boot NSPLIT=128 ${XENV:-no-probe} $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    ARC_MXKV_NSPLIT=128 \
    ${XENV:+$XENV} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R218_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r218_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R218_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r218:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r218] W14 quiesce $(date +%H:%M:%S) ==="
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

# order: full first (the anchor -- 3rd ns128 sample), then the
# severance ladder v6 (K out), v2 (V out), v8 (floor).
run_cell full ""  ""
run_cell v6   "ARC_MXKV_VPROBE" 6
run_cell v2   "ARC_MXKV_VPROBE" 2
run_cell v8   "ARC_MXKV_VPROBE" 8

echo "=== [r218] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
# r215 references (auto~40 splits, r212aa wheel) for delta-of-deltas
R215 = {"full": 70.2, "dbf0": 75.6, "v6": 68.6, "v2": 65.2, "v8": 50.8}
NS128 = 56.8   # r217b A/A (posture)
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
    tps = j.get("per_user_tps")
    tps = tps[0] if isinstance(tps, list) and tps else None
    rows = None
    df = os.path.join(P, name, "dispatch.txt")
    if os.path.exists(df):
        m = re.search(r"rows=(\d+)", open(df).read())
        rows = int(m.group(1)) if m else None
    return itl, errs, chunks, rows

ok = True
vals = {}
print("r218 ns128-conditioned lane decomposition -- mxfp4 c32768 n1")
print(f" posture ref: r217/r217b ns128 = 56.7/56.8; fp8 bar {FP8}; r215 auto-table {R215}")
for name, desc in (
    ("full", "no probe (in-batch anchor)"),
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
    rnote = "rows=%s" % rows + (" [DRIFT? ref 148]" if rows not in (None, 148) else "")
    ch = "~%4.0f" % chunks if chunks else " n/a"
    print(f" {name:>4} | {itl:7.1f} | chunks {ch} | {rnote} | errs={errs}  ({desc})")
    if errs or rows not in (None, 148):
        ok = False
if "full" in vals:
    d = (vals["full"] - NS128) / NS128 * 100.0
    if abs(d) <= 2.0:
        print(f" ANCHOR_OK full @ns128: {vals['full']:.1f} vs posture {NS128} ({d:+.2f}%)")
    else:
        print(f" ANCHOR_FLAG full @ns128: {vals['full']:.1f} ({d:+.2f}%) -- investigate")
        ok = False
if "v2" in vals and "v8" in vals and "v6" in vals and "full" in vals:
    F = vals["full"]
    k_lane = vals["v2"] - vals["v8"]
    v_lane = vals["v6"] - vals["v8"]
    p_resid = F - vals["v2"] - vals["v6"] + vals["v8"]
    mx_tot = F - FP8
    print(" ENGINE LANE TABLE @ns128 (ms/tok-step at c32768; r215-auto in parens):")
    print(f"   K lane  (v2-v8)          {k_lane:+7.1f}  ({R215['v2']-R215['v8']:+.1f})")
    print(f"   V lane  (v6-v8)          {v_lane:+7.1f}  ({R215['v6']-R215['v8']:+.1f})")
    print(f"   P+resid (full-v2-v6+v8)  {p_resid:+7.1f}  ({R215['full']-R215['v2']-R215['v6']+R215['v8']:+.1f})")
    print(f"   MX total (full-fp8)      {mx_tot:+7.1f}  ({R215['full']-FP8:+.1f})")
    floor_gap = vals["v8"] - FP8
    print(f"   FLOOR (v8-fp8)           {floor_gap:+7.1f}  ({R215['v8']-FP8:+.1f})"
          "  <-- ctx-flat skeleton cost")
    union = F - vals["v8"]
    print(f"   K/V union (full-v8)      {union:+7.1f}  ({R215['full']-R215['v8']:+.1f})"
          "  <-- ctx-scaling pool (kv_tile-64 attacks this)")
    if floor_gap > union:
        print("   TARGET: the FLOOR dominates the remaining gap -- skeleton/launch/")
        print("           addressing levers (kv_tile policy changes BOTH -- floor via")
        print("           fewer tile-iterations, union via wider walk)")
    elif union > floor_gap * 1.5:
        print("   TARGET: the union dominates -- kv_tile 16->64 walk policy first")
    else:
        print("   TARGET: balanced -- bundle kv_tile-64 + deeper ceiling in one build")
print("R218_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R218_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
