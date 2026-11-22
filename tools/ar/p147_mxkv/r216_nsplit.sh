#!/bin/bash
# r216_nsplit.sh -- arc7 Phase-8 r216: the ENGINE-SIDE SPLIT-DEPTH
# sweep on the converged r212aa wheel (9a01790b), mxfp4 lane, c32768
# n1 -- e2e re-price of the split surface after r215 (the r209/r210
# "reduce cap 40 optimal" verdict was probe-priced pre-template-split;
# sign-inversion law demands the engine face). The fork-owned overlay
# now pins num_splits_kv from ARC_MXKV_NSPLIT (unset = 0 = the C++
# auto heuristic, ~40 at this capacity-frozen bucket; serve script
# passes the env through). Cells (one boot each):
#   ns40 : ARC_MXKV_NSPLIT=40 -- the ANCHOR: must reproduce the r213
#          A/A mean 70.2 +-1 pct (validates the knob plumbing end to
#          end; also separates "auto really picks 40" from doubt)
#   ns8  : ARC_MXKV_NSPLIT=8  -- compact grid: fewer empty WGs (the
#          262400-priced 40 splits cover 8x the real 32K walk), less
#          reduce traffic
#   ns64 : ARC_MXKV_NSPLIT=64 -- deeper parallelism, more empty-WG +
#          reduce cost
# Readout (no direction gate on ns8/ns64; the anchor gates).
# Bank -> /root/build/p647/r216/{ns40,ns8,ns64}
set -uo pipefail
R=/root/build
P=$R/p647/r216
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r216
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R216_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R216_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R216_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R216_NO_OVERLAY"; exit 1; }
grep -q "num_splits_kv=_ns" "$OVERLAY/native_decode.py" \
  || { echo "R216_OVERLAY_HOOK_MISSING (NSPLIT pin not shipped)"; exit 1; }
grep -q "ARC_MXKV_NSPLIT" $R/serve_arc_b70.sh \
  || { echo "R216_SERVE_HOOK_MISSING (passthrough not inserted)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R216_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R216_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R216_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R216_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r216.log 2>&1 &
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
  echo "=== [r216:$CELL] boot ARC_MXKV_NSPLIT=$NS $(date +%H:%M:%S) ==="
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
    echo "R216_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r216_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R216_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r216:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r216] W14 quiesce $(date +%H:%M:%S) ==="
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

run_cell ns40 40
run_cell ns8 8
run_cell ns64 64

echo "=== [r216] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
REF = 70.2  # r213 A/A mean, mxfp4 c32768 n1, same wheel, auto splits

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
vals = {}
print("r216 engine split-depth sweep -- mxfp4 c32768 n1 (r212aa wheel)")
print(f" reference (auto splits, r213 A/A mean) {REF:.1f} ms/tok-step")
for name, desc in (
    ("ns40", "anchor: pinned 40 = expected auto"),
    ("ns8", "compact grid (fewer empty WGs / less reduce)"),
    ("ns64", "deeper splits (more empty WGs / reduce)"),
):
    itl, errs, chunks, rows = cell(name)
    if itl is None:
        print(f" {name:>4} | MISSING (errs={errs})")
        ok = False
        continue
    vals[name] = itl
    dev = (itl - REF) / REF * 100.0
    rnote = f"rows={rows}" + (" [DRIFT? r213 ref 148]" if rows not in (None, 148) else "")
    print(f" {name:>4} | {itl:7.1f} | {dev:+6.2f} pct | chunks~{chunks:5.0f} | {rnote} | errs={errs}"
          f"  ({desc})")
    if errs:
        ok = False
if "ns40" in vals:
    dev40 = abs(vals["ns40"] - REF) / REF * 100.0
    if dev40 > 1.0:
        print(f" ANCHOR_FAIL: ns40 {vals['ns40']:.1f} vs ref {REF} = {dev40:.2f} pct"
              " -- either auto does not pick 40 or the knob plumbing is off;"
              " treat ns8/ns64 deltas as unvalidated")
        ok = False
    else:
        print(f" ANCHOR_OK: ns40 reproduces the reference within {dev40:.2f} pct"
              " -- auto=40 confirmed, knob live, ns8/ns64 deltas valid")
        for n, lbl in (("ns8", "compact"), ("ns64", "deep")):
            if n in vals:
                d = (vals[n] - vals["ns40"]) / vals["ns40"] * 100.0
                if d < -2.0:
                    print(f"  WIN_{n}: {d:+.2f} pct vs anchor -- new split posture candidate")
                elif d > 2.0:
                    print(f"  LOSS_{n}: {d:+.2f} pct vs anchor")
                else:
                    print(f"  NEUTRAL_{n}: {d:+.2f} pct vs anchor")
for name in ("ns40", "ns8", "ns64"):
    tf = os.path.join(P, name, "telemetry.txt")
    nt = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
    if nt < 50:
        print(f"  WARN_telemetry_{name}: native_call_lines={nt} (<50)")
print("R216_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R216_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
