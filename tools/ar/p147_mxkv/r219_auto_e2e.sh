#!/bin/bash
# r219_auto_e2e.sh -- arc7 Phase-8 r219: the AUTO RE-PRICE validation
# on the r219 wheel (fafc6d4d: mx_get_num_splits multiplier 4 -> 13
# slots/head + policy_split_cap 256 + kernel ceiling 256 + overlay
# splits telemetry). The payoff cell is AUTO: ARC_MXKV_NSPLIT UNSET ->
# the C++ heuristic must land ~130 at the c32768 bucket (behavioral
# readout: @32K ~= the banked ns128 posture 56.8; if it reads ~62.2
# auto picked 64, ~70.2 picked 40). Tail cells ns192/ns256 price the
# ceiling extension (the r216-217 curve decelerates: -11.45/-6.45/
# -2.6 pct per octave; expect <=1-2 pct). QUALITY CROSS-CHECK (r216b
# law): every cell reports per_user_tps + chunks; chunks > 80 marks
# INVALID_FOR_SPEED. The native-call telemetry now prints
# splits=<int|auto> (r219 overlay) -- captured per cell.
# Bank -> /root/build/p647/r219/{auto,ns192,ns256}
set -uo pipefail
R=/root/build
P=$R/p647/r219
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r219
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=fafc6d4dedda71ea1a6957f88d849856
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R219_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R219_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R219_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R219_NO_OVERLAY"; exit 1; }
grep -q "num_splits_kv=_ns" "$OVERLAY/native_decode.py" \
  || { echo "R219_OVERLAY_HOOK_MISSING"; exit 1; }
grep -q "splits=%s" "$OVERLAY/native_decode.py" \
  || { echo "R219_OVERLAY_TELEMETRY_MISSING (splits= not logged)"; exit 1; }
grep -q "ARC_MXKV_NSPLIT" $R/serve_arc_b70.sh \
  || { echo "R219_SERVE_HOOK_MISSING"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R219_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R219_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R219_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R219_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r219.log 2>&1 &
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

run_cell() {  # tag nsplit (0 = auto: env left unset)
  local CELL=$1 NS=$2
  CD=$P/$CELL
  mkdir -p "$CD"
  local PIN=""
  [ "$NS" != "0" ] && PIN="ARC_MXKV_NSPLIT=$NS"
  echo "=== [r219:$CELL] boot pin=${PIN:-AUTO} $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    ${PIN:+$PIN} \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R219_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r219_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R219_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r219:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r219] W14 quiesce $(date +%H:%M:%S) ==="
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

run_cell auto 0
run_cell ns192 192
run_cell ns256 256

echo "=== [r219] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, re, sys
P = sys.argv[1]
# References: banked posture ns128 (r217/r217b/r218 full), ns64, auto-40
# (old wheel), fp8 bar. tps = per-user MTP tps (quality arbiter).
NS128 = {8192: 48.5, 16384: 51.2, 32768: 56.8}
NS128TPS = {8192: 71.7, 16384: 67.1, 32768: 66.7}
NS64 = {8192: 47.3, 16384: 52.3, 32768: 62.2}
OLDAUTO = {8192: 48.5, 16384: 55.9, 32768: 70.2}
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
print("r219 auto re-price + tail -- mxfp4 n1 (r219 wheel fafc6d4d)")
print(f" {'cell':>5} | {'ctx':>6} | {'itl':>7} | {'vs ns128':>8} | {'tps':>6} | {'chunks':>7} | qual | errs")
for name in ("auto", "ns192", "ns256"):
    for ctx in (8192, 16384, 32768):
        r = cell(name, ctx)
        if r is None:
            print(f" {name:>5} | {ctx:>6} | MISSING")
            ok = False
            continue
        itl, chunks, tps, errs = r
        dev = (itl - NS128[ctx]) / NS128[ctx] * 100.0
        qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
        if chunks > 80 or errs:
            ok = False
        ts = "%6.1f" % tps if isinstance(tps, (int, float)) else "   n/a"
        print(f" {name:>5} | {ctx:>6} | {itl:7.1f} | {dev:+7.2f}% | {ts} | ~{chunks:5.0f} | {qual} | {errs}")
    print(f"   telemetry splits={splits_logged(name)}")
a = cell("auto", 32768)
if a:
    itl = a[0]
    d = (itl - NS128[32768]) / NS128[32768] * 100.0
    if abs(d) <= 2.0:
        print(f" AUTO_CONFIRMED: auto @32K {itl:.1f} vs ns128 posture {NS128[32768]}"
              f" ({d:+.2f}%) -- the re-priced heuristic lands the validated posture")
    else:
        near64 = abs(itl - NS64[32768]) / NS64[32768]
        near40 = abs(itl - OLDAUTO[32768]) / OLDAUTO[32768]
        hint = "auto looks like it picked 64 (reduce multiplier arithmetic wrong)" if near64 < 0.02 \
            else "auto looks like 40 (old cap path?)" if near40 < 0.02 else "unexpected value"
        print(f" AUTO_FLAG: auto @32K {itl:.1f} ({d:+.2f}% vs ns128) -- {hint}")
        ok = False
    print(f"  @32K vs fp8: {itl/FP8[32768]:.2f}x")
print("R219_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R219_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
