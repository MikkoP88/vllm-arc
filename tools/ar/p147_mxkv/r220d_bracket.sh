#!/bin/bash
# r220d_bracket.sh -- arc7 Phase-8 r220d: split-depth bracket of the
# native _64 tile under ENGINE load. The r220b A/A verdict: p64 is
# +0.86/+5.63/+13.40% itl vs the same-wheel _16 control (48.7/53.8/64.0
# vs 48.3/51.0/56.4) while the r220a offline ladder says _64 is FASTER
# (-11.2% @32K single-call) -- the r213 probe/e2e sign-inversion class.
# Mechanism hypothesis: the _64 WG stages a 4x taller V image in SLM
# (DIRECTBF16), so per-WG SLM quadruples and residency collapses ONLY
# under the engine's 16-row concurrent-call load; the offline probe's
# single call never sees the contention.
# DECISIVE BRACKET: p64 with the split env pinned at 64 (fewest WGs,
# lowest staging pressure) and 256 (max WG count under the wide tile).
# If neither face rescues ctl@32K = 56.4, the inversion is fundamental
# occupancy, the tile lever closes e2e-negative, and the r221 road
# (if any) is V sub-staging (ping-pong 16-row V slices keeping the
# 4-SG K walk), not split tuning.
# Cells (one boot each, face c32768 n1):
#   n64s64  : ARC_MXKV_P64NATIVE=1 ARC_MXKV_NSPLIT=64
#   n64s256 : ARC_MXKV_P64NATIVE=1 ARC_MXKV_NSPLIT=256
# QUALITY CROSS-CHECK (r216b law): chunks <= 80 + tps per cell.
# Bank -> /root/build/p647/r220d/{n64s64,n64s256}
set -uo pipefail
R=/root/build
P=$R/p647/r220d
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r220d
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R220D_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R220D_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R220D_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
[ -f "$OVERLAY/engine.py" ] || { echo "R220D_NO_OVERLAY"; exit 1; }
grep -q "P64NATIVE" $R/serve_arc_b70.sh \
  || { echo "R220D_SERVE_HOOK_MISSING (P64NATIVE forward)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R220D_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R220D_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R220D_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R220D_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r220d.log 2>&1 &
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
  echo "=== [r220d:$CELL] boot p64=1 ns=$NS $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    ARC_MXKV_P64NATIVE=1 ARC_MXKV_NSPLIT=$NS \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R220D_BOOT_FAIL_$CELL"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r220d_$CELL $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R220D_PROBE_CP_FAIL_$CELL"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r220d:$CELL] done $(date +%H:%M:%S) ==="
}

echo "=== [r220d] W14 quiesce $(date +%H:%M:%S) ==="
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

run_cell n64s64 64
run_cell n64s256 256

echo "=== [r220d] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# r220b same-wheel anchors: ctl(_16, ns auto->128) and p64(ns auto->128)
CTL32 = 56.4
P64_128 = 64.0
POSTURE32 = 56.8

def cell(name):
    f = os.path.join(P, name, "c32768_n1", "p139_conc_c32768_n1.json")
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

print("r220d p64 split bracket @c32768 n1 -- mxfp4 (anchors: ctl 56.4, p64@128 64.0, posture 56.8)")
print(f" {'cell':>8} | {'itl':>6} | vs ctl | vs p64@128 | {'tps':>6} | {'chunks':>7} | qual | errs")
best = None
for name in ("n64s64", "n64s256"):
    r = cell(name)
    if r is None:
        print(f" {name:>8} | MISSING"); continue
    itl, chunks, tps, errs = r
    vc = 100.0 * (itl - CTL32) / CTL32
    vp = 100.0 * (itl - P64_128) / P64_128
    qual = "OK" if chunks <= 80 else "INVALID_FOR_SPEED"
    ts = "%6.1f" % tps if isinstance(tps, (int, float)) else "   n/a"
    print(f" {name:>8} | {itl:6.1f} | {vc:+5.1f}% | {vp:+9.1f}% | {ts} | ~{chunks:5.0f} | {qual} | {errs}")
    if qual == "OK" and (best is None or itl < best[1]):
        best = (name, itl)
if best:
    rescued = best[1] < CTL32
    print(f"BEST {best[0]} itl={best[1]:.1f} -> {'RESCUED_BELOW_CTL' if rescued else 'NOT_RESCUED (tile lever e2e-negative at every split depth)'}")
print("R220D_DONE")
PY
echo "R220D_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
