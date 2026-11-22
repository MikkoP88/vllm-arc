#!/bin/bash
# r214_k8v4.sh -- arc7 Phase-8 r214: the K-vs-V E2E LANE SPLIT on the
# converged r212aa wheel (9a01790b) via the mxk8v4 hybrid (e4m3 K +
# e2m1 V) -- zero code changes, real quality, the r213-lawful
# engine-level attribution the probe cells can no longer provide
# (probe/e2e sign inversion: probe lanes are shape-scoped).
#   m4   = e2m1 K + e2m1 V : 48.5 / 55.9 / 70.2  (r213 mean, A/A)
#   m8   = e4m3 K + e4m3 V : 51.6 / 63.2 / 84.8
#   k8v4 = e4m3 K + e2m1 V : THIS LEG (r212h-era bank 48.9/57.3/78.3 is
#                             cross-era -- pre-r212m wheel, unusable)
# Split arithmetic (per ctx): k8v4 - m4 = the e4m3-K excess (decode +
# 2x K bytes vs e2m1); m8 - k8v4 = the e4m3-V excess. The larger excess
# names the side the gap rides; sum must equal m8 - m4 (consistency
# check). per-user MTP tps rides along (acceptance signal): m4
# 75.8/62.6/51.8, m8 71.6/58.9/40.6.
# Gates: faces complete, 0 errors, native telemetry present. The split
# itself is READOUT (no pass/fail on direction).
# Bank -> /root/build/p647/r214/mkqa
set -uo pipefail
R=/root/build
P=$R/p647/r214
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r214
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
WHEELMD5=9a01790b5748cb4d7e1eac9c099a997e
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R214_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R214_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEELDIR/$WHEELNAME" | awk '{print $1}')
[ "$GOTMD5" = "$WHEELMD5" ] \
  || { echo "R214_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (stale wheel?)"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R214_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R214_STALE_OVERLAY (argument fix not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R214_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R214_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R214_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R214_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r214.log 2>&1 &
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

echo "=== [r214] W14 quiesce $(date +%H:%M:%S) ==="
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

CD=$P/mkqa
mkdir -p "$CD"
echo "=== [r214:mkqa] boot kv=mxk8v4 $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxk8v4 \
  ARC_MXKV_NATIVE=1 \
  ARC_MXKV_OVERLAY=$OVERLAY \
  ARC_MXKV_WHEEL_DIR=$WHEELDIR \
  ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R214_BOOT_FAIL_mkqa"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  echo "BOOT_FAIL" > "$CD/outcome.txt"
  exit 1
fi
echo "HEALTH_OK_r214_mkqa $(date +%H:%M:%S)"
docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
  echo "R214_PROBE_CP_FAIL"; exit 1; }
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
echo "=== [r214:mkqa] done $(date +%H:%M:%S) ==="

echo "=== [r214] report $(date +%H:%M:%S) ==="
python3 - "$P" <<'PY'
import json, os, sys
P = sys.argv[1]
# r213 A/A-confirmed means on this wheel (ms/tok ITL p50; tps n1).
M4 = {8192: 48.5, 16384: 55.9, 32768: 70.2}
M8 = {8192: 51.6, 16384: 63.2, 32768: 84.8}
M4TPS = {8192: 75.8, 16384: 62.6, 32768: 51.8}
M8TPS = {8192: 71.6, 16384: 58.9, 32768: 40.6}
FP8 = {8192: 40.4, 16384: 41.1, 32768: 42.5}

def face(ctx):
    f = os.path.join(P, "mkqa", f"c{ctx}_n1", f"p139_conc_c{ctx}_n1.json")
    if not os.path.exists(f):
        return None, None, None
    j = json.load(open(f))
    rs = [r for r in j["requests"] if r["error"] is None]
    errs = len([r for r in j["requests"] if r["error"] is not None])
    if not rs:
        return None, errs, None
    itls = [r["itl_ms_p50"] for r in rs]
    tps = j.get("per_user_tps")
    tps = tps[0] if isinstance(tps, list) and tps else tps
    return (sum(itls) / len(itls)), errs, tps

ok = True
print("r214 K-vs-V e2e lane split -- mxk8v4 (e4m3 K + e2m1 V) on r212aa")
print(f" {'ctx':>6} | {'k8v4':>7} | {'m4':>7} | {'m8':>7} | {'K-excess':>8} | {'V-excess':>8} | {'chk':>5} | {'tps k8v4':>8} | {'tps m4':>6} | errs")
for ctx in (8192, 16384, 32768):
    new, errs, tps = face(ctx)
    if new is None:
        print(f" {ctx:>6} | MISSING (errs={errs})")
        ok = False
        continue
    kex = new - M4[ctx]        # e4m3 K over e2m1 K
    vex = M8[ctx] - new        # e4m3 V over e2m1 V
    chk = abs((kex + vex) - (M8[ctx] - M4[ctx])) < 0.15
    ts = f"{tps:8.1f}" if isinstance(tps, (int, float)) else "     n/a"
    print(f" {ctx:>6} | {new:7.1f} | {M4[ctx]:7.1f} | {M8[ctx]:7.1f} | {kex:+8.1f} | {vex:+8.1f} | {'ok' if chk else 'BAD':>5} | {ts} | {M4TPS[ctx]:6.1f} | {errs}")
    if errs:
        ok = False
    if ctx == 32768:
        tot = new - FP8[ctx]
        print(f"  32K readout: k8v4-vs-fp8 gap {tot:+.1f} ms/tok;"
              f" K-side e4m3 excess {kex:+.1f} / V-side {vex:+.1f}")
        if abs(kex) > abs(vex) * 1.5:
            print("  SPLIT: the e4m3 K side carries the asymmetry -- the K")
            print("  walk/decode lane is the attack target")
        elif abs(vex) > abs(kex) * 1.5:
            print("  SPLIT: the e4m3 V side carries the asymmetry -- the V")
            print("  walk/decode lane is the attack target")
        else:
            print("  SPLIT: balanced -- both sides carry comparable excess")
tf = os.path.join(P, "mkqa", "telemetry.txt")
df = os.path.join(P, "mkqa", "dispatch.txt")
nt = sum(1 for _ in open(tf)) if os.path.exists(tf) else 0
nd = sum(1 for _ in open(df)) if os.path.exists(df) else 0
print(f" telemetry[mkqa]: native_call_lines={nt} dispatch_lines={nd}")
if nt < 50 or nd < 1:
    print("  FAIL_telemetry: faces did not ride the native bridge")
    ok = False
print("R214_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R214_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
