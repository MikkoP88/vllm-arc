#!/bin/bash
# r212g_e2e.sh -- arc7 Phase-8 r212g: e2e validation of the REAL
# split fix. r212f confirmed kernel-level causality (m9 36326.9 us
# vs ctx 1269.6 vs cap 1265.4, canary 0 -- 28.61x on one scalar
# arg); the bridge now passes max_seqlen_k=_max_sk (capacity
# 262400). One boot, native posture, the r212b face set (m4qa):
# c16384 n1 / c32768 n1 / c16384 n8. A-side (all-triton) and the
# pre-fix B-side numbers are BANKED in r212b -- this leg compares
# against them, no re-run needed.
# Guard (audit lesson r212f): the overlay must contain the ARGUMENT
# 'max_seqlen_k=_max_sk' -- a telemetry-only edit is a no-op.
# Bank -> /root/build/p647/r212g/B_native_fixed
set -uo pipefail
R=/root/build
P=$R/p647/r212g
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212g
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R212G_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R212G_NO_WHEEL"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212G_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R212G_STALE_OVERLAY (argument fix not shipped -- audit law: guard the ARG, not the log line)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212G_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212G_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212G_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212G_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212g.log 2>&1 &
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
  local cap=$(( 300 + CTX / 8192 * 90 + 120 + (NC == 8 ? 240 : 0) ))
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

echo "=== [r212g] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

CD=$P/B_native_fixed
mkdir -p "$CD"
md5sum "$OVERLAY"/*.py > "$P/overlay_md5.txt" 2>/dev/null || true
sha256sum "$WHEELDIR/$WHEELNAME" > "$P/wheel_sha256.txt"

echo "=== [r212g] boot native posture (fixed bridge) $(date +%H:%M:%S) ==="
nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
  ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
  ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp4 \
  ARC_MXKV_NATIVE=1 \
  ARC_MXKV_OVERLAY=$OVERLAY \
  ARC_MXKV_WHEEL_DIR=$WHEELDIR \
  ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
  bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
if ! boot_wait; then
  echo "R212G_BOOT_FAIL"
  docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
  docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
  exit 1
fi
echo "HEALTH_OK_r212g $(date +%H:%M:%S)"
docker cp "$PROBE" $CNAME:/root/p139_conc.py || { echo "R212G_PROBE_CP_FAIL"; exit 1; }

curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"warmup","max_tokens":16,"temperature":0}' > /dev/null
sleep 3
group 16384 1
group 32768 1
group 16384 8
docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
docker exec $CNAME grep "MXKV verify decoder" /root/serve_full.log 2>/dev/null \
  > "$CD/decoder.txt" || true
docker exec $CNAME grep "MXKV native dispatch" /root/serve_full.log 2>/dev/null \
  > "$CD/dispatch.txt" || true
docker exec $CNAME grep "MXKV native call" /root/serve_full.log 2>/dev/null \
  > "$CD/telemetry.txt" || true
docker exec $CNAME sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
  > "$CD/erraudit.txt" 2>/dev/null || true
docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true

echo "=== [r212g] report $(date +%H:%M:%S) ==="
python3 - "$R/p647" <<'PY'
import json, os, sys
root = sys.argv[1]
def itl_of(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    itls = [r["itl_ms_p50"] for r in j["requests"] if r["error"] is None]
    if not itls:
        return None
    return sum(itls) / len(itls)

SIDES = [
    ("A_triton  (r212b)", os.path.join(root, "r212b", "A_triton")),
    ("B_native  (r212b)", os.path.join(root, "r212b", "B_native")),
    ("B_FIXED   (r212g)", os.path.join(root, "r212g", "B_native_fixed")),
]
print("r212g -- ITL p50 ms/token, fix validation (banked r212b A/B vs fixed bridge)")
vals = {}
for side, d in SIDES:
    for tag in ("c16384_n1", "c32768_n1", "c16384_n8"):
        v = itl_of(os.path.join(d, tag, f"p139_conc_{tag}.json"))
        vals[(side, tag)] = v
        print(f" {side:19s} {tag:10s} {'itl %8.1f ms' % v if v is not None else 'MISSING'}")
ok = True
for tag in ("c16384_n1", "c32768_n1", "c16384_n8"):
    fixed = vals.get(("B_FIXED   (r212g)", tag))
    pre = vals.get(("B_native  (r212b)", tag))
    tri = vals.get(("A_triton  (r212b)", tag))
    if fixed is None or pre is None or tri is None:
        ok = False; continue
    print(f" {tag}: fixed/pre {pre/fixed:.2f}x faster   fixed vs triton {tri/fixed:.2f}x")
    if fixed > pre / 2:
        print(f"  FAIL_{tag}: fix did not improve >=2x"); ok = False
    elif fixed > tri:
        print(f"  SLOWER_THAN_TRITON_{tag}")
print("R212G_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "=== [r212g] telemetry (fixed) ==="
head -6 "$CD/telemetry.txt" 2>/dev/null
echo "R212G_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
