#!/bin/bash
# r212h_e2e.sh -- arc7 Phase-8 r212h: generalize the r212g split-fix
# validation to the remaining MX formats. mxfp8 and mxk8v4 ride the
# SAME bridge (native_decode.py md5 19f662b8 -- passes
# max_seqlen_k=_max_sk capacity), so the fix should transfer
# unchanged; this leg proves it e2e and prices the post-fix lane
# vs the launch4 pre-fix banks (same stack, /root/build/p647/r212a/
# {m8qa,mkqa}, 9 faces per format). Two boots (one per format),
# faces c8192_n1 / c16384_n1 / c32768_n1 (short/mid/long; m8qa's
# minlen was 16384 so 8192 tests below the old crossover too --
# capacity pricing freezes native everywhere now, benign per
# r212g). Guard: the ARGUMENT string (audit law).
# Bank -> /root/build/p647/r212h/{m8qa,mkqa}
set -uo pipefail
R=/root/build
P=$R/p647/r212h
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r212h
PROBE=$R/p139_conc.py
WHEELDIR=$R/wheels-arc-vxk
WHEELNAME=vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
OVERLAY=$R/p199/overlay/mx_kv

[ -f "$PROBE" ] || { echo "R212H_NO_PROBE"; exit 1; }
[ -f "$WHEELDIR/$WHEELNAME" ] || { echo "R212H_NO_WHEEL"; exit 1; }
[ -f "$OVERLAY/engine.py" ] || { echo "R212H_NO_OVERLAY"; exit 1; }
grep -q "max_seqlen_k=_max_sk" "$OVERLAY/native_decode.py" \
  || { echo "R212H_STALE_OVERLAY (argument fix not shipped)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212H_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212H_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212H_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212H_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212h.log 2>&1 &
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
  echo "=== [r212h:$FT] boot kv=$KV $(date +%H:%M:%S) ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME \
    ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 \
    ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=$KV \
    ARC_MXKV_NATIVE=1 \
    ARC_MXKV_OVERLAY=$OVERLAY \
    ARC_MXKV_WHEEL_DIR=$WHEELDIR \
    ARC_MXKV_WHEEL=/wheels/$WHEELNAME \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait; then
    echo "R212H_BOOT_FAIL_$FT"
    docker cp $CNAME:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker exec $CNAME tail -25 /root/serve_full.log 2>/dev/null
    docker rm -f $CNAME >/dev/null 2>&1; verify_removed || true
    echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r212h_$FT $(date +%H:%M:%S)"
  docker cp "$PROBE" $CNAME:/root/p139_conc.py || { \
    echo "R212H_PROBE_CP_FAIL_$FT"; docker rm -f $CNAME >/dev/null 2>&1; \
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
  echo "=== [r212h:$FT] done $(date +%H:%M:%S) ==="
}

echo "=== [r212h] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

mkdir -p "$P"
md5sum "$OVERLAY"/*.py > "$P/overlay_md5.txt" 2>/dev/null || true
sha256sum "$WHEELDIR/$WHEELNAME" > "$P/wheel_sha256.txt"

run_fmt mxfp8 m8qa
run_fmt mxk8v4 mkqa

echo "=== [r212h] report $(date +%H:%M:%S) ==="
python3 - "$R/p647" <<'PY'
import json, os, sys
root = sys.argv[1]
def itl_of(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    itls = [r["itl_ms_p50"] for r in j["requests"] if r["error"] is None]
    return (sum(itls) / len(itls)) if itls else None

ok = True
for fmt in ("m8qa", "mkqa"):
    print(f"r212h {fmt} -- ITL p50 ms/token (launch4 pre-fix vs r212h fixed)")
    for tag in ("c8192_n1", "c16384_n1", "c32768_n1"):
        pre = itl_of(os.path.join(root, "r212a", fmt, tag, f"p139_conc_{tag}.json"))
        fix = itl_of(os.path.join(root, "r212h", fmt, tag, f"p139_conc_{tag}.json"))
        ps = f"{pre:8.1f}" if pre is not None else " MISSING"
        fs = f"{fix:8.1f}" if fix is not None else " MISSING"
        ratio = f"{pre/fix:5.2f}x faster" if (pre and fix) else ""
        print(f" {fmt} {tag:10s} pre {ps}  fixed {fs}  {ratio}")
        if fix is None or pre is None:
            ok = False
        elif fix > pre / 2:
            print(f"  FAIL_{fmt}_{tag}: fix did not improve >=2x"); ok = False
print("R212H_" + ("GATE_PASS" if ok else "GATE_FAIL"))
PY
echo "R212H_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
