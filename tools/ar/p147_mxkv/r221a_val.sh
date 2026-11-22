#!/bin/bash
# r221a_val.sh -- arc7 Phase-8 r221a: ENGINE-SHAPE offline A/A of the
# _64 tile vs _16. The r220b/r220d e2e verdict inverted the r220a
# offline win (+13.4% itl @32K vs -11.2% per call); the r220
# correction (native_decode.py) rules out cross-call SLM contention
# (16 SEQUENTIAL layer-calls) and localizes the probe's blindness to
# the ENGINE CALL'S SHAPE: bt_width 4100 tables, stale beyond-fill
# entries, pool over-allocation 11281 blocks, capacity-priced extent
# with early-exit, stride 544. The r212d diagnostics already built the
# knobs -- r220a set only MAXSK+ESHAPE. This leg sets ALL of them and
# asks two questions:
#   Q1 (shape cost): does _16 at engine shape slow vs the r220a
#       tight-shape bank (s16 @32K = 772.1 us)? The delta localizes
#       table-indirection/early-exit/stride cost on the SHIPPING route.
#   Q2 (mechanism): does the _64 penalty REPRODUCE offline once the
#       shape matches? YES -> cheap kernel iteration instrument (fix
#       without boots); NO -> the residual is engine-only (graph
#       replay mix / inter-layer L2 state) and offline work on this
#       lever stops.
# Cells (m0, wheel 62f234c8, all engine-shape env):
#   e64/e16 @ {1024, 32768}
# Artifacts -> /root/build/p199/out/p221a_<tag>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r221a

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R221A_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R221A_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
grep -q "ARC_MXKV_P64NATIVE" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { echo "R221A_TREE_MISSING_ROUTE"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R221A_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R221A_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R221A_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R221A_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R221A_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r221a.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r221a] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag p64native ctx
  local TAG=$1
  local NATIVE=$2
  local CTX=$3
  docker rm -f $CNAME >/dev/null 2>&1
  local P64ENV=""
  # r212j tile-unit law: the harness work-list partitions in P204_KVT
  # rows matching the kernel's walk -- 64 native, 16 shipping.
  [ "$NATIVE" = "1" ] && P64ENV="-e ARC_MXKV_P64NATIVE=1 -e P204_KVT=64"
  echo "=== [r221a:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=${R221A_NCAP:-32} -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281 \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e ARC_MXKV_VPROBE=0 -e ARC_MXKV_DIRECTBF16=1 \
    $P64ENV \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r221a_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r221a:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m0.json" ]; then
    mv "$P/out/p204_graphtime_m0.json" "$P/out/p221a_${TAG}_n${R221A_NCAP:-32}_$CTX.json"
  else
    echo "R221A_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

run_cell e64 1 1024
run_cell e64 1 32768
run_cell e16 0 1024
run_cell e16 0 32768

echo "=== [r221a] report $(date +%H:%M:%S) ==="
python3 - "$P/out" "${R221A_NCAP:-32}" <<'PY'
import json, sys, os
out = sys.argv[1]
ncap = int(sys.argv[2])
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p221a_{tag}_n{ncap}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}

# r220a tight-shape anchors (same wheel, MAXSK=cap+ESHAPE only)
TIGHT = {("mxfp8", 1024): 698.5, ("mxfp8", 32768): 772.1,
         ("mxfp4", 1024): 524.3, ("mxfp4", 32768): 603.2,
         ("mxk8v4", 1024): 662.0, ("mxk8v4", 32768): 718.6}
print(f"r221a ENGINE-SHAPE A/A ncap={ncap} -- us/call (bt 4100, stale pad, pool 11281, cap extent)")
print(f" {'fmt':>6} {'ctx':>6} | {'e64':>8} | {'e16':>8} | {'64vs16':>7} | {'e16 vs tight':>12} | canary")
for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
    for ctx in (1024, 32768):
        c64 = cell("e64", ctx, fmt)
        c16 = cell("e16", ctx, fmt)
        if not c64 or not c16:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        d = 100.0 * (c64["dev_us"] - c16["dev_us"]) / c16["dev_us"]
        t = TIGHT.get((fmt, ctx))
        ts = f"{100.0 * (c16['dev_us'] - t) / t:+11.1f}%" if t else "          n/a"
        can = c16.get("canary")
        ok = can is not None and can <= 2e-3
        if not ok:
            fail = 1
        print(f" {fmt:>6} {ctx:>6} | {c64['dev_us']:8.1f} | {c16['dev_us']:8.1f} | {d:+6.1f}% | {ts} | {can} {'OK' if ok else 'FAIL'}")
print("R221A_" + ("PASS" if not fail else "FAIL"))
PY
grc=$?
echo "R221A_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
