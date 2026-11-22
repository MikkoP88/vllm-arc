#!/bin/bash
# r221b_val.sh -- arc7 Phase-8 r221b: COLD-GRAPH A/A of _64 vs _16.
# The r221a finding: the probe cannot reproduce the r220b e2e
# inversion under ANY shape knob (bt 4100/stale/pool/cap extent all
# set, engine row count NCAP=4) -- _64 still wins offline everywhere.
# Surviving hypothesis: HOT-L2 blindness -- the 50x replay re-reads
# the same KV pages (L2-resident) while the engine reads KV cold from
# HBM every layer (weights evict L2 between attention layers); a hot
# probe cannot see LATENCY-HIDING structure, which is exactly _16's
# 4x-higher WG count's advantage (more outstanding loads).
# This leg uses the new P204_COLDMB knob (harness r221b edit): a
# COLDMB=160 device buffer swept in-place between EVERY captured
# call, evicting L2 so each replayed call reads KV cold. The sweep is
# common-mode; the harness reports sweep-only us so this report
# subtracts it for per-call cold numbers.
# VERDICT FRAME: if the _64-vs-16 sign FLIPS cold (or the win
# collapses toward zero), the mechanism is confirmed and the tile
# lever REOPENS with a fix frame (restore latency-hiding under the
# wide tile); if _64 still wins cold by the hot margin, the residual
# is engine-only beyond offline reach and the lever stays closed.
# Cells (m0, wheel 62f234c8, engine-shape env + COLDMB=160):
#   c64/c16 @ {1024, 32768}
# Artifacts -> /root/build/p199/out/p221b_<tag>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r221b

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R221B_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R221B_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
grep -q "ARC_MXKV_P64NATIVE" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { echo "R221B_TREE_MISSING_ROUTE"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R221B_NO_HARNESS"; exit 1; }
grep -q "COLDMB" "$HARNESS" || { echo "R221B_HARNESS_NO_COLD_KNOB"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R221B_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R221B_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R221B_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R221B_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r221b.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r221b] W14 quiesce $(date +%H:%M:%S) ==="
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
  [ "$NATIVE" = "1" ] && P64ENV="-e ARC_MXKV_P64NATIVE=1 -e P204_KVT=64"
  echo "=== [r221b:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281 \
    -e P204_COLDMB=160 \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e ARC_MXKV_VPROBE=0 -e ARC_MXKV_DIRECTBF16=1 \
    $P64ENV \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r221b_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r221b:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m0.json" ]; then
    mv "$P/out/p204_graphtime_m0.json" "$P/out/p221b_${TAG}_$CTX.json"
  else
    echo "R221B_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

run_cell c64 1 1024
run_cell c64 1 32768
run_cell c16 0 1024
run_cell c16 0 32768

echo "=== [r221b] report $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p221b_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None, None
    j = json.load(open(f))
    sw = j.get("_cold_sweep_us")
    return (j.get(fmt) or {}).get("native_auto") or {}, sw

# r221a hot anchors (engine shape, ncap32, same wheel)
HOT = {("mxfp8", 1024): (-53.1, 374.5, 798.3),
       ("mxfp8", 32768): (-9.6, 682.0, 754.3),
       ("mxfp4", 1024): (-55.5, 277.6, 623.6),
       ("mxfp4", 32768): (-17.2, 503.6, 608.5),
       ("mxk8v4", 1024): (-56.2, 333.9, 762.2),
       ("mxk8v4", 32768): (-13.0, 603.9, 694.3)}
print("r221b COLD-GRAPH A/A (COLDMB=160, engine shape) -- us/call, sweep-subtracted")
print(f" {'fmt':>6} {'ctx':>6} | {'c64':>8} | {'c16':>8} | {'64vs16':>7} | {'hot 64vs16':>10} | canary")
flips = 0
for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
    for ctx in (1024, 32768):
        c64, s64 = cell("c64", ctx, fmt)
        c16, s16 = cell("c16", ctx, fmt)
        if not c64 or not c16:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        sw = s64 if s64 is not None else s16
        u64 = c64["dev_us"] - (sw or 0.0)
        u16 = c16["dev_us"] - (sw or 0.0)
        d = 100.0 * (u64 - u16) / u16
        hd = HOT.get((fmt, ctx), (None,))[0]
        can = c16.get("canary")
        ok = can is not None and can <= 2e-3
        if not ok:
            fail = 1
        if hd is not None and d > 0 and hd < 0:
            flips += 1
        print(f" {fmt:>6} {ctx:>6} | {u64:8.1f} | {u16:8.1f} | {d:+6.1f}% | {hd:+9.1f}% | {can} {'OK' if ok else 'FAIL'}")
print(f"SIGN_FLIPS cold-vs-hot: {flips} cell(s)")
print("R221B_" + ("PASS" if not fail else "FAIL"))
PY
grc=$?
echo "R221B_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
