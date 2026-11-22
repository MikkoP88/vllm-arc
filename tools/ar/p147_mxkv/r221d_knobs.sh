#!/bin/bash
# r221d_knobs.sh -- arc7 Phase-8 r221d: knob isolation of the ENGINE
# SHAPE COST on the SHIPPING _16 route. The r221a side finding: the
# full engine-shape env (BTWIDTH 4100 + BTPAD stale + POOLBLKS 11281
# on top of the r220a base of ESHAPE+MAXSK=cap) costs _16 +14-20% at
# 1K and ~0 at 32K -- a real on-posture cost whose e2e weight sits in
# the short/8K faces. This leg isolates WHICH knob carries it, one at
# a time, to aim a future table/pool surgery.
# Cells (all _16 = P64NATIVE unset, m0, wheel 62f234c8):
#   base : ESHAPE=1 MAXSK=cap (the r220a config -- r221a's "tight")
#   bt   : base + BTWIDTH=4100
#   pad  : base + BTPAD=stale  (needs BTWIDTH to matter: bt 4100)
#   pool : base + POOLBLKS=11281
#   all  : base + bt + pad + pool (the r221a engine shape)
# Faces: ctx {1024, 8192} (the 8K e2e face is where the cost would
# land; 32K measured ~0).
# Artifacts -> /root/build/p199/out/p221d_<tag>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r221d

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R221D_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R221D_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5"
  exit 1
fi
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R221D_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R221D_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R221D_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R221D_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R221D_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r221d.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r221d] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

# tag extra-env
run_cell() {  # tag ctx extraenv
  local TAG=$1
  local CTX=$2
  local EXTRA=$3
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r221d:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e ARC_MXKV_VPROBE=0 -e ARC_MXKV_DIRECTBF16=1 \
    $EXTRA \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r221d_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r221d:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m0.json" ]; then
    mv "$P/out/p204_graphtime_m0.json" "$P/out/p221d_${TAG}_$CTX.json"
  else
    echo "R221D_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

run_cell base 1024 ""
run_cell bt   1024 "-e P204_BTWIDTH=4100"
run_cell pad  1024 "-e P204_BTWIDTH=4100 -e P204_BTPAD=stale"
run_cell pool 1024 "-e P204_POOLBLKS=11281"
run_cell all  1024 "-e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281"
run_cell base 8192 ""
run_cell bt   8192 "-e P204_BTWIDTH=4100"
run_cell pad  8192 "-e P204_BTWIDTH=4100 -e P204_BTPAD=stale"
run_cell pool 8192 "-e P204_POOLBLKS=11281"
run_cell all  8192 "-e P204_BTWIDTH=4100 -e P204_BTPAD=stale -e P204_POOLBLKS=11281"

echo "=== [r221d] report $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p221d_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}

print("r221d engine-shape knob isolation on _16 -- us/call, delta vs base")
print(f" {'fmt':>6} {'ctx':>6} | {'base':>8} | {'+bt':>8} | {'+pad':>8} | {'+pool':>8} | {'all':>8} | dominant")
for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
    for ctx in (1024, 8192):
        cells = {t: cell(t, ctx, fmt) for t in ("base", "bt", "pad", "pool", "all")}
        if any(c is None for c in cells.values()):
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        b = cells["base"]["dev_us"]
        row = " ".join(f"{cells[t]['dev_us']:8.1f}" for t in ("base", "bt", "pad", "pool", "all"))
        deltas = {t: cells[t]["dev_us"] - b for t in ("bt", "pad", "pool", "all")}
        dom = max(deltas, key=lambda t: deltas[t]) if any(d > 0 for d in deltas.values()) else "-"
        can = cells["base"].get("canary")
        ok = can is not None and can <= 2e-3
        if not ok:
            fail = 1
        print(f" {fmt:>6} {ctx:>6} | {row} | {dom:>8} (can {can} {'OK' if ok else 'FAIL'})")
        print(f"        delta | {'':8} | " + " ".join(f"{deltas[t]:+8.1f}" for t in ("bt", "pad", "pool", "all")))
print("R221D_" + ("PASS" if not fail else "FAIL"))
PY
grc=$?
echo "R221D_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
