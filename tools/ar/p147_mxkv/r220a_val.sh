#!/bin/bash
# r220a_val.sh -- arc7 Phase-8 r220a: offline validation of the TRUE
# _64 policy at the engine geometry (D=256 HREP=6 page 64) through the
# ARC_MXKV_P64NATIVE env route, vs the shipping _16 sub-tile frame on
# the SAME wheel. The r220a3 surgery (wheel 62f234c8, after r220a/r220c
# diagnosed the double-bake): copies/prefetches slice WG-linear
# (get_slice(thr_id)) over UNSLICED tiles -- the bwidth TiledCopy
# builders compose sg_to_vnk into svB, so partition coords are
# WG-canonical and the scale-row anchors self-resolve with NO bake;
# wi16 = thr_id % sg_size ONLY at fragment-tv decode sites (scatter pc,
# vslm kr); sg_row_off ONLY at the SLM V image's WG frame (scatter
# destination row, vslm slm base). The m0 canary here is weak
# (graph-vs-eager self-reference -- r220c_verify.py is the value
# instrument); this ladder's product is the TIMING table + the s16
# same-wheel control proving behavior-identity on the shipping route.
#   n64  : ARC_MXKV_P64NATIVE=1, m0 @ {1024, 32768} -- canary gate +
#          fixed term; n64m{2,6,8} @ 32768 -- lane decomposition
#   s16  : env unset, m0 @ {1024, 32768} -- same-wheel control: the
#          band edits must be behavior-identical on the shipping route
# Artifacts -> /root/build/p199/out/p220a_<tag>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zc}
CNAME=arc-r220a

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R220A_NO_WHEEL"; exit 1; }
GOTMD5=$(md5sum "$WHEEL" | awk '{print $1}')
echo "wheel md5 $GOTMD5"
if [ -n "${WHEELMD5:-}" ] && [ "$GOTMD5" != "$WHEELMD5" ]; then
  echo "R220A_WHEEL_MISMATCH got=$GOTMD5 want=$WHEELMD5 (build not finished?)"
  exit 1
fi
grep -q "ARC_MXKV_P64NATIVE" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_xe2.cpp \
  || { echo "R220A_TREE_MISSING_ROUTE (edits not shipped?)"; exit 1; }
grep -q "r220a" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R220A_TREE_MISSING_BAND_LAWS"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R220A_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R220A_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R220A_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R220A_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R220A_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r220a.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r220a] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag p64native ctx probe
  local TAG=$1
  local NATIVE=$2
  local CTX=$3
  local PROBE=$4
  docker rm -f $CNAME >/dev/null 2>&1
  local P64ENV=""
  # r212j tile-unit law: the harness work-list partitions in P204_KVT
  # rows; it must MATCH the kernel's walk — 64 under the native policy,
  # 16 under the shipping sub-tile frame.
  [ "$NATIVE" = "1" ] && P64ENV="-e ARC_MXKV_P64NATIVE=1 -e P204_KVT=64"
  echo "=== [r220a:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e ARC_MXKV_VPROBE=$PROBE -e ARC_MXKV_DIRECTBF16=1 \
    $P64ENV \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r220a_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r220a:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p220a_${TAG}_$CTX.json"
  else
    echo "R220A_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

# correctness + fixed term: m0 fulls, native and control
run_cell n64 1 1024 0
run_cell n64 1 32768 0
run_cell s16 0 1024 0
run_cell s16 0 32768 0
# lane telemetry under the native tile (timing decomposition only --
# probe-mode canaries are vacuous by the harness law)
run_cell n64m2 1 32768 2
run_cell n64m6 1 32768 6
run_cell n64m8 1 32768 8

echo "=== [r220a] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p220a_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}

print("r220a canary gate (m0 cells -- the probe cells' canary is vacuous)")
for tag in ("n64", "s16"):
    for ctx in (1024, 32768):
        for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
            c = cell(tag, ctx, fmt)
            if not c or c.get("dev_us") is None:
                print(f" {tag:>3} c{ctx:<6} {fmt:>6} MISSING"); fail = 1; continue
            can = c.get("canary")
            ok = can is not None and can <= 2e-3
            print(f" {tag:>3} c{ctx:<6} {fmt:>6} canary {can} {'OK' if ok else 'FAIL'}")
            if not ok:
                fail = 1

print("r220a timing -- us/call, native _64 tile vs _16 sub-tile (same wheel)")
BASE = {("mxfp4", 1024): 594.5, ("mxfp4", 32768): 1028.3,
        ("mxfp8", 1024): 658.3, ("mxfp8", 32768): 1359.0}
print(f" {'fmt':>6} {'ctx':>6} | {'_64':>8} | {'_16':>8} | {'delta':>7} | vs r212m3 base")
for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
    for ctx in (1024, 32768):
        cn = cell("n64", ctx, fmt)
        cs = cell("s16", ctx, fmt)
        if not cn or not cs:
            continue
        d = 100.0 * (cn["dev_us"] - cs["dev_us"]) / cs["dev_us"]
        b = BASE.get((fmt, ctx))
        bs = f"{100.0 * (cn['dev_us'] - b) / b:+6.1f}%" if b else "     n/a"
        print(f" {fmt:>6} {ctx:>6} | {cn['dev_us']:8.1f} | {cs['dev_us']:8.1f} | {d:+6.1f}% | {bs}")

print("r220a lane decomposition at 32K under the native tile (n64 cells)")
for fmt in ("mxfp8", "mxfp4", "mxk8v4"):
    cs = {}
    for probe, tag in (("0", "n64"), ("2", "n64m2"), ("6", "n64m6"), ("8", "n64m8")):
        c = cell(tag, 32768, fmt)
        if c and c.get("dev_us") is not None:
            cs[probe] = c["dev_us"]
    if len(cs) == 4:
        full, m2, m6, m8 = (cs[p] for p in ("0", "2", "6", "8"))
        print(f" {fmt:>6}: K {full-m6:7.1f}  Vg {full-m2:7.1f}  floor {m8:7.1f}  full {full:7.1f}")
    else:
        print(f" {fmt:>6}: probe cells missing ({sorted(cs)})")
print("R220A_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R220A_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
