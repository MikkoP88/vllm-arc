#!/bin/bash
# r212m3_val.sh -- arc7 Phase-8 r212m3: validate the e2m1-gated
# direct-bf16 (e4m3 reverted to the r212l float path) with
# COMPREHENSIVE per-lane telemetry -- the r212m/r212m2 mxfp8
# float-path regression was inferred, not lane-attributed; this leg
# captures the root issue end-to-end:
#   - probes {0,2,3,6,8} x 32768 -> K_chain / V_gather / V_stage /
#     floor decomposition per format, vs the banked r212k/r212l
#     slice baselines (which lane moved and by how much)
#   - m0 x {1024, 8192} fulls (cross-wheel table continuity)
#   - dbf0 x 32768: with r212m3, DIRECTBF16 touches ONLY the e2m1
#     lanes -- mxfp8 dbf0 == m0 byte-equivalent code path; timing
#     equality PROVES the e4m3 revert is clean, delta on mxfp4
#     isolates the e2m1 fast-path gain
# Artifacts -> /root/build/p199/out/p212m3_{m0,dbf0}_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212m3

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212M3_NO_WHEEL"; exit 1; }
grep -q "r212m3 E4M3 REVERT" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212M3_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212M3_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212M3_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212M3_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212M3_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212M3_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212m3.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212m3] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag ctx probe dbf
  local TAG=$1
  local CTX=$2
  local PROBE=$3
  local DBF=$4
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212m3:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=$PROBE -e ARC_MXKV_DIRECTBF16=$DBF \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212m3_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212m3:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212m3_${TAG}_$CTX.json"
  else
    echo "R212M3_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

# comprehensive telemetry: full decomposition at 32K + fulls + oracle
for PROBE in 0 2 3 6 8; do
  run_cell m$PROBE 32768 $PROBE 1
done
run_cell m0 1024 0 1
run_cell m0 8192 0 1
run_cell dbf0 32768 0 0

echo "=== [r212m3] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p212m3_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
print("r212m3 fulls -- us/call vs r212l float-path baseline")
BASE_FULL = {("mxfp4", 1024): 594.5, ("mxfp4", 8192): 420.6, ("mxfp4", 32768): 1028.3,
             ("mxfp8", 1024): 658.3, ("mxfp8", 8192): 541.7, ("mxfp8", 32768): 1359.0}
print(f" {'fmt':>6} {'ctx':>6} | {'new':>8} | {'base':>8} | {'delta':>7} | canary")
for fmt in ("mxfp4", "mxfp8"):
    for ctx in (1024, 8192, 32768):
        c = cell("m0", ctx, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        b = BASE_FULL[(fmt, ctx)]
        d = 100.0 * (c["dev_us"] - b) / b
        print(f" {fmt:>6} {ctx:>6} | {c['dev_us']:8.1f} | {b:8.1f} | {d:+6.1f}% | {c.get('canary')}")
        if c.get("canary") is None or c["canary"] > 2e-3:
            print(f"          CANARY_FAIL {c.get('canary')}"); fail = 1
print("r212m3 per-lane telemetry at 32K vs r212l slices (ROOT-ISSUE TABLE)")
BASE_32K = {"mxfp4": {"K": 365.9, "Vg": 375.9, "Vs": 83.9, "fl": 139.5},
            "mxfp8": {"K": 470.0, "Vg": 292.3, "Vs": 364.3, "fl": 171.0}}
for fmt in ("mxfp4", "mxfp8"):
    cs = {}
    ok = True
    for probe in ("0", "2", "3", "6", "8"):
        c = cell(f"m{probe}", 32768, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt}: MISSING m{probe}"); fail = 1; ok = False; break
        cs[probe] = c["dev_us"]
    if not ok:
        continue
    full, c2, c3, c6, c8 = (cs[p] for p in ("0", "2", "3", "6", "8"))
    b = BASE_32K[fmt]
    print(f" {fmt}: K {full-c6:6.1f} (base {b['K']:6.1f}, {100.0*((full-c6)-b['K'])/b['K']:+5.1f}%)"
          f"  Vg {full-c2:6.1f} (base {b['Vg']:6.1f}, {100.0*((full-c2)-b['Vg'])/b['Vg']:+5.1f}%)"
          f"  Vs {c2-c3:6.1f} (base {b['Vs']:6.1f})"
          f"  floor {c8:6.1f} (base {b['fl']:6.1f})")
print("r212m3 dtype-split oracle (dbf0) at 32K -- e4m3 revert proof + e2m1 gain")
for fmt in ("mxfp4", "mxfp8"):
    c0 = cell("dbf0", 32768, fmt)
    c1 = cell("m0", 32768, fmt)
    if not c0 or c0.get("dev_us") is None or not c1 or c1.get("dev_us") is None:
        print(f" {fmt:>6} | MISSING"); fail = 1; continue
    g = 100.0 * (c0["dev_us"] - c1["dev_us"]) / c0["dev_us"]
    print(f" {fmt:>6} | dbf0 {c0['dev_us']:8.1f} | fast {c1['dev_us']:8.1f} | gain {g:6.1f}% | canary(dbf0) {c0.get('canary')}")
    if c0.get("canary") is None or c0["canary"] > 2e-3:
        print(f"          CANARY_FAIL dbf0 {c0.get('canary')}"); fail = 1
print("R212M3_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212M3_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
