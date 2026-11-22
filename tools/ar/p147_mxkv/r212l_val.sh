#!/bin/bash
# r212l_val.sh -- arc7 Phase-8 r212l: validate the invariant-hoist kernel
# iteration (V-lane scale-line/row hoist + K static window law) on the
# r212l wheel. Cells: probe 0 (full, canary-gated) x ctx {1024,8192,
# 32768} + probes {2,3,6,8} x ctx 32768 for the slice re-decomposition,
# FMTS=mxfp4,mxfp8, SPLITS=auto, engine geometry, graph 32x50.
# Baselines = banked build21 r212k/r212j2 numbers (hardcoded below for
# the delta table; binary-layout dice law: +-6% on untouched lanes).
# Artifacts -> /root/build/p199/out/p212l_m<PROBE>_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212l

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212L_NO_WHEEL"; exit 1; }
grep -q "r212l STATIC WINDOW LAW" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212L_WHEEL_TREE_MISMATCH (source reverted?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212L_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212L_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212L_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212L_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212L_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212l.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212l] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # probe ctx
  local PROBE=$1
  local CTX=$2
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212l:m$PROBE:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=$PROBE \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212l_m${PROBE}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212l:m$PROBE:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212l_m${PROBE}_$CTX.json"
  else
    echo "R212L_MISSING_ARTIFACT_m${PROBE}_$CTX"
  fi
}

for CTX in 1024 8192 32768; do
  run_cell 0 $CTX
done
for PROBE in 2 3 6 8; do
  run_cell $PROBE 32768
done

echo "=== [r212l] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
# build21 baselines (banked r212k/r212j2)
BASE_FULL = {("mxfp4", 1024): 598.4, ("mxfp4", 8192): 423.0, ("mxfp4", 32768): 1040.3,
             ("mxfp8", 1024): 643.1, ("mxfp8", 8192): 535.1, ("mxfp8", 32768): 1344.9}
BASE_32K = {"mxfp4": {"K": 357.6, "Vg": 392.1, "Vs": 82.1, "fl": 142.9},
            "mxfp8": {"K": 407.2, "Vg": 317.1, "Vs": 390.9, "fl": 163.7}}
def cell(probe, ctx, fmt):
    f = os.path.join(out, f"p212l_m{probe}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
print("r212l full cells (probe 0) -- us/call vs build21 baseline")
print(f" {'fmt':>6} {'ctx':>6} | {'new':>8} | {'base':>8} | {'delta':>7}")
for fmt in ("mxfp4", "mxfp8"):
    for ctx in (1024, 8192, 32768):
        c = cell(0, ctx, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        b = BASE_FULL[(fmt, ctx)]
        d = 100.0 * (c["dev_us"] - b) / b
        print(f" {fmt:>6} {ctx:>6} | {c['dev_us']:8.1f} | {b:8.1f} | {d:+6.1f}%")
        if c.get("canary") is None or c["canary"] > 2e-3:
            print(f"          CANARY_FAIL {c.get('canary')}"); fail = 1
print("r212l slice re-decomposition at 32K vs build21")
for fmt in ("mxfp4", "mxfp8"):
    cs = {}
    ok = True
    for probe in ("0", "2", "3", "6", "8"):
        c = cell(probe, 32768, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt}: MISSING m{probe}"); fail = 1; ok = False; break
        cs[probe] = c["dev_us"]
    if not ok:
        continue
    full, c2, c3, c6, c8 = (cs[p] for p in ("0", "2", "3", "6", "8"))
    b = BASE_32K[fmt]
    print(f" {fmt}: K {full-c6:6.1f} (base {b['K']:6.1f})"
          f"  Vg {full-c2:6.1f} (base {b['Vg']:6.1f})"
          f"  Vs {c2-c3:6.1f} (base {b['Vs']:6.1f})"
          f"  floor {c8:6.1f} (base {b['fl']:6.1f})")
print("R212L_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212L_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
