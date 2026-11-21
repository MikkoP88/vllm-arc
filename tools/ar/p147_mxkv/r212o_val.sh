#!/bin/bash
# r212o_val.sh -- arc7 Phase-8 r212o: validate the loop-split wheel
# (per-element runtime branches hoisted out of both e2m1 walks + dead
# e2m1 sc prep removed; e4m3 source byte-identical) with the FULL
# telemetry-law shape:
#   - probes {0,2,3,6,8,9} x 32768 -> lane table + fresh probe-9
#     residual on the NEW wheel (r212n baselines shown; cross-wheel =
#     dice-polluted, the honest instruments are the same-wheel dbf0
#     oracle and the m8 stability check)
#   - m0 x {1024, 8192} fulls
#   - dbf0 x 32768: DIRECTBF16=0 -- with the split, the slow loop IS
#     the float path; delta on mxfp4 isolates the split's fast gain,
#     equality on mxfp8 proves the e4m3 byte-identity
# Gate: canaries on m0/dbf0 cells only (probe cells vacuous), no
# missing artifacts, m8 full within +-6% of r212n's 1320.7 (the e4m3
# paths are byte-identical source; beyond the dice band = investigate
# before any e2e use of this wheel).
# Artifacts -> /root/build/p199/out/p212o_{m0,dbf0,m2,m3,m6,m8,m9}_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212o

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212O_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212o LOOP SPLIT" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212O_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212O_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212O_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212O_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212O_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212O_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212o.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212o] W14 quiesce $(date +%H:%M:%S) ==="
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
  echo "=== [r212o:$TAG:c$CTX] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212o_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212o:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212o_${TAG}_$CTX.json"
  else
    echo "R212O_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

for PROBE in 0 2 3 6 8 9; do
  run_cell m$PROBE 32768 $PROBE 1
done
run_cell m0 1024 0 1
run_cell m0 8192 0 1
run_cell dbf0 32768 0 0

echo "=== [r212o] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p212o_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
# r212n baselines (the m3 wheel): cross-wheel display only, +-6% dice band
N_FULL = {"mxfp4": {1024: 399.8, 8192: 306.8, 32768: 749.5},
          "mxfp8": {1024: 633.9, 8192: 525.7, 32768: 1320.7}}
N_LANE = {"mxfp4": {"K": 178.2, "Vg": 196.7, "Vs": 16.7, "fl": 246.6, "m9": 379.0},
          "mxfp8": {"K": 390.6, "Vg": 334.2, "Vs": 364.9, "fl": 171.6, "m9": 588.6}}
print("r212o fulls vs r212n (cross-wheel, dice band +-6%) + canaries")
for fmt in ("mxfp4", "mxfp8"):
    for ctx in (1024, 8192, 32768):
        c = cell("m0", ctx, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        b = N_FULL[fmt][ctx]
        d = 100.0 * (c["dev_us"] - b) / b
        flag = " DICE" if abs(d) > 6.0 else ""
        print(f" {fmt:>6} {ctx:>6} | {c['dev_us']:8.1f} | base {b:8.1f} | {d:+6.1f}%{flag} | canary {c.get('canary')}")
        if c.get("canary") is None or c["canary"] > 2e-3:
            print(f"          CANARY_FAIL {fmt} m0 c{ctx}"); fail = 1
        if fmt == "mxfp8" and ctx == 32768 and abs(d) > 6.0:
            print("          M8_STABILITY_FAIL (e4m3 source is byte-identical;"
                  " beyond-dice drift needs investigation)"); fail = 1
print("r212o lane table at 32K (vs r212n cross-wheel) + probe-9 residual")
for fmt in ("mxfp4", "mxfp8"):
    cs = {}
    ok = True
    for probe in ("0", "2", "3", "6", "8", "9"):
        c = cell(f"m{probe}", 32768, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt}: MISSING m{probe}"); fail = 1; ok = False; break
        cs[probe] = c["dev_us"]
    if not ok:
        continue
    full, c2, c3, c6, c8, m9 = (cs[p] for p in ("0", "2", "3", "6", "8", "9"))
    b = N_LANE[fmt]
    k, vg, vs = full - c6, full - c2, c2 - c3
    print(f" {fmt:>6}: K {k:6.1f} (n {b['K']:6.1f})  Vg {vg:6.1f} (n {b['Vg']:6.1f})"
          f"  Vs {vs:5.1f} (n {b['Vs']:5.1f})  floor {c8:6.1f} (n {b['fl']:6.1f})"
          f"  m9 {m9:6.1f} (n {b['m9']:6.1f})  dec_resid {full-m9:6.1f}  pool {m9-c8:6.1f}")
print("r212o same-wheel dtype oracle (dbf0) at 32K")
for fmt in ("mxfp4", "mxfp8"):
    c0 = cell("dbf0", 32768, fmt)
    c1 = cell("m0", 32768, fmt)
    if not c0 or c0.get("dev_us") is None or not c1 or c1.get("dev_us") is None:
        print(f" {fmt:>6} | MISSING"); fail = 1; continue
    g = 100.0 * (c0["dev_us"] - c1["dev_us"]) / c0["dev_us"]
    print(f" {fmt:>6} | dbf0 {c0['dev_us']:8.1f} | fast {c1['dev_us']:8.1f} | gain {g:6.1f}% | canary(dbf0) {c0.get('canary')}")
    if c0.get("canary") is None or c0["canary"] > 2e-3:
        print(f"          CANARY_FAIL dbf0 {fmt}"); fail = 1
print("note: probe-cell canaries vacuous (not computed in probe mode)")
print("R212O_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212O_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
