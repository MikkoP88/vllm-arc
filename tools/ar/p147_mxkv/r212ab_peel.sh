#!/bin/bash
# r212ab_peel.sh -- arc7 Phase-8 r212ab: COMPILE-TIME-PARITY PEELED
# K-DECODE PIPELINE in the SHIPPING e2m1 body only (the r212z overlap
# minus the runtime-parity operand demotion that collapsed the
# region). Three same-wheel cells at 32K:
#   m0  = SHIPPING body (pipelined e2m1; m8 mxfp8 oracle rides here)
#   m11 = K const + V light   (body<true>; stability anchor)
#   m17 = K real + V light    (body<true>; k_lane = m17-m11)
# HEADLINE: m0 < 900 = PIPELINE WON (vs r212aa 971.6; the ~490 us
#           overlap partially lands); m0 in 940-1010 = wash or mild
#           win; m0 > 1030 = pressure/spill class (revert or halve).
# m0's canary is the byte-level net. m11 > 800 = instrumented-body
# collapse. m8 gates the e4m3 lane vs the r212aa baseline 1193.7
# (1-sample; fail beyond +8%).
# Artifacts -> /root/build/p199/out/p212ab_{m0,m11,m17}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212ab

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212AB_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212ab PEELED-BY-2" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212AB_TREE_MISMATCH (p212ab not applied?)"; exit 1; }
grep -q "r212x NESTED-FORM GUARD" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212AB_WALK_TREE_MISMATCH (p212x2 guard missing?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212AB_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212AB_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212AB_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212AB_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212AB_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212ab.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212ab] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212ab:vp$VP:c32768] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=32768 -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=$VP -e ARC_MXKV_DIRECTBF16=1 \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212ab_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212ab:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212ab_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212AB_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17

echo "=== [r212ab] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212ab_{name}_32768.json")
    if not os.path.exists(f):
        print(f"{name.upper()}_MISSING"); sys.exit(1)
    j = json.load(open(f))
    c = (j.get(fmt) or {}).get("native_auto") or {}
    if not c or c.get("dev_us") is None:
        print(f"{name.upper()}_NO_CELL"); sys.exit(1)
    return c
m0, m11, m17 = (cell(n) for n in ("m0", "m11", "m17"))
m8 = cell("m0", "mxfp8")
can = m0.get("canary")
if can not in (0, 0.0):
    print(f"CANARY_FAIL m0={can} -- PEELED PIPELINE CORRUPTED OUTPUT;"
          " check the parity mapping (prologue + lookahead sites) and"
          " revert"); fail = 1
v0, v11, v17 = m0["dev_us"], m11["dev_us"], m17["dev_us"]
w8 = m8.get("dev_us") or float("nan")
k_lane = v17 - v11
print("r212ab peeled K pipeline (mxfp4 32K, same wheel)")
print(f"  m0  (SHIPPING, pipelined e2m1) {v0:7.1f}   (r212aa 971.6;"
      " r212u-era 718.0)")
print(f"  m11 (K const + V light, instr) {v11:7.1f}   (r212aa 414.2)")
print(f"  m17 (K real + V light, instr)  {v17:7.1f}   k_lane {k_lane:+7.1f}"
      "   (band 595-609)")
m8dev = w8 / 1193.7 - 1.0
m8ok = m8dev <= 0.08
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212aa 1193.7"
      " -- 1-sample baseline)")
if v11 > 800.0:
    print("FORK: INSTRUMENTED-BODY COLLAPSE -- m11 ballooned; k_lane"
          " invalid; the wrapper disturbed the body<true> layout")
if v0 < 900.0:
    print("HEADLINE: PIPELINE WON -- the static-parity overlap lands"
          " (r212z's ~490 us mechanism without the demotion); bank,"
          " then tune (Q-copy prologue, wider lookahead)")
elif v0 <= 1010.0:
    print("HEADLINE: WASH-TO-MILD -- overlap roughly offset by"
          " pressure/dice; disasm for spills before iterating")
else:
    print("HEADLINE: PRESSURE/SPILL CLASS -- register cost dominates"
          " (2x32-reg B fragments); revert or halve the buffer")
if m8dev < -0.01:
    print("M8_NEW_BEST again -- e4m3 lane improved (wrapper side"
          " effect); verify canary")
if not m8ok:
    print("M8_STABILITY_FAIL (e4m3 lane regressed past +8% vs the"
          " r212aa baseline -- the wrapper disturbed it)"); fail = 1
print("R212AB_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212AB_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
