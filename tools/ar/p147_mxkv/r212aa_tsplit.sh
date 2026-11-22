#!/bin/bash
# r212aa_tsplit.sh -- arc7 Phase-8 r212aa: TEMPLATE-SPLIT attractor
# escape. operator() -> thin dispatcher (vlane_probe==0 -> body<false>
# = SHIPPING, zero probe branches; else body<true> = the probe
# apparatus). Three same-wheel cells at 32K:
#   m0  = SHIPPING body (vprobe 0; m8 mxfp8 oracle rides here)
#   m11 = K const + V light   (body<true>; baseline)
#   m17 = K real + V light    (body<true>; k_lane = m17-m11)
# HEADLINE: m8 in band (<= +-6% of 1320.5) = attractor ESCAPED ->
#           m0 becomes the trustworthy shipping absolute (vs r212u
#           clean 718.0) and k_lane the calibrated K-lane cost; m8
#           still +34 = the basin survived the split (escalate:
#           structure).
# m0's canary is the byte-level net. m11 > 800 = the INSTRUMENTED
# body collapsed (k_lane invalid again).
# Artifacts -> /root/build/p199/out/p212aa_{m0,m11,m17}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212aa

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212AA_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212aa TEMPLATE-SPLIT" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212AA_TREE_MISMATCH (p212aa not applied?)"; exit 1; }
grep -q "r212x NESTED-FORM GUARD" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212AA_WALK_TREE_MISMATCH (p212x2 guard missing?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212AA_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212AA_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212AA_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212AA_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212AA_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212aa.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212aa] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212aa:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212aa_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212aa:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212aa_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212AA_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17

echo "=== [r212aa] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212aa_{name}_32768.json")
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
    print(f"CANARY_FAIL m0={can} -- SHIPPING BODY CORRUPTED;"
          " the split must be byte-neutral on vprobe 0; revert and"
          " diff"); fail = 1
v0, v11, v17 = m0["dev_us"], m11["dev_us"], m17["dev_us"]
w8 = m8.get("dev_us") or float("nan")
k_lane = v17 - v11
print("r212aa template-split (mxfp4 32K, same wheel)")
print(f"  m0  (SHIPPING body, K+V real) {v0:7.1f}   (r212y attractor"
      " 1006.6; r212u clean 718.0)")
print(f"  m11 (K const + V light, instr) {v11:7.1f}   (band ~414)")
print(f"  m17 (K real + V light, instr)  {v17:7.1f}   k_lane {k_lane:+7.1f}")
# One-sided (post-hoc correction of the first run's misfire): only a
# REGRESSION beyond +6% is wheel instability (the attractor signature).
# A canary-clean improvement on the low side is a genuine win.
m8dev = w8 / 1320.5 - 1.0
m8ok = m8dev <= 0.06
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)")
if m8dev < -0.01:
    print("M8_NEW_BEST -- shipping e4m3 layout beats the r212q-era"
          " reference; canary must gate the bytes")
if v11 > 800.0:
    print("FORK: INSTRUMENTED-BODY COLLAPSE -- m11 ballooned vs the 414"
          " band; k_lane invalid (two regressed cells differencing);"
          " the body<true> instantiation took a bad layout")
if m8ok:
    print("HEADLINE: ATTRACTOR ESCAPED -- the shipping body (m0/m8) is"
          " clean; m0 is the trustworthy absolute, k_lane the"
          " calibrated K-lane cost; bank, then attack the K chain with"
          " compile-time parity on clean absolutes")
else:
    print("HEADLINE: BASIN SURVIVED THE SPLIT -- m0/m8 still taxed;"
          " escalate structure (peel the D-loop for compile-time"
          " parity) or noinline firewalls around the GEMM regions")
if can not in (0, 0.0):
    print("FORK: CORRUPTION on a byte-neutral split -- stop and diff"
          " the two instantiations' arguments")
if not m8ok:
    print("M8_STABILITY_FAIL (regression side only)"); fail = 1
print("R212AA_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212AA_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
