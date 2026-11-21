#!/bin/bash
# r212w_kdecomp.sh -- arc7 Phase-8 r212w: K-DECOMPOSITION RUNGS.
# Five same-wheel cells at 32K:
#   m0  = K real + V real   (pool reference; m8 oracle rides here)
#   m11 = K const + V light (the m-series baseline)
#   m17 = K real + V light  (the K lane)
#   m18 = K real + reorder SEVERED (dtile runs; tSrK const after)
#   m20 = K real + dtile light tail + REAL reorder
#   reorder = m17-m18 ; (a)+(b) = m18-m11 ; (a)+(c) = m20-m11
#   a = (m20-m11) - (m17-m18) ; b = (m18-m11) - a
#   (a) staging/loads/extracts, (b) decode, (c) reorder+interleave
# The m8 oracle gates the wheel's absolutes (r212v FAILed +33.8%).
# Artifacts -> /root/build/p199/out/p212w_{m0,m11,m17,m18,m20}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212w

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212W_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212w: sever ONLY the reorder" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212W_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212W_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212W_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212W_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212W_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212W_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212w.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212w] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212w:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212w_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212w:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212w_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212W_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17
run_cell 18
run_cell 20

echo "=== [r212w] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212w_{name}_32768.json")
    if not os.path.exists(f):
        print(f"{name.upper()}_MISSING"); sys.exit(1)
    j = json.load(open(f))
    c = (j.get(fmt) or {}).get("native_auto") or {}
    if not c or c.get("dev_us") is None:
        print(f"{name.upper()}_NO_CELL"); sys.exit(1)
    return c
m0, m11, m17, m18, m20 = (cell(n) for n in ("m0", "m11", "m17", "m18", "m20"))
m8 = cell("m0", "mxfp8")
if m0.get("canary") not in (0, 0.0):
    print(f"CANARY_FAIL m4 m0={m0.get('canary')}"); fail = 1
v0, v11, v17, v18, v20 = (c["dev_us"] for c in (m0, m11, m17, m18, m20))
w8 = m8.get("dev_us") or float("nan")
pool = v0 - v11
k_lane = v17 - v11
reorder = v17 - v18
ab = v18 - v11
ac = v20 - v11
a = ac - reorder
b = ab - a
print("r212w k-decomposition (mxfp4 32K, same wheel)")
print(f"  m0  (K real + V real)          {v0:7.1f}")
print(f"  m11 (K const + V light)        {v11:7.1f}")
print(f"  m17 (K real + V light)         {v17:7.1f}")
print(f"  m18 (K real, reorder severed)  {v18:7.1f}")
print(f"  m20 (K light tail + reorder)   {v20:7.1f}")
print(f"  pool (m0-m11):                 {pool:7.1f} us")
print(f"  k_lane (m17-m11):              {k_lane:7.1f} us")
print(f"  (c) reorder share (m17-m18):   {reorder:7.1f} us  ({100*reorder/max(k_lane,1e-9):.0f}% of k_lane)")
print(f"  (a)+(b) loads+decode (m18-m11):{ab:7.1f} us")
print(f"  (a)+(c) staging+reorder:       {ac:7.1f} us")
print(f"  (a) staging/loads/extracts:    {a:7.1f} us")
print(f"  (b) decode:                    {b:7.1f} us")
m8dev = abs(w8 / 1320.5 - 1.0)
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)"
      + (" OK" if m8dev <= 0.06 else " FAIL (wheel dice-regressed; absolutes directional only)"))
if m8dev <= 0.06:
    if reorder > 0.5 * k_lane:
        print("FORK: REORDER DOMINANT -- delete the K reorder the way"
              " r200b deleted the V one (write the MMA layout directly"
              " from dtile / SLM-stage); biggest single lever")
    elif a > 0.5 * k_lane:
        print("FORK: STAGING DOMINANT -- the K byte loads/window path"
              " is the wall; coalesce/SLM-stage/eliminate the walk")
    elif b > 0.5 * k_lane:
        print("FORK: DECODE DOMINANT -- the dtile decode itself; deeper"
              " u32/SWAR surgery on the K fast loop")
    else:
        print("FORK: SPLIT -- attack the bigger components in order")
else:
    print("M8_STABILITY_FAIL"); fail = 1
    print("FORK: re-roll the dice (same tree, new build) before acting"
          " on absolutes; the layout sensitivity itself is suspect")
print("R212W_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212W_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
