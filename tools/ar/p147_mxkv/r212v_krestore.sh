#!/bin/bash
# r212v_krestore.sh -- arc7 Phase-8 r212v: K-RESTORE RUNG.
# Three same-wheel cells at 32K:
#   m0  = fast full (K dtile real + V vslm real; pool reference)
#   m11 = K CONST + V light (the old m-series baseline)
#   m17 = K REAL + V light  (the missing cell)
#   k_lane = m17 - m11  (the K walk's real cost; V light both arms)
#   v_pool = m0 - m17   (the V fast-light delta; K real both arms)
#   k_lane + v_pool = m0 - m11 (telescoping check)
#   k_lane > 0.7*pool -> K DOMINANT -> dtile surgery next
#   v_pool > 0.7*pool -> V dominant -> V-side hunt continues
# m8 rides the m0 cell as the byte-identity oracle.
# Artifacts -> /root/build/p199/out/p212v_{m0,m11,m17}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212v

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212V_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212v: 17 is EXEMPT" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212V_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212V_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212V_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212V_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212V_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212V_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212v.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212v] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212v:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212v_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212v:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212v_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212V_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17

echo "=== [r212v] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212v_{name}_32768.json")
    if not os.path.exists(f):
        print(f"{name.upper()}_MISSING"); sys.exit(1)
    j = json.load(open(f))
    c = (j.get(fmt) or {}).get("native_auto") or {}
    if not c or c.get("dev_us") is None:
        print(f"{name.upper()}_NO_CELL"); sys.exit(1)
    return c
m0, m11, m17 = (cell(n) for n in ("m0", "m11", "m17"))
m8 = cell("m0", "mxfp8")
if m0.get("canary") not in (0, 0.0):
    print(f"CANARY_FAIL m4 m0={m0.get('canary')}"); fail = 1
v0, v11, v17 = m0["dev_us"], m11["dev_us"], m17["dev_us"]
w8 = m8.get("dev_us") or float("nan")
pool = v0 - v11
k_lane = v17 - v11
v_pool = v0 - v17
print("r212v k-restore rung (mxfp4 32K, same wheel)")
print(f"  m0  (K real + V real)      {v0:7.1f}")
print(f"  m11 (K const + V light)    {v11:7.1f}")
print(f"  m17 (K real + V light)     {v17:7.1f}")
print(f"  pool (m0-m11):             {pool:7.1f} us")
print(f"  k_lane (m17-m11):          {k_lane:7.1f} us  ({100*k_lane/pool:.0f}% of pool)")
print(f"  v_pool (m0-m17):           {v_pool:7.1f} us  ({100*v_pool/pool:.0f}% of pool)")
telesc = abs((k_lane + v_pool) - pool)
print(f"  telescoping check:         {telesc:7.2f} us drift"
      + (" OK" if telesc < 1.0 else " FAIL"))
if telesc >= 1.0:
    print("TELESCOPY_FAIL"); fail = 1
if k_lane > 0.7 * pool:
    print("FORK: K DOMINANT -- the pool is mostly the K walk;"
          " next = dtile surgery (u32/SWAR/window) with V at floor")
elif v_pool > 0.7 * pool:
    print("FORK: V DOMINANT -- the V fast-vs-light delta carries the"
          " pool after all; the V-side hunt continues with K pinned")
else:
    print("FORK: SPLIT -- attack the bigger half first")
m8dev = abs(w8 / 1320.5 - 1.0)
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)"
      + (" OK" if m8dev <= 0.06 else " FAIL"))
if m8dev > 0.06:
    print("M8_STABILITY_FAIL"); fail = 1
print("R212V_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212V_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
