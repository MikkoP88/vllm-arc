#!/bin/bash
# r212s_ladder.sh -- arc7 Phase-8 r212s: INT-OP PRICING LADDER.
# Four same-wheel cells at 32K:
#   m0  = fast full (byte-identical fast-loop text; same-binary
#         reference for the pool)
#   m11 = light tail (extracts + convert + store)
#   m12 = light tail + 4 dependent int ops/element
#   m13 = light tail + 8 dependent int ops/element
#   slope    = (m13 - m12) / 4 us per scalar int op
#   ops share = slope * 8   (the m0-m11 op-delta class)
#   residual = (m0 - m11) - slope * 8  = memcpy/store/other pool
# Fork: ops share >> residual -> algebraic op reduction (r212t)
#       residual >> ops share -> store/memcpy path (fragment packing)
# m8 rides along on the m0 cell as the byte-identity oracle.
# Artifacts -> /root/build/p199/out/p212s_{m0,m11,m12,m13}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212s

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212S_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212s INT-OP LADDER" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212S_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212S_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212S_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212S_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212S_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212S_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212s.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212s] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212s:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212s_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212s:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212s_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212S_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 12
run_cell 13

echo "=== [r212s] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212s_{name}_32768.json")
    if not os.path.exists(f):
        print(f"{name.upper()}_MISSING"); sys.exit(1)
    j = json.load(open(f))
    c = (j.get(fmt) or {}).get("native_auto") or {}
    if not c or c.get("dev_us") is None:
        print(f"{name.upper()}_NO_CELL"); sys.exit(1)
    return c
m0, m11, m12, m13 = (cell(n) for n in ("m0", "m11", "m12", "m13"))
m8 = cell("m0", "mxfp8")
if m0.get("canary") not in (0, 0.0):
    print(f"CANARY_FAIL m4 m0={m0.get('canary')}"); fail = 1
v0, v11, v12, v13 = m0["dev_us"], m11["dev_us"], m12["dev_us"], m13["dev_us"]
w8 = m8.get("dev_us") or float("nan")
print("r212s int-op ladder (mxfp4 32K, same wheel)")
print(f"  m0  (fast full)      {v0:7.1f}")
print(f"  m11 (light, 0 ops)   {v11:7.1f}")
print(f"  m12 (light + 4 ops)  {v12:7.1f}")
print(f"  m13 (light + 8 ops)  {v13:7.1f}")
slope1 = (v12 - v11) / 4.0
slope2 = (v13 - v12) / 4.0
slope = (v13 - v11) / 8.0
print(f"  slope rung1: {slope1:7.2f} us/op   rung2: {slope2:7.2f} us/op"
      f"   overall: {slope:7.2f} us/op")
pool = v0 - v11
ops_share = slope * 8.0
resid = pool - ops_share
print(f"  assembly pool (m0-m11):        {pool:7.1f} us")
print(f"  8-op share at slope:           {ops_share:7.1f} us")
print(f"  residual (memcpy/store/other): {resid:7.1f} us")
if slope <= 0:
    print("LADDER_DEGENERATE (slope <= 0) — ladder ops folded or"
          " noise; do not fork on this leg")
    fail = 1
elif ops_share >= 0.6 * pool:
    print("FORK: OPS-BOUND — algebraic op reduction (r212t: fewer"
          " ops/element in the fast loop)")
else:
    print("FORK: RESIDUAL-BOUND — memcpy/store pool dominates;"
          " fragment store-packing lane next")
m8dev = abs(w8 / 1320.5 - 1.0)
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)"
      + (" OK" if m8dev <= 0.06 else " FAIL"))
if m8dev > 0.06:
    print("M8_STABILITY_FAIL"); fail = 1
print("R212S_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212S_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
