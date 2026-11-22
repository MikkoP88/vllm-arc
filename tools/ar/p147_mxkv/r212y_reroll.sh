#!/bin/bash
# r212y_reroll.sh -- arc7 Phase-8 r212y: ATTRACTOR RE-ROLL.
# Zero-semantic-change wheel (probe gates hoisted to precomputed
# bools). Three same-wheel cells at 32K:
#   m0  = fast full (K->tSrK direct; m8 oracle rides here)
#   m11 = K const + V light (baseline)
#   m17 = K real + V light (k_lane = m17-m11)
# HEADLINE: m8 in band = CLEAN WHEEL (attractor escaped) -> k_lane
#           becomes the calibrated K-lane cost (vs r212u pool 338,
#           attractor k_lane 600.1); m8 still +34 = attractor held.
# m0's canary is the byte-level net.
# Artifacts -> /root/build/p199/out/p212y_{m0,m11,m17}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212y

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212Y_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212y ATTRACTOR RE-ROLL" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212Y_TREE_MISMATCH (p212y not applied?)"; exit 1; }
grep -q "r212x NESTED-FORM GUARD" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212Y_WALK_TREE_MISMATCH (p212x2 guard missing?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212Y_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212Y_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212Y_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212Y_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212Y_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212y.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212y] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212y:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212y_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212y:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212y_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212Y_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17

echo "=== [r212y] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212y_{name}_32768.json")
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
    print(f"CANARY_FAIL m0={can} -- OUTPUT CORRUPTED; investigate"
          " (r212y is semantically null -- should be impossible)"); fail = 1
v0, v11, v17 = m0["dev_us"], m11["dev_us"], m17["dev_us"]
w8 = m8.get("dev_us") or float("nan")
k_lane = v17 - v11
print("r212y attractor re-roll (mxfp4 32K, same wheel)")
print(f"  m0  (K->tSrK + V real)   {v0:7.1f}   (r212u clean 718.0;"
      " r212x attractor 995.9)")
print(f"  m11 (K const + V light)  {v11:7.1f}")
print(f"  m17 (K->tSrK + V light)  {v17:7.1f}   k_lane {k_lane:+7.1f}"
      f"   (attractor 600.1; clean pool 338)")
m8dev = abs(w8 / 1320.5 - 1.0)
m8ok = m8dev <= 0.06
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)")
if m8ok:
    print("HEADLINE: CLEAN WHEEL -- attractor ESCAPED; k_lane is now"
          " the calibrated K-lane cost; bank + design r212z pipeline"
          " against this baseline")
else:
    print("HEADLINE: ATTRACTOR HELD (4th wheel) -- k_lane directional"
          " only; next escalation: structural perturbation (noinline"
          " const-fill helper or else-if reorder) or accept and"
          " pipeline on same-wheel deltas")
if can not in (0, 0.0):
    print("FORK: CORRUPTION on a null-change wheel -- stop and diff"
          " the binary vs r212x")
if not m8ok:
    print("M8_STABILITY_FAIL"); fail = 1
print("R212Y_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212Y_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
