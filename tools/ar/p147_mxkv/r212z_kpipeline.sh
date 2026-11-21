#!/bin/bash
# r212z_kpipeline.sh -- arc7 Phase-8 r212z: DOUBLE-BUFFERED K DECODE.
# Zero-semantic-change wheel (probe gates hoisted to precomputed
# bools). Three same-wheel cells at 32K:
#   m0  = fast full (K->tSrK direct; m8 oracle rides here)
#   m11 = K const + V light (baseline)
#   m17 = K real + V light (k_lane = m17-m11)
# HEADLINE: m8 in band = CLEAN WHEEL (attractor escaped) -> k_lane
#           becomes the calibrated K-lane cost (vs r212u pool 338,
#           attractor k_lane 600.1); m8 still +34 = attractor held.
# m0's canary is the byte-level net.
# Artifacts -> /root/build/p199/out/p212z_{m0,m11,m17}_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212z

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212Z_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212z DOUBLE-BUFFERED K DECODE" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_paged_decode_mainloop.hpp \
  || { echo "R212Z_TREE_MISMATCH (p212y not applied?)"; exit 1; }
grep -q "r212x NESTED-FORM GUARD" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212Z_WALK_TREE_MISMATCH (p212x2 guard missing?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212Z_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212Z_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212Z_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212Z_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212Z_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212z.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212z] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {
  VP=$1
  echo "=== [r212z:vp$VP:c32768] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212z_vp${VP}_c32768.log 2>&1
  rc=$?
  echo "=== [r212z:vp$VP:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
  # vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
  for cand in p204_graphtime_m${VP}.json p204_graphtime.json; do
    if [ -f "$P/out/$cand" ]; then
      mv "$P/out/$cand" "$P/out/p212z_m${VP}_32768.json"
      return 0
    fi
  done
  echo "R212Z_MISSING_ARTIFACT_m${VP}"
}

run_cell 0
run_cell 11
run_cell 17

echo "=== [r212z] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
def cell(name, fmt="mxfp4"):
    f = os.path.join(out, f"p212z_{name}_32768.json")
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
    print(f"CANARY_FAIL m0={can} -- PIPELINE CORRUPTED OUTPUT;"
          " check the prologue/lookahead parity and revert"); fail = 1
v0, v11, v17 = m0["dev_us"], m11["dev_us"], m17["dev_us"]
w8 = m8.get("dev_us") or float("nan")
k_lane = v17 - v11
print("r212z k-decode pipeline (mxfp4 32K, same wheel)")
print(f"  m0  (K pipelined + V real) {v0:7.1f}   (r212y 1006.6;"
      " r212u clean 718.0)")
print(f"  m11 (K const + V light)  {v11:7.1f}")
print(f"  m17 (K pipelined + V light) {v17:7.1f}   k_lane {k_lane:+7.1f}"
      f"   (band 595-606; clean pool 338)")
m8dev = abs(w8 / 1320.5 - 1.0)
m8ok = m8dev <= 0.06
print(f"  m8 m0 oracle: {w8:.1f} ({m8dev*100:+.1f}% vs r212q 1320.5)")
# m11 anchor FIRST: if the severed floor ballooned (band ~414), the
# whole GEMM-1 region regressed and k_lane is two regressed cells
# differencing -- never read it as an overlap win.
if v11 > 800.0:
    print("FORK: REGION COLLAPSE -- m11 (severed floor) ballooned vs the"
          " 414 band: the runtime-parity fragment selection (tKrCur)"
          " demoted the MMA operand off static registers (r212t"
          " scratch-wall class); m0/m17/m8 all pay it. Revert; the"
          " overlap needs compile-time parity (peeled D-loop) or a"
          " single-buffer restructure")
elif k_lane < 450:
    print("FORK: OVERLAP WON -- the chain hid under the DPAS pipe;"
          " bank, then depth-2 lookahead or cross-K prologue next")
elif k_lane <= 660:
    print("FORK: IN BAND -- no overlap won (scheduler or register"
          " pressure); disasm the D-loop before deeper pipelining")
else:
    print("FORK: REGRESSED -- likely spills (r212t scratch-wall"
          " class); revert to sequential decode, halve the buffer")
if m8ok:
    print("M8_CLEAN (unexpected off-attractor) -- absolutes regain"
          " trust; recalibrate vs r212u")
if can not in (0, 0.0):
    print("FORK: CORRUPTION -- revert to the sequential decode and"
          " re-derive the parity mapping")
if not m8ok:
    print("M8_STABILITY_FAIL"); fail = 1
print("R212Z_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212Z_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
