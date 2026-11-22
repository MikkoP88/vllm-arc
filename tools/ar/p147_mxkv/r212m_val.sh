#!/bin/bash
# r212m_val.sh -- arc7 Phase-8 r212m: validate the direct-bf16 decode
# iteration (e8m0 exponent folded into the bf16 exponent field; probe 9
# proved the decode tail is 56-72% of the 32K walk). Cells:
#   - probe 0 (full, canary-gated -- bit-exactness of the fast path is
#     the claim) x ctx {1024,8192,32768}, FMTS=mxfp4,mxfp8
#   - ARC_MXKV_DIRECTBF16=0 x ctx 32768 (same-wheel float-path oracle:
#     reproduces r212l-class timing AND proves the gate wiring)
# Artifacts -> /root/build/p199/out/p212m2_{m0,dbf0}_<CTX>.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212mv

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212MV_NO_WHEEL"; exit 1; }
grep -q "r212m DIRECT-BF16 DECODE LAW" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212MV_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212MV_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212MV_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212MV_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212MV_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212MV_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212mv.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212mv] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

run_cell() {  # tag ctx dbf
  local TAG=$1
  local CTX=$2
  local DBF=$3
  docker rm -f $CNAME >/dev/null 2>&1
  echo "=== [r212mv:$TAG:c$CTX] $(date +%H:%M:%S) ==="
  docker run --rm --name $CNAME \
    --device /dev/dri --group-add video \
    -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
    -e P204_CTX=$CTX -e P204_NCAP=32 -e P204_REPLAY=50 \
    -e P204_SPLITS=auto -e P204_MAXSK=cap \
    -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
    -e P204_FMTS=mxfp4,mxfp8 \
    -e ARC_MXKV_VPROBE=0 -e ARC_MXKV_DIRECTBF16=$DBF \
    -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
    -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
    --entrypoint /bin/bash \
    $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
       && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
    > $P/logs/r212mv_${TAG}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212mv:$TAG:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m0.json" ]; then
    mv "$P/out/p204_graphtime_m0.json" "$P/out/p212m2_${TAG}_$CTX.json"
  else
    echo "R212MV_MISSING_ARTIFACT_${TAG}_$CTX"
  fi
}

for CTX in 1024 8192 32768; do
  run_cell m0 $CTX 1
done
run_cell dbf0 32768 0

echo "=== [r212mv] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
# r212l baselines (banked, the float-path state)
BASE = {("mxfp4", 1024): 594.5, ("mxfp4", 8192): 420.6, ("mxfp4", 32768): 1028.3,
        ("mxfp8", 1024): 658.3, ("mxfp8", 8192): 541.7, ("mxfp8", 32768): 1359.0}
def cell(tag, ctx, fmt):
    f = os.path.join(out, f"p212m2_{tag}_{ctx}.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get(fmt) or {}).get("native_auto") or {}
print("r212m2 direct-bf16 fulls (probe 0) -- us/call vs r212l float-path baseline")
print(f" {'fmt':>6} {'ctx':>6} | {'fast':>8} | {'base':>8} | {'delta':>7} | canary")
for fmt in ("mxfp4", "mxfp8"):
    for ctx in (1024, 8192, 32768):
        c = cell("m0", ctx, fmt)
        if not c or c.get("dev_us") is None:
            print(f" {fmt:>6} {ctx:>6} | MISSING"); fail = 1; continue
        b = BASE[(fmt, ctx)]
        d = 100.0 * (c["dev_us"] - b) / b
        print(f" {fmt:>6} {ctx:>6} | {c['dev_us']:8.1f} | {b:8.1f} | {d:+6.1f}% | {c.get('canary')}")
        if c.get("canary") is None or c["canary"] > 2e-3:
            print(f"          CANARY_FAIL {c.get('canary')}"); fail = 1
print("same-wheel float-path oracle (DIRECTBF16=0) at 32K")
print(f" {'fmt':>6} | {'dbf0':>8} | {'fast':>8} | {'gain':>7} | canary(dbf0)")
for fmt in ("mxfp4", "mxfp8"):
    c0 = cell("dbf0", 32768, fmt)
    c1 = cell("m0", 32768, fmt)
    if not c0 or c0.get("dev_us") is None or not c1 or c1.get("dev_us") is None:
        print(f" {fmt:>6} | MISSING"); fail = 1; continue
    g = 100.0 * (c0["dev_us"] - c1["dev_us"]) / c0["dev_us"]
    print(f" {fmt:>6} | {c0['dev_us']:8.1f} | {c1['dev_us']:8.1f} | {g:6.1f}% | {c0.get('canary')}")
    if c0.get("canary") is None or c0["canary"] > 2e-3:
        print(f"          CANARY_FAIL dbf0 {c0.get('canary')}"); fail = 1
print("R212MV_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212MV_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
