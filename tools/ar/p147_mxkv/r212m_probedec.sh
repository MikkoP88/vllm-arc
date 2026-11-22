#!/bin/bash
# r212m_probedec.sh -- arc7 Phase-8 r212m pre-flight: the decode-math
# probe. Probe 9 severs EXACTLY the per-element decode tail (e2m1/e4m3
# bit assembly + scale multiply + f32->bf16 convert) at all three
# dequant sites; loads, extracts, selects and fragment writes stay.
# Cells: probe 0 (full, canary-gated) + probe 9 (timing-only, canary
# EXPECTED wrong -- const-fill ablation law) x ctx 32768,
# FMTS=mxfp4,mxfp8, SPLITS=auto, engine geometry, graph 32x50.
# Decision rule: delta(0->9) >= ~100 us/2fmt-average -> the direct-bf16
# exponent-fold iteration is worth the edge-band surgery; ~0 -> decode
# ALU already hidden under load latency -> pivot to load-structure
# (V-stage u32 stores, tile width).
# Artifacts -> /root/build/p199/out/p212m_m<PROBE>_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212m

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212M_NO_WHEEL"; exit 1; }
grep -q "r212m probe 9" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212M_WHEEL_TREE_MISMATCH (probe patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212M_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212M_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212M_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212M_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212M_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212m.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212m] W14 quiesce $(date +%H:%M:%S) ==="
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
  echo "=== [r212m:m$PROBE:c$CTX] $(date +%H:%M:%S) ==="
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
    > $P/logs/r212m_m${PROBE}_c$CTX.log 2>&1
  local rc=$?
  echo "=== [r212m:m$PROBE:c$CTX] exit rc=$rc $(date +%H:%M:%S) ==="
  if [ -f "$P/out/p204_graphtime_m$PROBE.json" ]; then
    mv "$P/out/p204_graphtime_m$PROBE.json" "$P/out/p212m_m${PROBE}_$CTX.json"
  else
    echo "R212M_MISSING_ARTIFACT_m${PROBE}_$CTX"
  fi
}

run_cell 0 32768
run_cell 9 32768

echo "=== [r212m] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
cells = {}
for probe in ("0", "9"):
    f = os.path.join(out, f"p212m_m{probe}_32768.json")
    if not os.path.exists(f):
        print(f"MISSING m{probe}"); fail = 1; continue
    cells[probe] = json.load(open(f))
print("r212m decode-math probe at 32K -- us/call (probe 9 = decode tail severed)")
print(f" {'fmt':>6} | {'full m0':>8} | {'m9':>8} | {'decode slice':>12} | {'pct':>6}")
for fmt in ("mxfp4", "mxfp8"):
    c0 = (cells.get("0") or {}).get(fmt) or {}
    c9 = (cells.get("9") or {}).get(fmt) or {}
    a = (c0.get("native_auto") or {}).get("dev_us")
    b = (c9.get("native_auto") or {}).get("dev_us")
    if a is None or b is None:
        print(f" {fmt:>6} | MISSING CELL"); fail = 1; continue
    d = a - b
    print(f" {fmt:>6} | {a:8.1f} | {b:8.1f} | {d:12.1f} | {100.0*d/a:5.1f}%")
    can0 = (c0.get("native_auto") or {}).get("canary")
    if can0 is None or can0 > 2e-3:
        print(f"          CANARY_FAIL m0 {can0}"); fail = 1
    can9 = (c9.get("native_auto") or {}).get("canary")
    if can9 is not None and can9 <= 2e-3:
        print(f"  NOTE m9 canary clean ({can9}) -- probe 9 did not sever; "
              "wheel tree mismatch?")
        fail = 1
print("R212M_GATE_" + ("PASS" if not fail else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212M_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
