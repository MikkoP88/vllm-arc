#!/bin/bash
# r212q_u32.sh -- arc7 Phase-8 r212q: ONE-CELL u32-domain readout.
# The fast-path bit assembly runs end to end in u32 (fused nibble
# shift, u32 field math, 2-byte memcpy only at the store) per
# p212q_patch.py. One run = m4 m0 (the subject) + m8 m0 (byte-identical
# source -> stability oracle) at 32K.
#   r212o/r212p baselines (same geometry): m4 m0 741.8, m9 380.6,
#   m11 380.8, c8 248.7; m8 m0 1323.1.
# Fork logic: the tail base (~380.7) is tail-invariant, so
#   m4_new - 380.7 = the surviving bit-assembly codegen cost
# (was 361.0). Big drop -> codegen confirmed, full telemetry next;
# flat -> u32 is not the wall, spill hypothesis takes over.
# Artifacts -> /root/build/p199/out/p212q_m0_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212q

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212Q_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212q U32-DOMAIN ASSEMBLY" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212Q_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212Q_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212Q_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212Q_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212Q_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212Q_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212q.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212q] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

echo "=== [r212q:m0:c32768] $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e P204_CTX=32768 -e P204_NCAP=32 -e P204_REPLAY=50 \
  -e P204_SPLITS=auto -e P204_MAXSK=cap \
  -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
  -e P204_FMTS=mxfp4,mxfp8 \
  -e ARC_MXKV_VPROBE=0 -e ARC_MXKV_DIRECTBF16=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
  > $P/logs/r212q_m0_c32768.log 2>&1
rc=$?
echo "=== [r212q:m0:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
# vprobe=0 falsy -> harness suffix is EMPTY (p204_graphtime.json)
for cand in p204_graphtime_m0.json p204_graphtime.json; do
  if [ -f "$P/out/$cand" ]; then
    mv "$P/out/$cand" "$P/out/p212q_m0_32768.json"
    break
  fi
done
[ -f "$P/out/p212q_m0_32768.json" ] || echo "R212Q_MISSING_ARTIFACT_m0"

echo "=== [r212q] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
f = os.path.join(out, "p212q_m0_32768.json")
if not os.path.exists(f):
    print("M0_MISSING"); sys.exit(1)
j = json.load(open(f))
m4 = (j.get("mxfp4") or {}).get("native_auto") or {}
m8 = (j.get("mxfp8") or {}).get("native_auto") or {}
if not m4 or m4.get("dev_us") is None:
    print("M0_NO_MXFP4_CELL"); sys.exit(1)
if m4.get("canary") not in (0, 0.0) or m8.get("canary") not in (0, 0.0):
    print(f"CANARY_FAIL m4={m4.get('canary')} m8={m8.get('canary')}")
    fail = 1
v = m4["dev_us"]; w = m8.get("dev_us") or float("nan")
M0, M9, M11, C8, TAIL = 741.8, 380.6, 380.8, 248.7, 380.7
M8 = 1323.1
print("r212q u32-domain one-cell (mxfp4/mxfp8 32K, r212o/r212p baselines)")
print(f"  m4 m0 (this leg)        {v:7.1f}   (r212o {M0:.1f})")
print(f"  m8 m0 (stability oracle){w:7.1f}   (r212o {M8:.1f})")
print(f"  surviving assembly cost (m4 - tail {TAIL}): {v - TAIL:7.1f} us"
      f"   (was {M0 - TAIL:.1f})")
print(f"  extract/store/walk/softmax (m11 - c8):     {M11 - C8:7.1f} us")
m8dev = abs(w / M8 - 1.0)
if m8dev > 0.06:
    print(f"M8_STABILITY_FAIL dev={m8dev*100:.1f}% (byte-identical source"
          " must hold +-6%)")
    fail = 1
else:
    print(f"  m8 stability: {m8dev*100:+.1f}% OK (e4m3 discipline)")
if v <= 0.90 * M0:
    print("FORK: CODEGEN CONFIRMED -- large collapse; full telemetry next")
elif v <= 0.97 * M0:
    print("FORK: PARTIAL -- u32 helps some; consider SWAR follow-on")
elif v <= 1.03 * M0:
    print("FORK: NULL -- u32 domain is not the wall; spill hypothesis"
          " -> disassembly route")
elif v <= 1.06 * M0:
    print("FORK: MILD REGRESSION -- layout dice or worse codegen; bank"
          " as null, revert decision by full-cell check")
    fail = 1
else:
    print("FORK: REGRESSION > 6% -- revert candidates: wheel.r212pbak")
    fail = 1
print("R212Q_GATE_" + ("PASS" if fail == 0 else "FAIL"))
sys.exit(fail)
PY
grc=$?
echo "R212Q_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
