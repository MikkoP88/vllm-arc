#!/bin/bash
# r212p_light.sh -- arc7 Phase-8 r212p: ONE-CELL light-probe pricing.
# probe 11 (light tail: extracts + float convert + store; prep
# severed) on the r212o wheel decides the next fork:
#   m0 - m11 = FULL decode machinery (assembly + scales + prep)
#   m9 - m11 = the heavy probe tail's extra machinery
#   m11 - c8 = extracts + stores + walk loads + softmax/P + staging
# m11 ~= m9  -> extracts/stores/walk dominate -> store packing lever
# m11 ~= c8+x small -> decode machinery dominates -> codegen lever
# Baselines: the banked r212o same-wheel cells (m0 741.8 / m9 380.6 /
# c8 248.7 for mxfp4). m8's m11 cell is PARTIAL (dequant_reg keeps
# its full decode under probe 11 — e4m3 byte-identity discipline);
# the gate prices m4 only.
# Artifacts -> /root/build/p199/out/p212p_m11_32768.json
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r212p

WHEEL=$R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-cp312-cp312-linux_x86_64.whl
[ -f "$WHEEL" ] || { echo "R212P_NO_WHEEL"; exit 1; }
md5sum "$WHEEL"
grep -q "r212p LIGHT PROBE TAIL" \
  /root/build/vxk-arc-src/csrc/xpu/attn/xe_2/mx/mx_dequant.hpp \
  || { echo "R212P_WHEEL_TREE_MISMATCH (patch not applied?)"; exit 1; }
HARNESS=$W/tools/ar/p147_mxkv/p204_graphtime.py
[ -f "$HARNESS" ] || { echo "R212P_NO_HARNESS"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R212P_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R212P_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R212P_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R212P_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_r212p.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r212p] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
sleep 5

echo "=== [r212p:m11:c32768] $(date +%H:%M:%S) ==="
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e P204_CTX=32768 -e P204_NCAP=32 -e P204_REPLAY=50 \
  -e P204_SPLITS=auto -e P204_MAXSK=cap \
  -e P204_ESHAPE=1 -e P204_D=256 -e P204_HREP=6 -e P204_BLOCK=64 \
  -e P204_FMTS=mxfp4,mxfp8 \
  -e ARC_MXKV_VPROBE=11 -e ARC_MXKV_DIRECTBF16=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p204_graphtime.py /out" \
  > $P/logs/r212p_m11_c32768.log 2>&1
rc=$?
echo "=== [r212p:m11:c32768] exit rc=$rc $(date +%H:%M:%S) ==="
[ -f "$P/out/p204_graphtime_m11.json" ] && \
  mv "$P/out/p204_graphtime_m11.json" "$P/out/p212p_m11_32768.json" || \
  echo "R212P_MISSING_ARTIFACT_m11"

echo "=== [r212p] gate $(date +%H:%M:%S) ==="
python3 - "$P/out" <<'PY'
import json, sys, os
out = sys.argv[1]
fail = 0
f = os.path.join(out, "p212p_m11_32768.json")
if not os.path.exists(f):
    print("M11_MISSING"); sys.exit(1)
j = json.load(open(f))
c = (j.get("mxfp4") or {}).get("native_auto") or {}
if not c or c.get("dev_us") is None:
    print("M11_NO_MXFP4_CELL"); sys.exit(1)
m11 = c["dev_us"]
M0, M9, C8 = 741.8, 380.6, 248.7
print("r212p three-way split (mxfp4 32K, r212o-wheel baselines)")
print(f"  m0  (fast full)      {M0:7.1f}")
print(f"  m9  (heavy probe)    {M9:7.1f}")
print(f"  m11 (light probe)    {m11:7.1f}   <- this leg")
print(f"  c8  (const floor)    {C8:7.1f}")
print(f"  decode machinery (m0-m11): {M0-m11:7.1f} us")
print(f"  heavy-tail extra  (m9-m11): {M9-m11:7.1f} us")
print(f"  extract/store/walk/softmax (m11-c8): {m11-C8:7.1f} us")
verdict = ("STORE-PACKING LEVER (extracts/stores/walk dominate)"
           if m11 - C8 > M0 - m11 else
           "CODEGEN LEVER (decode machinery dominates the residual)")
print(f"  FORK: {verdict}")
c8m = (j.get("mxfp8") or {}).get("native_auto") or {}
if c8m and c8m.get("dev_us") is not None:
    print(f"  (m8 m11 partial = {c8m['dev_us']:.1f} — reg lane rides full"
          " decode under probe 11; not an m8 instrument)")
print("R212P_GATE_PASS")
PY
grc=$?
echo "R212P_DONE rc=$grc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $grc
