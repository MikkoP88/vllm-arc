#!/bin/bash
# r199_ablation.sh -- arc7 Phase-8 r199a: host/device attribution
# telemetry for the MX-native decode (arc7mx wheel) inside cert25zc.
# Splits each native call into host (wall-device) vs device (XPU
# events) time at 1K/8K/32K per format -- the 0.10-0.26x v2 gap
# decomposes into C++-entry overhead vs the dequant walk.
# Dequant-ablation wheels reuse this driver unchanged.
# Prereq: /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl
# + work tree at /root/build/p198/../p199/work (p198's mx_kv tree +
# p199_ablation.py). W14 quiesce before, trap restore after (rig law;
# R199_KEEP_DOWN=1 for batch rounds). Artifacts -> /root/build/p199/out.
# Exit 0 = telemetry collected.
set -uo pipefail
R=/root/build
P=$R/p199
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r199

WHEEL=$(ls $R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "R199_NO_WHEEL (run build_vxk_wheel_arc7mx.sh first)"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p199_ablation.py" ] || { echo "R199_NO_HARNESS -- ship first"; exit 1; }
[ -d "$W/vllm_gaudi/xpu/patches/mx_kv" ] || { echo "R199_NO_MXKV_TREE -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R199_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R199_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R199_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R199_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p199.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r199] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R199_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r199] attribution telemetry in $IMG + arc7mx wheel $(date +%H:%M:%S) ==="
echo "[r199] wheel: $WHEEL"
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p199_ablation.py /out" \
  > $P/logs/r199.out 2>&1
rc=$?
echo "=== [r199] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^  |P199_DONE|Error|error" $P/logs/r199.out | head -20 || true

cd $P/out
sha256sum p199_ablation.json $P/logs/r199.out 2>/dev/null > SHA256SUMS || true
echo "R199_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
