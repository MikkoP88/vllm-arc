#!/bin/bash
# r198_native_offline.sh -- arc7 Phase-8 r198: offline XPU validation of
# the MX-NATIVE paged-decode kernel (arc7mx wheel) inside cert25zc with
# the wheel installed over the image venv (NO engine boot).
# Faces: native-exact (vs fp64 on dequantized cache), native-v2 (vs the
# certified Triton decoder on the same cache), bitwise (seq_len==1
# dequant bit-math oracle), robust (adversarial + block-16 boundary
# lens, shuffled tables), splits (forced 8-split ReduceSplitK + default
# plan), telemetry (native vs v2 timings 1K/8K/32K -- always-loop
# bottleneck datapoint). Gates: p198_native_offline.py face law
# (d<=2e-3 or bf16-rounded-match; cos>=0.99999; bitwise 0 mismatches).
# Prereq: /root/build/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl
# (build_vxk_wheel_arc7mx.sh). Work tree staged at /root/build/p198/work
# by the r198 ship step (repo subset: harness + vllm_gaudi/xpu/patches/
# mx_kv, md5-verified). W14 quiesce before, trap restore after (rig
# law; fullres recipe -- radiance-test was recreated 2026-11-13 and
# docker start alone cannot rebuild it). Artifacts ->
# /root/build/p198/{out,logs}. Exit 0 = r198 offline PASS.
set -uo pipefail
R=/root/build
P=$R/p198
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert25zc
CNAME=arc-r198

WHEEL=$(ls $R/wheels-arc-vxk/vllm_xpu_kernels-0.1.14.1+arc7mx-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "R198_NO_WHEEL (run build_vxk_wheel_arc7mx.sh first)"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p198_native_offline.py" ] || { echo "R198_NO_HARNESS -- ship first"; exit 1; }
[ -d "$W/vllm_gaudi/xpu/patches/mx_kv" ] || { echo "R198_NO_MXKV_TREE -- ship first"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R198_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R198_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R198_PORT_HELD (unknown holder)"; exit 3
  fi
fi

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if [ "${R198_KEEP_DOWN:-0}" = "1" ]; then
    echo "W14_LEFT_DOWN (debug batch mode -- restore manually when done)"
    return 0
  fi
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (the actor removed radiance-test 2026-11-13;
  # docker start + serve_only can never restore it -- see r185b/r186)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p198.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r198] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R198_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f $CNAME >/dev/null 2>&1 || true

echo "=== [r198] offline harness in $IMG + arc7mx wheel $(date +%H:%M:%S) ==="
echo "[r198] wheel: $WHEEL"
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out -v "$R/wheels-arc-vxk":/wheels:ro \
  --entrypoint /bin/bash \
  $IMG -c "/opt/venv/bin/pip install --quiet --force-reinstall --no-deps /wheels/$(basename "$WHEEL") \
     && cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p198_native_offline.py /out" \
  > $P/logs/r198.out 2>&1
rc=$?
echo "=== [r198] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^VERDICT|telemetry/" $P/logs/r198.out > $P/out/r198_summary.txt || true
cat $P/out/r198_summary.txt

cd $P/out
sha256sum r198_summary.txt p198_native.json $P/logs/r198.out 2>/dev/null > SHA256SUMS || true
echo "R198_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
