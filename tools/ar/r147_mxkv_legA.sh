#!/bin/bash
# r147_mxkv_legA.sh -- P6.47 leg A: offline XPU validation of the MX
# MXFP4/MXFP8 KV-cache primitives (P6.47 goal, 2026-11-11).
# Runs p147_mxkv_offline.py INSIDE cert23 (docker run, XPU devices) --
# NO engine boot. Gates: ref (exact-math self-test), bytes (store
# kernel == reference bytes), decode (vs fp64 exact on dequantized
# cache), accuracy (vs unquantized), robust (adversarial + boundary
# lens + fragmented paging), telemetry (store/decode timings at
# 1K/8K/32K -- always-loop bottleneck datapoint).
# W14 quiesce before, trap restore after (rig law). Files arrive via
# the p147 ship step (md5-verified tar); this script only runs.
# Artifacts -> /root/build/p147/{out,logs}. Exit 0 = leg A PASS.
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=vllm-arc:v0.30.0-b70-cert23

w14_restore() {
  docker rm -f arc-p147a >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p147a.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out

echo "=== [r147:legA] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R147_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p147a >/dev/null 2>&1 || true

echo "=== [r147:legA] offline harness in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p147a \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p147_mxkv_offline.py /out" \
  > $P/logs/legA.out 2>&1
rc=$?
echo "=== [r147:legA] harness exit rc=$rc $(date +%H:%M:%S) ==="

# verdict + key lines to the gate log (full log banked)
grep -E "^\[face|^  ok |^  FAIL|^VERDICT|telemetry/" $P/logs/legA.out > $P/out/legA_summary.txt || true
cat $P/out/legA_summary.txt

cd $P/out
sha256sum legA_summary.txt p147_offline.json $P/logs/legA.out 2>/dev/null > SHA256SUMS || true
echo "R147_LEGA_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
