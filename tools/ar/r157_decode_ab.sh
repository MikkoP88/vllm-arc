#!/bin/bash
# r157_decode_ab.sh -- P6.47 leg A4: offline XPU A/B of the split-KV
# decode v2 vs certified v1 (p157_mxkv_decode_ab.py inside the shipped
# image layer; decode-perf research results/2026-11-12 bank).
#
# Runs AFTER the run3 correctness window (GPU busy until then) and
# AFTER a p156-style work-tree refresh that carries this commit (the
# script verifies the v2 markers are present, else refuses — a stale
# tree would silently A/B v1 against v1).
# W14 quiesce before, trap restore after (rig law; the offline harness
# shares the GPUs with the W14 serve pool).
# Artifacts -> /root/build/p147/out4. Exit 0 = leg A4 PASS.
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}

w14_restore() {
  docker rm -f arc-p157 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p157.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out4

# ---- tree markers: refuse a stale work tree (v1-vs-v1 vacuous pass) ----
grep -q "_mx_decode_split" "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R157_STALE_TREE (no _mx_decode_split in work triton_mx_decode.py) -- refresh via the p156 bundle first"; exit 1; }
grep -q "num_stages=num_stages" "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R157_STALE_TREE (no num_stages wiring -- A6 needs the stages axis) -- refresh via the p156 bundle first"; exit 1; }
grep -q "num_stages" "$W/tools/ar/p147_mxkv/p157_mxkv_decode_ab.py" 2>/dev/null \
  || { echo "R157_STALE_TREE (harness lacks the stages sweep) -- refresh via the p156 bundle first"; exit 1; }
[ -f "$W/tools/ar/p147_mxkv/p157_mxkv_decode_ab.py" ] \
  || { echo "R157_STALE_TREE (harness missing) -- refresh via the p156 bundle first"; exit 1; }

echo "=== [r157:A4] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R157_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p157 >/dev/null 2>&1 || true

echo "=== [r157:A4] decode A/B in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p157 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out4":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p157_mxkv_decode_ab.py /out" \
  > $P/logs/legA4.out 2>&1
rc=$?
echo "=== [r157:A4] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^VERDICT" $P/logs/legA4.out | tee $P/out4/legA4_summary.txt || true

cd $P/out4
sha256sum legA4_summary.txt p157_decode_ab.json $P/logs/legA4.out 2>/dev/null > SHA256SUMS || true
echo "R157_LEGA4_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
