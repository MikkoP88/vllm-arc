#!/bin/bash
# r165_splitsweep.sh -- P6.47 leg A11: _MX_SPLIT_MAX sweep probe
# (p158_splitsweep.py) on the cert25y-tree work copy.
#
# A10 attribution banked the split cap as the 32K decode cliff (serial
# walk 8x growth from 8K to 32K; m4 byte-law inversion at the 24,576-
# program cell). The sweep monkeypatches the module global (read at
# launch: grid z, NUM_SPLITS constexpr, mid-cache key) across
# {32, 64, 128} at the A10 problem cells. W14 quiesce before, trap
# restore after (rig law). Refuses a stale tree. Artifacts ->
# /root/build/p147/outA11. Exit 0 = leg A11 PASS (correctness gate;
# timing rows are data).
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}

w14_restore() {
  docker rm -f arc-p165 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p165.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/outA11

# ---- tree markers: refuse a stale work tree ----
grep -q "_MX_SPLIT_MAX" \
  "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R165_STALE_TREE (no split-max surface) -- ship first"; exit 1; }
grep -q "p158_splitsweep" "$W/tools/ar/p147_mxkv/p158_splitsweep.py" 2>/dev/null \
  || { echo "R165_STALE_TREE (probe missing) -- ship it first"; exit 1; }
grep -q "HK = 4" "$W/tools/ar/p147_mxkv/p158_splitsweep.py" 2>/dev/null \
  || { echo "R165_STALE_TREE (probe is not the true-shape variant)"; exit 1; }

echo "=== [r165:A11] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R165_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p165 >/dev/null 2>&1 || true

echo "=== [r165:A11] split-cap sweep probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p165 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/outA11":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p158_splitsweep.py" \
  > $P/logs/legA11.out 2>&1
rc=$?
echo "=== [r165:A11] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[|A11_PROBE" $P/logs/legA11.out | tee $P/outA11/legA11_summary.txt || true

cd $P/outA11
sha256sum legA11_summary.txt $P/logs/legA11.out 2>/dev/null > SHA256SUMS || true
echo "R165_LEGA11_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
