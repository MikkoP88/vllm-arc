#!/bin/bash
# r166_v4rows.sh -- P6.47 leg A12: v4 verify-row-shared decode probe
# (p159_v4rows.py) on the cert25y-tree work copy.
#
# A11 lever 2 (A11-NEXT-LEVERS.md): mx_decode_rows walks the request
# ONCE for all R_TILE=4 MTP verify rows — shared K/V dequant at half of
# v3's spilled state. Correctness vs per-row v2 at LIMIT-LAW boundary
# shapes (gate d<=2e-3 + finite) + timing at the A10 problem cells
# (G 1/8 x ctx 2048/8192/32768 x 3 formats). W14 quiesce before, trap
# restore after (rig law). Refuses a stale tree. Artifacts ->
# /root/build/p147/outA12. Exit 0 = leg A12 PASS.
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}

w14_restore() {
  docker rm -f arc-p166 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p166.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/outA12

# ---- tree markers: refuse a stale work tree ----
grep -q "mx_decode_rows" \
  "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R166_STALE_TREE (no v4 kernel) -- ship first"; exit 1; }
grep -q "ARC_MXKV_V4_VERIFY" "$W/vllm_gaudi/xpu/patches/mx_kv/engine.py" 2>/dev/null \
  || { echo "R166_STALE_TREE (no v4 engine wiring) -- ship first"; exit 1; }
grep -q "p159_v4rows" "$W/tools/ar/p147_mxkv/p159_v4rows.py" 2>/dev/null \
  || { echo "R166_STALE_TREE (probe missing) -- ship it first"; exit 1; }
grep -q "R_TILE = 4" "$W/tools/ar/p147_mxkv/p159_v4rows.py" 2>/dev/null \
  || { echo "R166_STALE_TREE (probe is not the true-shape variant)"; exit 1; }

echo "=== [r166:A12] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R166_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p166 >/dev/null 2>&1 || true

echo "=== [r166:A12] v4 rows probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p166 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/outA12":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p159_v4rows.py" \
  > $P/logs/legA12.out 2>&1
rc=$?
echo "=== [r166:A12] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[|A12_PROBE" $P/logs/legA12.out | tee $P/outA12/legA12_summary.txt || true

cd $P/outA12
sha256sum legA12_summary.txt $P/logs/legA12.out 2>/dev/null > SHA256SUMS || true
echo "R166_LEGA12_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
