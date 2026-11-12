#!/bin/bash
# r167_v5dot.sh -- P6.47 leg A13: v5 XMX tl.dot decode probe
# (p160_v5dot.py) on the cert25y-tree work copy.
#
# A11 lever 3 (A11-NEXT-LEVERS.md): mx_decode_rows_dot runs the v4
# row-group walk with the score path in tl.dot — fp16 operands, f32
# accumulate, M = R_TILE x N_REP = 24 lanes (padded to 32). Correctness
# vs per-row v2 at LIMIT-LAW boundary shapes (gate d<=2e-3 + finite;
# fp16-operand class, the ctrl lane's own) + timing at the A10 problem
# cells (G 1/8 x ctx 2048/8192/32768 x 3 formats, v5 vs v4 vs v2). W14
# quiesce before, trap restore after (rig law). Refuses a stale tree.
# Artifacts -> /root/build/p147/outA13. Exit 0 = leg A13 PASS.
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}

w14_restore() {
  docker rm -f arc-p167 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p167.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/outA13

# ---- tree markers: refuse a stale work tree ----
grep -q "_mx_decode_split_dot" \
  "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R167_STALE_TREE (no v5 kernel) -- ship first"; exit 1; }
grep -q "ARC_MXKV_V5_DOT" "$W/vllm_gaudi/xpu/patches/mx_kv/engine.py" 2>/dev/null \
  || { echo "R167_STALE_TREE (no v5 engine wiring) -- ship first"; exit 1; }
grep -q "p160_v5dot" "$W/tools/ar/p147_mxkv/p160_v5dot.py" 2>/dev/null \
  || { echo "R167_STALE_TREE (probe missing) -- ship it first"; exit 1; }
grep -q "M_LANES" "$W/tools/ar/p147_mxkv/p160_v5dot.py" 2>/dev/null \
  || { echo "R167_STALE_TREE (probe is not the M-lane variant)"; exit 1; }

echo "=== [r167:A13] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R167_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p167 >/dev/null 2>&1 || true

echo "=== [r167:A13] v5 dot probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p167 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/outA13":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p160_v5dot.py" \
  > $P/logs/legA13.out 2>&1
rc=$?
echo "=== [r167:A13] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[|A13_PROBE" $P/logs/legA13.out | tee $P/outA13/legA13_summary.txt || true

cd $P/outA13
sha256sum legA13_summary.txt $P/logs/legA13.out 2>/dev/null > SHA256SUMS || true
echo "R167_LEGA13_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
