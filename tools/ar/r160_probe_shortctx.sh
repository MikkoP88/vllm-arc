#!/bin/bash
# r160_probe_shortctx.sh -- P6.47 leg A9: true-shape short-ctx probe
# (p157_v3_shortctx.py) on the s_eff-floor v3 tree.
#
# Validates the v3 short-ctx occupancy fix (1-tile s_eff floor) and the
# TRUE qwen shape (HK=4/Hq=24/D=256, N_REP=6) that the offline batteries
# never tuned at. Runs after the r159 battery frees the GPUs; W14 quiesce
# before, trap restore after (rig law). Refuses a stale tree (no s_eff
# floor marker / no probe). Artifacts -> /root/build/p147/outA9.
# Exit 0 = leg A9 PASS (correctness gate; timing rows are data).
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}

w14_restore() {
  docker rm -f arc-p160 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p160.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/outA9

# ---- tree markers: refuse a stale work tree (probe would A/B v2 vs the
# pre-fix v3 and "validate" nothing) ----
grep -q "s_eff = tl.minimum(NUM_SPLITS, tl.maximum(1, n_tiles))" \
  "$W/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py" 2>/dev/null \
  || { echo "R160_STALE_TREE (no v3 s_eff floor) -- ship the fix first"; exit 1; }
grep -q "_split_verify_decode" \
  "$W/vllm_gaudi/xpu/patches/mx_kv/engine.py" 2>/dev/null \
  || { echo "R160_STALE_TREE (no batched verify decode) -- ship the fix first"; exit 1; }
grep -q "p157_v3_shortctx" "$W/tools/ar/p147_mxkv/p157_v3_shortctx.py" 2>/dev/null \
  || { echo "R160_STALE_TREE (probe missing) -- ship it first"; exit 1; }
grep -q "HK = 4" "$W/tools/ar/p147_mxkv/p157_v3_shortctx.py" 2>/dev/null \
  || { echo "R160_STALE_TREE (probe is not the true-shape variant)"; exit 1; }
grep -q "verify-batch" "$W/tools/ar/p147_mxkv/p157_v3_shortctx.py" 2>/dev/null \
  || { echo "R160_STALE_TREE (probe lacks the verify-batch face)"; exit 1; }

echo "=== [r160:A9] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R160_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p160 >/dev/null 2>&1 || true

echo "=== [r160:A9] true-shape short-ctx probe in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p160 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/outA9":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p157_v3_shortctx.py" \
  > $P/logs/legA9.out 2>&1
rc=$?
echo "=== [r160:A9] probe exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[|A9_PROBE" $P/logs/legA9.out | tee $P/outA9/legA9_summary.txt || true

cd $P/outA9
sha256sum legA9_summary.txt $P/logs/legA9.out 2>/dev/null > SHA256SUMS || true
echo "R160_LEGA9_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
