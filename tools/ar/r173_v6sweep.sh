#!/bin/bash
# r173_v6sweep.sh -- P6.47 A14: v6 deep-split sweep probe on the
# PATCHED kernel (2026-11-13
# kernel-fix mandate).
#
# The v6 patch (triton_mx_decode.py, this branch): v5 split cap 32 ->
# 256 with the adaptive >=4-serial-tile s_eff rule, a v5-only BLOCK_N=64
# MMA tune, and the chunk-loop combine (one chunk = the certified
# 32-split math; v2/v3/v4 lanes byte-identical). Single-variable A/B:
# the work tree = the INSTALLED cert25z mx_kv package with ONLY the
# patched triton_mx_decode.py overlaid (MXKV_DIR=/work) — everything
# else matches the shipping image byte-for-byte.
#
# p173_v6sweep.py: correctness v6 vs per-row v2 at LIMIT-LAW boundary
# shapes (gate d<=2e-3 + finite) + the BN {32,64} x splits {32,64,128,
# 256} sweep at G 1/8 x ctx 8192/32768 per format, A13-v5 (32/32) as
# the baseline cell. Winner table -> the bake decision. Refuses a
# stale ship (tree markers). W14 quiesce before, trap restore after
# (GPU probe — NEVER concurrent with a serving gate leg). Artifacts ->
# /root/build/p147/outA14. Exit 0 = correctness PASS (timings bank
# regardless — measurement, not a gate).
set -uo pipefail
R=/root/build
P=$R/p147
PATCHED=$P/p173_triton_mx_decode.py
PROBE=$P/p173_v6sweep.py
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25z}
OUT=$P/outA14

[ -f "$PATCHED" ] || { echo "R173_STALE_TREE (no patched kernel at $PATCHED) -- ship first"; exit 1; }
[ -f "$PROBE" ] || { echo "R173_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "_V5_SPLIT_DEFAULT" "$PATCHED" \
  || { echo "R173_STALE_TREE (patched kernel lacks the v6 line)"; exit 1; }
grep -q "ARC_MXKV_V5_SPLITS" "$PATCHED" \
  || { echo "R173_STALE_TREE (patched kernel lacks the sweep env knobs)"; exit 1; }
grep -q "p173_v6sweep" "$PROBE" \
  || { echo "R173_STALE_TREE (probe identity marker missing)"; exit 1; }

mkdir -p "$OUT" $P/logs

w14_restore() {
  docker rm -f arc-p173 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r173.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r173:A14] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R173_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p173 >/dev/null 2>&1 || true

echo "=== [r173:A14] v6 sweep in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p173 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$P":/p147:ro -v "$OUT":/out \
  --entrypoint /bin/bash \
  $IMG -c 'set -e; \
    SP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv; \
    rm -rf /work && mkdir -p /work && cp "$SP"/*.py /work/; \
    cp /p147/p173_triton_mx_decode.py /work/triton_mx_decode.py; \
    cd /tmp && MXKV_DIR=/work /opt/venv/bin/python /p147/p173_v6sweep.py' \
  > $P/logs/legA14.out 2>&1
rc=$?
echo "=== [r173:A14] sweep exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[|A14_V6" $P/logs/legA14.out | tee $OUT/legA14_summary.txt || true

cd "$OUT"
sha256sum legA14_summary.txt $P/logs/legA14.out 2>/dev/null > SHA256SUMS || true
echo "R173_LEGA14_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
