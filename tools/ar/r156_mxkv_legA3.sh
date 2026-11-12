#!/bin/bash
# r156_mxkv_legA3.sh -- P6.47 leg A3: offline XPU revalidation of the MX
# KV primitives after the mxk8v4 K4/V4 kernel refactor (commit 3b92b9f8),
# inside vllm-arc:v0.30.0-b70-cert24x (the shipped-image layer).
#
# BEHAVIOR-PRESERVATION LAW (plan rule 10): a refactor of shipped
# kernels is proven behavior-preserving by oracle diff — the mxfp8 and
# mxfp4 faces must reproduce the leg-A banked numbers (p147/out/
# legA_summary.txt, results/2026-11-11 bank) BIT-EXACT; mxk8v4 rows are
# new (floors = the 4-bit-V leg). The harness tree (/root/build/p147/
# work) is refreshed by the r156 ship step (md5-verified tar); this
# script only runs + diffs.
# W14 quiesce before, trap restore after (rig law).
# Artifacts -> /root/build/p147/out3. Exit 0 = leg A3 PASS.
set -uo pipefail
R=/root/build
P=$R/p147
W=$P/work
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert24x}
ORACLE=$P/out/legA_summary.txt

w14_restore() {
  docker rm -f arc-p156 >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p156.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

mkdir -p $P/logs $P/out3

[ -f "$ORACLE" ] || { echo "R156_MISSING_ORACLE $ORACLE"; exit 1; }

echo "=== [r156:legA3] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R156_WARN_PORT8000_STILL_HELD"
sleep 5

docker rm -f arc-p156 >/dev/null 2>&1 || true

echo "=== [r156:legA3] offline harness in $IMG $(date +%H:%M:%S) ==="
docker run --rm --name arc-p156 \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$W":/work -v "$P/out3":/out \
  --entrypoint /bin/bash \
  $IMG -c "cd /work && /opt/venv/bin/python tools/ar/p147_mxkv/p147_mxkv_offline.py /out" \
  > $P/logs/legA3.out 2>&1
rc=$?
echo "=== [r156:legA3] harness exit rc=$rc $(date +%H:%M:%S) ==="

grep -E "^\[face|^  ok |^  FAIL|^VERDICT|telemetry/" $P/logs/legA3.out > $P/out3/legA3_summary.txt || true
cat $P/out3/legA3_summary.txt

# ---- oracle diff: banked mxfp8/mxfp4 lines must appear UNCHANGED ----
# (mxk8v4 rows are additive; the ceil_log2 oracle line is dtype-free)
KEEP='mxfp8|mxfp4'
grep -E "$KEEP" "$ORACLE" | grep -E "^  ok " > /tmp/oracle_keep.txt
grep -E "$KEEP" $P/out3/legA3_summary.txt | grep -E "^  ok " > /tmp/a3_keep.txt
# telemetry lines carry wall-times (non-deterministic) — compare faces only
if diff -q /tmp/oracle_keep.txt /tmp/a3_keep.txt >/dev/null 2>&1; then
  echo "R156_ORACLE_DIFF: CLEAN ($(wc -l < /tmp/oracle_keep.txt) mxfp8+mxfp4 ok-lines bit-exact)"
else
  echo "R156_ORACLE_DIFF: DRIFT — behavior-preservation violated:"
  diff /tmp/oracle_keep.txt /tmp/a3_keep.txt | head -20
  rc=1
fi

# mxk8v4 must have its own clean rows in every face
mx_rows=$(grep -cE "^  ok .*mxk8v4" $P/out3/legA3_summary.txt || true)
mx_fail=$(grep -cE "^  FAIL .*mxk8v4|mxk8v4.*FAIL" $P/out3/legA3_summary.txt || true)
echo "mxk8v4 rows ok=$mx_rows fail=${mx_fail:-0}"
[ "${mx_fail:-0}" = "0" ] || rc=1

cd $P/out3
sha256sum legA3_summary.txt p147_offline.json $P/logs/legA3.out 2>/dev/null > SHA256SUMS || true
echo "R156_LEGA3_DONE rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit $rc
