#!/bin/bash
# r182_kdiag.sh -- P6.47 convergence decider: dequant-free / softmax-free
# diagnostic bounds probe (p179_kdiag.py) on cert25zb — the installed
# kernel already carries the K2/K3 lanes default-on (no bind-mount).
# Stale-tree guards: image tag present + fastcycleA20 evidence PASS +
# the probe's own identity asserts (P179_STALE_TREE).
#
# w14_restore carries the arc-in-flight guard. Artifacts ->
# /root/build/p647/kdiagA21/. Exit 0 = probe finite + F0 clean.
set -uo pipefail
R=/root/build
P=$R/p647/kdiagA21
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p179_kdiag.py
CNAME=arc-p182

[ -f "$PROBE" ] || { echo "R182_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p179_kdiag" "$PROBE" \
  || { echo "R182_STALE_TREE (probe identity marker missing)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "R182_STALE_TREE (no image $IMG)"; exit 1; }
grep -q "P647_FASTCYCLE_DONE fail=0" "$R/p647/fastcycleA20/launch.log" \
  || { echo "R182_NO_FASTCYCLE_EVIDENCE (fastcycleA20)"; exit 1; }
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r182.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r182:kdiag] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R182_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r182:kdiag] walk-residual bounds in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p179_kdiag.py /tmp/ && cd /tmp && /opt/venv/bin/python p179_kdiag.py' \
  > "$P/kdiag.txt" 2>&1
PRC=$?
echo "=== [r182:kdiag] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|KDIAG_" "$P/kdiag.txt" | head -40
[ "$PRC" = 0 ] || { echo "R182_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "KDIAG_PASS" "$P/kdiag.txt" \
  || { echo "R182_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum kdiag.txt > SHA256SUMS 2>/dev/null || true
echo "R182_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
