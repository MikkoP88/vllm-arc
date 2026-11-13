#!/bin/bash
# r178_k2ab.sh -- P6.47 K2 cycle 1: packed-fp16 dequant A/B probe
# (p176_k2ab.py) on cert25za with the shipped triton_mx_decode.py
# bind-mounted over the installed copy — the exact shipping path, no
# bake needed for the cycle. Stale-tree guards require the K2 markers
# in the shipped kernel and probe.
#
# w14_restore carries the r170b-incident guard: skip when any arc-*
# container is Up. Artifacts -> /root/build/p647/k2cycleA17/. Exit 0
# = probe finite + F0 correctness clean (d <= 2e-3).
set -uo pipefail
R=/root/build
P=$R/p647/k2cycleA17
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25za}
PROBE=$R/p147/p176_k2ab.py
KERNEL=$R/p647/k2cycle/triton_mx_decode.py
INSTALLED=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv/triton_mx_decode.py
CNAME=arc-p176

[ -f "$PROBE" ] || { echo "R178_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
[ -f "$KERNEL" ] || { echo "R178_STALE_TREE (no patched kernel at $KERNEL) -- ship first"; exit 1; }
grep -q "p176_k2ab" "$PROBE" \
  || { echo "R178_STALE_TREE (probe identity marker missing)"; exit 1; }
for MARKER in "_f16_scale" "F16OUT" "ARC_MXKV_K2"; do
  grep -q "$MARKER" "$KERNEL" \
    || { echo "R178_STALE_TREE (kernel lacks $MARKER)"; exit 1; }
done
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r178.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r178:K2ab] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R178_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r178:K2ab] packed-fp16 A/B in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  -v "$KERNEL":"$INSTALLED":ro \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p176_k2ab.py /tmp/ && cd /tmp && /opt/venv/bin/python p176_k2ab.py' \
  > "$P/k2ab.txt" 2>&1
PRC=$?
echo "=== [r178:K2ab] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|K2_" "$P/k2ab.txt" | head -50
[ "$PRC" = 0 ] || { echo "R178_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "K2_AB_PASS" "$P/k2ab.txt" \
  || { echo "R178_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum k2ab.txt > SHA256SUMS 2>/dev/null || true
echo "R178_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
