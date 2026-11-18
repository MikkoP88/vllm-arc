#!/bin/bash
# r189_deqv2ab.sh -- P6.47 decode-parity fast-cycle 2: whole-cache
# dequant kernel v2 A/B probe (p182_dequantv2.py) on cert25zb. r185
# ramp forensics pinned the 3.3-3.7 s prefill-interleave stalls on BOTH
# MX lanes to mx_dequant_cache (single-warp per-(token,head) v1, no
# K2/K3 lanes, no tiling) running once per chunked-prefill chunk over
# the whole cached prefix (O(ctx^2)). v2 = token tiles reusing the
# certified _mx_dequant_block helper + packed lanes, bit-identical by
# the K2/K3 single-rounding law. Probe-only leg: the winner lands in
# mx_dequant.py only after this banks PASS. Offline docker-run pattern
# (r179); no serve boot, no kernel bind-mount (cert25zb ships the
# K2/K3 markers in its installed triton_mx_decode.py).
# Artifacts -> /root/build/p647/deqv2A26/. Exit 0 = DEQV2_PASS.
set -uo pipefail
R=/root/build
P=$R/p647/deqv2A26
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p182_dequantv2.py
CNAME=arc-p182

[ -f "$PROBE" ] || { echo "R189_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p182_dequantv2" "$PROBE" \
  || { echo "R189_STALE_TREE (probe identity marker missing)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R189_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R189_BUSY (arc leg in flight)"; exit 2
fi
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  # canonical full recreate (radiance-test removed by the external actor
  # 2026-11-13; docker start + serve_only can never restore it)
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r189.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r189:deqv2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
sleep 5

echo "=== [r189:deqv2] probe in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p182_dequantv2.py /tmp/ && cd /tmp && /opt/venv/bin/python p182_dequantv2.py' \
  > "$P/deqv2.txt" 2>&1
PRC=$?
echo "=== [r189:deqv2] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|^  |DEQV2" "$P/deqv2.txt" | tail -40
[ "$PRC" = 0 ] || { echo "R189_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "DEQV2_PASS" "$P/deqv2.txt" \
  || { echo "R189_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum deqv2.txt > SHA256SUMS 2>/dev/null || true
echo "R189_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
