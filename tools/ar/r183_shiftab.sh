#!/bin/bash
# r183_shiftab.sh -- P6.47 final kernel rung: join vs shift flavor A/B
# on the K3 e2m1 lane (p180_shiftab.py) on cert25zb. r182 closed the
# m8 walk (converged — dot+load floor + irreducible cvt) and killed the
# LUT rung (byte-map share zero); the join/reshape cross-lane shuffle
# is the last suspected share of the m4/mkv4 residual. Flavor flip is
# the module attr (launcher reads it per call — no patching needed).
#
# w14_restore carries the arc-in-flight guard. Artifacts ->
# /root/build/p647/shiftabA22/. Exit 0 = probe finite + F0 clean.
set -uo pipefail
R=/root/build
P=$R/p647/shiftabA22
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p180_shiftab.py
CNAME=arc-p183

[ -f "$PROBE" ] || { echo "R183_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p180_shiftab" "$PROBE" \
  || { echo "R183_STALE_TREE (probe identity marker missing)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 \
  || { echo "R183_STALE_TREE (no image $IMG)"; exit 1; }
grep -q "KDIAG_PASS" "$R/p647/kdiagA21/kdiag.txt" \
  || { echo "R183_NO_KDIAG_EVIDENCE (r182 bank)"; exit 1; }
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r183.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r183:shiftab] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R183_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r183:shiftab] join-vs-shift A/B in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p180_shiftab.py /tmp/ && cd /tmp && /opt/venv/bin/python p180_shiftab.py' \
  > "$P/shiftab.txt" 2>&1
PRC=$?
echo "=== [r183:shiftab] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|SHIFT_AB_" "$P/shiftab.txt" | head -40
[ "$PRC" = 0 ] || { echo "R183_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "SHIFT_AB_PASS" "$P/shiftab.txt" \
  || { echo "R183_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum shiftab.txt > SHA256SUMS 2>/dev/null || true
echo "R183_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
