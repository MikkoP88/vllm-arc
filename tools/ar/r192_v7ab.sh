#!/bin/bash
# r192_v7ab.sh -- P6.47 Phase 7 attempt 1: v7 FA-shaped walk A/B probe
# (p183_v7ab.py) on cert25zb with the v7-bearing triton_mx_decode.py
# swapped over the installed one INSIDE the ephemeral --rm container
# (image untouched; the swap is cp-only and dies with the container).
# The T1 verdict (r191) put the kernel at 20-35x over ctrl FA at
# G1@32K with the serial bt->slot->gather chain as the floor; v7
# (_mx_decode_split_dot2) breaks it by hand: raw-tile double
# buffering, block-id prefetch at the loop top, raw-byte prefetch
# under the softmax/p.V cover, dequant-at-consume. Gate: correctness
# d <= 2e-3 vs v5 on every edge cell (V7AB_PASS); the T1-cell perf
# table banks for the threshold call (<= 1.0 ms at G1@32K) in the
# analysis. Offline docker-run pattern (r179/r189/r191); probe-only
# leg; quiesces W14 itself (port guard BEFORE quiesce, r188 law) and
# restores fullres on exit. Artifacts -> /root/build/p647/v7A26.
set -uo pipefail
R=/root/build
P=$R/p647/v7A26
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p183_v7ab.py
KFILE=$R/p147/triton_mx_decode.py
CNAME=arc-p183

[ -f "$PROBE" ] || { echo "R192_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p183_v7ab" "$PROBE" \
  || { echo "R192_STALE_TREE (probe identity marker missing)"; exit 1; }
[ -f "$KFILE" ] || { echo "R192_STALE_TREE (no kernel file at $KFILE) -- ship first"; exit 1; }
grep -q "_mx_decode_split_dot2" "$KFILE" \
  || { echo "R192_STALE_KERNEL (v7 marker missing from shipped file)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R192_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R192_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R192_PORT_HELD (unknown holder)"; exit 3
  fi
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
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r192.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r192:v7ab] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R192_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r192:v7ab] probe in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
SITE=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mx_kv
timeout 900 docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c "cp $SITE/triton_mx_decode.py $SITE/triton_mx_decode.v5orig.py \
    && cp /p147/triton_mx_decode.py $SITE/triton_mx_decode.py \
    && cp /p147/p183_v7ab.py /tmp/ && cd /tmp \
    && /opt/venv/bin/python -u p183_v7ab.py" \
  > "$P/v7ab.txt" 2>&1
PRC=$?
echo "=== [r192:v7ab] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|^CORRECTNESS|^==|^V7AB|mxkv_src" "$P/v7ab.txt" | tail -45
[ "$PRC" = 0 ] || { echo "R192_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "V7AB_PASS" "$P/v7ab.txt" \
  || { echo "R192_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum v7ab.txt > SHA256SUMS 2>/dev/null || true
echo "R192_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
