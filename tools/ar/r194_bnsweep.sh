#!/bin/bash
# r194_bnsweep.sh -- P6.47 Phase 7 attempt 3: BLOCK_N re-sweep probe
# (p185_bnsweep.py) on cert25zb with the TILE_FLOOR-bearing
# triton_mx_decode.py swapped over the installed one INSIDE the
# ephemeral --rm container (image untouched; r192 mechanism). The v6
# BN=64 refutation ("double-N tile re-climbs the A9 spill slope")
# predates K2/K3 — the packed lanes eliminated the f32 dequant
# materialization that drove the spills ((64,256) f32 = 64 KB vs
# 32 KB packed f16). r193 showed the 32K residual is work/traffic-
# floored: per-tile fixed cost is exactly the term larger BN amortizes
# (half/quarter the tiles = half/quarter the bt-chain fixed costs,
# longer gather runs). BN rides the existing lawful knob
# ARC_MXKV_V5_BN (no code change); TILE_FLOOR via ARC_MXKV_V5_TFLOOR.
# Sweep: (BN, TF, SPLITS) in {(32,2,256) base, (64,2,256), (64,4,256),
# (128,2,256), (128,4,256)} at the T1 money cells. Gate: correctness
# d <= 2e-3 vs (32,2,256) (BNS_PASS); perf table banks for the winner
# call in the analysis. Offline docker-run pattern
# (r179/r189/r191/r192/r193); probe-only; quiesces W14 itself (port guard
# BEFORE quiesce, r188 law) and restores fullres on exit. Artifacts
# -> /root/build/p647/bnsA26.
set -uo pipefail
R=/root/build
P=$R/p647/bnsA26
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p185_bnsweep.py
KFILE=$R/p147/triton_mx_decode.py
CNAME=arc-p185

[ -f "$PROBE" ] || { echo "R194_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p185_bnsweep" "$PROBE" \
  || { echo "R194_STALE_TREE (probe identity marker missing)"; exit 1; }
[ -f "$KFILE" ] || { echo "R194_STALE_TREE (no kernel file at $KFILE) -- ship first"; exit 1; }
grep -q "ARC_MXKV_V5_BN" "$KFILE" \
  || { echo "R194_STALE_KERNEL (TILE_FLOOR marker missing from shipped file)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R194_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R194_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R194_PORT_HELD (unknown holder)"; exit 3
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
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r194.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r194:bnsweep] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R194_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r194:bnsweep] probe in $IMG $(date +%H:%M:%S) ==="
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
    && cp /p147/p185_bnsweep.py /tmp/ && cd /tmp \
    && /opt/venv/bin/python -u p185_bnsweep.py" \
  > "$P/tfloor.txt" 2>&1
PRC=$?
echo "=== [r194:bnsweep] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|^CORRECTNESS|^==|^TFL|mxkv_src" "$P/tfloor.txt" | tail -55
[ "$PRC" = 0 ] || { echo "R194_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "BNS_PASS" "$P/tfloor.txt" \
  || { echo "R194_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum tfloor.txt > SHA256SUMS 2>/dev/null || true
echo "R194_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
