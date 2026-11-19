#!/bin/bash
# r193_tfloor.sh -- P6.47 Phase 7 attempt 2: deep-split re-sweep probe
# (p184_tfloor.py) on cert25zb with the TILE_FLOOR-bearing
# triton_mx_decode.py swapped over the installed one INSIDE the
# ephemeral --rm container (image untouched; r192 mechanism). The v6
# >=4-serial-tile floor was tuned pre-K2/K3 (per-tile scalar-ALU
# dominated); the packed lanes cut that cost and the r192 toolchain
# verdict closed in-language prefetch (num_stages no-op + manual
# double buffering counterproductive) — so attempt 2 REMOVES serial
# chain tiles instead: TILE_FLOOR 1-2 raises s_eff (deep splits ride
# concurrency; combine cost rises — the sweep finds the U-curve min).
# TILE_FLOOR is bit-frozen at the certified default 4; the env knob
# (ARC_MXKV_V5_TFLOOR, call-time read, capture law) only sweeps here.
# Gate: correctness d <= 2e-3 vs (4,256) (TFL_PASS); perf table banks
# for the winner call in the analysis. Offline docker-run pattern
# (r179/r189/r191/r192); probe-only; quiesces W14 itself (port guard
# BEFORE quiesce, r188 law) and restores fullres on exit. Artifacts
# -> /root/build/p647/tflA26.
set -uo pipefail
R=/root/build
P=$R/p647/tflA26
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25zb}
PROBE=$R/p147/p184_tfloor.py
KFILE=$R/p147/triton_mx_decode.py
CNAME=arc-p184

[ -f "$PROBE" ] || { echo "R193_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p184_tfloor" "$PROBE" \
  || { echo "R193_STALE_TREE (probe identity marker missing)"; exit 1; }
[ -f "$KFILE" ] || { echo "R193_STALE_TREE (no kernel file at $KFILE) -- ship first"; exit 1; }
grep -q "ARC_MXKV_V5_TFLOOR" "$KFILE" \
  || { echo "R193_STALE_KERNEL (TILE_FLOOR marker missing from shipped file)"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "R193_NO_IMAGE $IMG"; exit 1; }
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-r'; then
  echo "R193_BUSY (arc leg in flight)"; exit 2
fi
if ss -tln 2>/dev/null | grep -q ':8000 '; then
  if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'radiance-test'; then
    echo "R193_PORT_HELD (unknown holder)"; exit 3
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
  nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p647r193.log 2>&1 &
  echo "W14_RESTORE_FIRED_fullres $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r193:tfloor] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R193_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r193:tfloor] probe in $IMG $(date +%H:%M:%S) ==="
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
    && cp /p147/p184_tfloor.py /tmp/ && cd /tmp \
    && /opt/venv/bin/python -u p184_tfloor.py" \
  > "$P/tfloor.txt" 2>&1
PRC=$?
echo "=== [r193:tfloor] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|^CORRECTNESS|^==|^TFL|mxkv_src" "$P/tfloor.txt" | tail -55
[ "$PRC" = 0 ] || { echo "R193_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "TFL_PASS" "$P/tfloor.txt" \
  || { echo "R193_VERDICT_FAIL"; exit 1; }
cd "$P"
sha256sum tfloor.txt > SHA256SUMS 2>/dev/null || true
echo "R193_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
