#!/bin/bash
# r177_k2stage.sh -- P6.47 K2: stage-1/stage-2 split probe of the v6
# decode call (p175_k2stage.py, one bare-container GPU window on
# cert25za). Decides K2's first lever: combine parallelism (math-
# neutral grid change) vs walk-body pipelining, plus the untested
# num_stages-on-dot face. Measurement only — no patch, no bake.
#
# w14_restore carries the r170b-incident guard: skip when any arc-*
# container is Up. Artifacts -> /root/build/p647/k2stageA16/. Exit 0
# = probe finite + F0 correctness clean.
set -uo pipefail
R=/root/build
P=$R/p647/k2stageA16
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25za}
PROBE=$R/p147/p175_k2stage.py
CNAME=arc-p175

[ -f "$PROBE" ] || { echo "R177_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p175_k2stage" "$PROBE" \
  || { echo "R177_STALE_TREE (probe identity marker missing)"; exit 1; }
grep -q "len(_here.parents) > 3" "$PROBE" \
  || { echo "R177_STALE_TREE (probe lacks the parents[3] guard)"; exit 1; }
mkdir -p "$P"

w14_restore() {
  docker rm -f $CNAME >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r177.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r177:K2] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R177_WARN_PORT8000_STILL_HELD"
sleep 5

echo "=== [r177:K2] stage-split probe in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME >/dev/null 2>&1 || true
docker run --rm --name $CNAME \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p175_k2stage.py /tmp/ && cd /tmp && /opt/venv/bin/python p175_k2stage.py' \
  > "$P/k2stage.txt" 2>&1
PRC=$?
echo "=== [r177:K2] probe exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|K2_" "$P/k2stage.txt" | head -60
[ "$PRC" = 0 ] || { echo "R177_PROBE_FAIL rc=$PRC"; exit 1; }
grep -q "K2_STAGE_DONE" "$P/k2stage.txt" \
  || { echo "R177_NO_VERDICT_MARKER"; exit 1; }
grep -q "FAIL" "$P/k2stage.txt" \
  && { echo "R177_F0_CORRECTNESS_FAIL"; exit 1; }
cd "$P"
sha256sum k2stage.txt > SHA256SUMS 2>/dev/null || true
echo "R177_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit 0
