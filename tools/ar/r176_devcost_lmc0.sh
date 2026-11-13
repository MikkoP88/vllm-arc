#!/bin/bash
# r176_devcost_lmc0.sh -- P6.47 A15: K2 device-cost probe + the LMC0
# TTFT discriminator, one GPU window (owner parity directive 2026-11-13).
#
# phase 0 (bare container, cert25za): p174_devcost — splits the v6
# decode call's ms into wall / device-event / amortized-launcher
# faces + torch.profiler kernel table + identity-bt pointer-chase
# share. Decides whether K2 (kernel restructure) attacks device time
# or the launcher/combine overhead.
#
# phase 1 (serving legs): mxfp8 boots — (A) default-posture replica of
# r172's m8 leg (the PURE v6 e2e A/B), then (B) ARC_LMC=0 — the
# r172-banked evidence (serve-log LMCache store lines: q8 2264 ms /
# m8 3386 ms / m4 0 on the prefill critical path; TTFT 3.07/6.04/
# 2.92 s) says the m8 TTFT +2.97 s delta is dominated by LMCache
# stores. Leg B measures TTFT+ITL at {8192, 32768} x n1 with LMC off
# (r168 posture precedent: auto-off already exists for the e2m1-V
# lanes). Clean + faster -> posture amendment candidate (evidence-
# cited, full battery at convergence).
#
# w14_restore carries the r170b-incident guard: skip when any arc-*
# container is Up. Artifacts -> /root/build/p647/devcostA15/. Exit 0
# = probe finite + leg rungs banked (measurements, not gates).
set -uo pipefail
R=/root/build
P=$R/p647/devcostA15
IMG=${IMG:-vllm-arc:v0.30.0-b70-cert25za}
PROBE=$R/p147/p174_devcost.py
CNAME_P=arc-p174
CNAME_S=arc-r176
MT=600
SERVED=qwen3.8-27b-fp8
FAILED=""

[ -f "$PROBE" ] || { echo "R176_STALE_TREE (no probe at $PROBE) -- ship first"; exit 1; }
grep -q "p174_devcost" "$PROBE" \
  || { echo "R176_STALE_TREE (probe identity marker missing)"; exit 1; }
grep -q "len(_here.parents) > 3" "$PROBE" \
  || { echo "R176_STALE_TREE (probe lacks the parents[3] guard)"; exit 1; }
mkdir -p "$P/logs" "$P/lmc0"

w14_restore() {
  docker rm -f $CNAME_P $CNAME_S >/dev/null 2>&1
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q '^arc-'; then
    echo "W14_RESTORE_SKIPPED (arc leg in flight -- its trap owns the restore)"
    return 0
  fi
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p647r176.log 2>&1 &
  echo "W14_RESTORED $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
trap w14_restore EXIT

echo "=== [r176:A15] W14 quiesce $(date +%H:%M:%S) ==="
docker stop radiance-test >/dev/null 2>&1 || true
Q=0
while [ $Q -lt 90 ]; do
  ss -tln 2>/dev/null | grep -q ':8000 ' || break
  sleep 2; Q=$((Q+1))
done
ss -tln 2>/dev/null | grep -q ':8000 ' && echo "R176_WARN_PORT8000_STILL_HELD"
sleep 5

# ---- phase 0: device-cost probe (bare container)
echo "=== [r176:A15] devcost probe in $IMG $(date +%H:%M:%S) ==="
docker rm -f $CNAME_P >/dev/null 2>&1 || true
docker run --rm --name $CNAME_P \
  --device /dev/dri --group-add video \
  -e ZES_ENABLE_SYSMAN=1 -e DISABLE_VECLIB=1 \
  -e LD_LIBRARY_PATH=/opt/venv/lib:/opt/venv/lib/python3.12/site-packages/torch/lib \
  -v "$R/p147":/p147:ro -v "$P":/out \
  --entrypoint /bin/bash \
  $IMG -c 'cp /p147/p174_devcost.py /tmp/ && cd /tmp && /opt/venv/bin/python p174_devcost.py' \
  > "$P/devcost.txt" 2>&1
PRC=$?
echo "=== [r176:A15] devcost exit rc=$PRC $(date +%H:%M:%S) ==="
grep -E "^\[|A15_" "$P/devcost.txt" | head -40
[ "$PRC" = 0 ] || FAILED="$FAILED devcost"

# ---- phase 1+2: two m8 serving legs, p139 rungs {8192, 32768} x n1
#   leg A (replica): default posture (LMC on) — the PURE v6 e2e A/B vs
#     r172's cert25z m8 leg (byte-exact posture + rungs; expect
#     173.3 -> ~117 ms @32K if the 1.66x kernel win carries).
#   leg B (lmc0): ARC_LMC=0 — the TTFT discriminator.
boot_wait() {
  local TAG=$1 i=0 ok="" SEEN=0 ABSENT=0
  while [ $i -lt 210 ]; do
    curl -sf -m 4 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    if ! docker inspect $CNAME_S >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 5 ] && { echo "[r176:$TAG] CONTAINER_ABSENT poll=$i"; break; }
    else ABSENT=0; fi
    if docker exec $CNAME_S pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local tb
      tb=$(docker exec $CNAME_S grep -c "Traceback (most recent call last)" /root/serve_full.log 2>/dev/null || echo 0)
      [ "${tb:-0}" -gt 0 ] && { echo "[r176:$TAG] TRACEBACK poll=$i"; break; }
      [ "$SEEN" = 1 ] && [ $i -gt 12 ] && { echo "[r176:$TAG] ENGINE_DEAD poll=$i"; break; }
    fi
    i=$((i + 1)); sleep 10
  done
  [ "$ok" = "1" ]
}

run_leg() {
  # $1=leg name; rest = env pairs
  local LEG=$1; shift
  local CD=$P/$LEG
  mkdir -p "$CD"
  echo "=== [r176:$LEG] boot $(date +%H:%M:%S) env=[$*] ==="
  nohup env ARC_IMAGE="$IMG" ARC_NAME=$CNAME_S "$@" \
    bash $R/serve_arc_b70.sh > "$CD/boot.txt" 2>&1 &
  if ! boot_wait "$LEG"; then
    echo "R176_BOOT_FAIL_$LEG -- banking log"
    docker cp $CNAME_S:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
    docker rm -f $CNAME_S >/dev/null 2>&1
    FAILED="$FAILED ${LEG}_boot"; echo "BOOT_FAIL" > "$CD/outcome.txt"
    return 0
  fi
  echo "HEALTH_OK_r176_$LEG $(date +%H:%M:%S)"
  docker cp $R/p139_conc.py $CNAME_S:/root/p139_conc.py
  grep -h "ARC_SERVE_LAUNCHED" "$CD/boot.txt" | tail -1 > "$CD/posture.txt"
  docker exec $CNAME_S grep -c "LMCacheConnector" /root/serve_full.log 2>/dev/null \
    > "$CD/lmc_lines.txt" || echo 0 > "$CD/lmc_lines.txt"
  curl -s -m 120 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$SERVED\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" > /dev/null
  sleep 3
  for CTX in 8192 32768; do
    local TAG=c${CTX}_n1
    docker exec $CNAME_S rm -f /root/p139_conc_$TAG.json
    docker exec -d $CNAME_S /opt/venv/bin/python /root/p139_conc.py \
      "$CTX" "$MT" 1 "$TAG" "$SERVED" 0.0
    local cap=$(( 300 + CTX / 8192 * 90 + 120 )) i=0
    while [ $i -lt $((cap / 4)) ]; do
      docker exec $CNAME_S test -f /root/p139_conc_$TAG.json && break
      docker inspect $CNAME_S >/dev/null 2>&1 || break
      sleep 4; i=$((i+1))
    done
    if docker exec $CNAME_S test -f /root/p139_conc_$TAG.json 2>/dev/null; then
      docker cp $CNAME_S:/root/p139_conc_$TAG.json "$CD/" >/dev/null 2>&1
      echo "CONC_OK_${LEG}_$TAG"
    else
      echo "CONC_TIMEOUT_${LEG}_$TAG"; FAILED="$FAILED ${LEG}_$TAG"
    fi
  done
  docker exec $CNAME_S bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  docker cp $CNAME_S:/root/serve_full.log "$CD/serve.log" >/dev/null 2>&1 || true
  docker rm -f $CNAME_S >/dev/null 2>&1 || true
}

QWEN_BASE="ARC_MODEL=/models/qwen3.8-27b-fp8 ARC_SERVED=qwen3.8-27b-fp8 ARC_GRAPH=1 ARC_SPEC=3 ARC_FP8=1 ARC_KVDTYPE=mxfp8"
run_leg replica $QWEN_BASE
run_leg lmc0 $QWEN_BASE ARC_LMC=0

cd "$P"
find . -name 'p139_conc_*.json' -o -name 'devcost.txt' -o -name 'posture.txt' \
  -o -name 'lmc_lines.txt' -o -name 'serve.log' | sort | xargs sha256sum > SHA256SUMS 2>/dev/null || true
echo "R176_DONE failed=[$FAILED] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ -z "$FAILED" ]
