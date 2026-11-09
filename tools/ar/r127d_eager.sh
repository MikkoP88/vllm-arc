#!/bin/bash
# r127d_eager.sh -- EXP-2 retry: gemma m816 skinny ON, ARC_GRAPH=0 (eager).
# r127c exp2 boot-failed at poll 30 -- launched 11s after exp1 engine kill
# (port/LMC bind race; r127 legs settle via the full battery+leg_done flow).
# This retry: settle 45s after kill, docker cp serve.log on any failure.
set -uo pipefail
R=/root/build
P=$R/p633
IMG6=vllm-arc:v0.30.0-b70-arc6test

w14_start() {
  docker exec arc-test bash -c "pkill -f \"vllm serve\"" >/dev/null 2>&1
  sleep 8
  docker rm -f arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  setsid nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p633r127d.log 2>&1 < /dev/null &
}
trap w14_start EXIT

kill_arc() {
  docker exec arc-test bash -c "pkill -f \"vllm serve\"" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 5
}

gcount() {
  local c
  c=$(docker exec arc-test grep -c "$1" /root/serve_full.log 2>/dev/null)
  echo "${c:-0}" | head -1
}

boot_wait() {
  LAST_WARM=0; ok=""; SEEN=0; ABSENT=0
  for i in $(seq 1 1500); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
    if ! docker inspect arc-test >/dev/null 2>&1; then
      ABSENT=$((ABSENT+1)); [ "$ABSENT" -gt 8 ] && { echo "[$1] container ABSENT"; break; }
    else ABSENT=0; fi
    if docker exec arc-test pgrep -f EngineCore >/dev/null 2>&1; then SEEN=1; else
      local TB; TB=$(gcount "Traceback (most recent call last)")
      [ "$TB" -ge 1 ] 2>/dev/null && { echo "[$1] TRACEBACK at poll $i"; break; }
      [ "$SEEN" = 1 ] && [ "$i" -gt 24 ] && { echo "[$1] ENGINECORE died poll $i"; break; }
    fi
    local NW; NW=$(gcount "ARC XPU tune warmup: load")
    [ "$NW" != "$LAST_WARM" ] && { echo "[$1 boot] warm $NW/32 ($(date +%H:%M:%S))"; LAST_WARM=$NW; }
    sleep 5
  done
  [ "$ok" = "1" ]
}

echo "=== [exp2_eager retry] boot $(date +%H:%M:%S) skinny=on graph=0 ==="
docker stop radiance-test >/dev/null 2>&1; sleep 8
docker rm -f arc-test >/dev/null 2>&1; sleep 45
ARC_IMAGE=$IMG6 ARC_NAME=arc-test \
ARC_MODEL=/models/gemma4-31b ARC_SERVED=gemma4-31b \
ARC_QUANT=mxfp8 \
ARC_GMU=0.8 \
ARC_GRAPH=0 ARC_SPEC=0 ARC_LMC=0 ARC_RPC=1200 \
ARC_MAXLEN=40960 ARC_FP8=0 \
  bash $R/serve_arc_b70.sh
if ! boot_wait exp2; then
  echo "BOOT_FAILED exp2 -- serve traceback:"
  docker cp arc-test:/root/serve_full.log $P/r127d_eager_fail.log >/dev/null 2>&1 || true
  grep -n -B2 -A25 "Traceback (most recent call last" $P/r127d_eager_fail.log 2>/dev/null | tail -90 || true
  exit 1
fi
echo "--- exp2 HEALTH UP $(date +%H:%M:%S) ---"
sleep 90
curl -s -m 600 http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"gemma4-31b\",\"max_tokens\":48,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}]}" \
  > /tmp/probe_exp2.out 2>&1
python3 $R/r113_battery.py decode gemma4-31b 2>&1 | tail -10
echo "wrap=$(gcount \"skinny-M wrap installed\") first=$(gcount \"skinny-M route FIRST\") failed=$(gcount \"route FAILED\")"
docker exec arc-test grep -h "skinny-M route FIRST" /root/serve_full.log 2>/dev/null | sort -u | head -8
kill_arc
echo "=== [exp2] done $(date +%H:%M:%S); W14 restoring ==="
