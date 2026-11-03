#!/bin/bash
# p618e — cert7 performance faces: the U-bar TTFT ladder + bystander-stall
# timeline on (P1) the certified default fp8 posture and (P2) the
# ARC_MNBT=4096 first-choice stall lever. Same harness as the cert6 gate
# face (P65_FACES="U32k,U64k" P65_W2MT=4000, p65_timeline.py v5.1).
# t0/t1.0/spec/integrity/long-return are already banked in the cert7 gate
# (results/2026-11-03-p618-cert7/). W14 restored by trap.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert7
CTROOT=$R/p618/cert7/perf
mkdir -p "$CTROOT"
cd "$CTROOT"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p618e.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1
sleep 20

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
}

timeline() {  # $1 = tag
  P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $R/p65/p65_timeline.py \
    "$CTROOT/p65_tl_$1.json" > "$CTROOT/p65_sum_$1.txt" 2>&1
  tail -30 "$CTROOT/p65_sum_$1.txt"
}

perf_leg() {  # $1=tag $2=mnbt ("8192"|"4096")
  local TAG=$1 MNBT=$2
  echo "=== P618E PERF LEG $TAG (mnbt=$MNBT) $(date +%H:%M:%S) ==="
  kill_arc
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 ARC_MNBT=$MNBT \
    bash $R/serve_arc_b70.sh > "boot_$TAG.out" 2>&1
  grep ARC_SERVE_LAUNCHED "boot_$TAG.out" || { echo "FATAL: launch failed"; cat "boot_$TAG.out"; exit 1; }
  grep ARC_SERVE_LAUNCHED "boot_$TAG.out" | grep -o "mnbt=[0-9]*"
  wait_health | tee "health_$TAG.txt"
  docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > "posture_$TAG.txt" || true
  cat "posture_$TAG.txt"
  sleep 30
  timeline "$TAG"
  docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' > "erraudit_$TAG.txt" 2>/dev/null || true
  cat "erraudit_$TAG.txt"
  docker exec arc-test cat /root/serve_full.log > "serve_full_$TAG.log" 2>/dev/null || true
  echo "LEG $TAG DONE $(date +%H:%M:%S)"
}

perf_leg default 8192
perf_leg mnbt4096 4096
echo "P618E PERF COMPLETE $(date +%H:%M:%S)"
