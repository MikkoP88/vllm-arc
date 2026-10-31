#!/bin/bash
# p66_gate3.sh — P6.6 improvement leg 1b (knob): MNBT 16384. Evidence chain:
# r66a curve (thr2048 +24-27% vs none) + r66b decomposition (device-bound,
# small-M inefficiency; host glue <=10ms) => bigger chunks amortize both.
# Boot: cert4 (no probe needed — client-side TTFT is the metric), thr none,
# MNBT 16384, nolmc, graph, k3, greedy, solo faces U8k/U32k/U64k (unique
# seeds, same probe as r66a gate1 for exact comparability).
set -uo pipefail
R=/root/build
CT=$R/p66

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66j.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=== boot cert4 mnbt=16384 thr=none (nolmc) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 ARC_MNBT=16384 \
  bash $R/serve_arc_b70.sh > "$CT/boot_mnbt16k.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_mnbt16k.out" || { echo "FATAL: launch"; cat "$CT/boot_mnbt16k.out"; exit 1; }
grep "mnbt=16384" "$CT/boot_mnbt16k.out" || { echo "FATAL: mnbt not applied"; exit 1; }
wait_health
cd $CT && P66_FACES="U8k,U32k,U64k" python3 $CT/p66_solo_ttft.py $CT/p66_mnbt16k.json 2>&1 | tee $CT/p66_mnbt16k.txt
docker exec arc-test cat /root/serve_full.log > "$CT/serve_mnbt16k.log" 2>/dev/null
echo "=== boot cert4 mnbt=8192 thr=none (same-session A/B refresher) ==="
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=vllm-arc:v0.30.0-b70-cert4 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70.sh > "$CT/boot_mnbt8k_ab.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_mnbt8k_ab.out" || { echo "FATAL: launch ref"; exit 1; }
wait_health
cd $CT && P66_FACES="U8k,U32k,U64k" python3 $CT/p66_solo_ttft.py $CT/p66_mnbt8k_ab.json 2>&1 | tee $CT/p66_mnbt8k_ab.txt
docker exec arc-test cat /root/serve_full.log > "$CT/serve_mnbt8k_ab.log" 2>/dev/null
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
docker rm -f arc-test >/dev/null 2>&1; sleep 5
echo "P66_GATE3: DONE"
