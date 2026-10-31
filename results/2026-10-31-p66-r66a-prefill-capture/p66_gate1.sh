#!/bin/bash
# p66_gate1.sh — P6.6 leg 1: SOLO prefill throughput curve, pure engine.
# Sweep --long-prefill-token-threshold in {none, 2048, 4096, 8192} with
# ARC_LMC=0 (no LMCache store/load interference) and NO bystanders: the
# TTFT of a single fresh prompt / prompt_tokens = clean engine prefill
# rate per chunk size. Places the [viii] engine-side share of the p65
# matrix on a curve and gives the P6.6 baseline (is ~2k tok/s the eager
# ceiling, and how much do small chunks cost?).
# Faces per boot: warmup -> U8k, U32k, U64k (unique seeds, greedy, 32 tok).
set -uo pipefail
R=/root/build
CT=$R/p66
IMG=vllm-arc:v0.30.0-b70-cert4
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p66a.log 2>&1 &
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

for THR in none 2048 4096 8192; do
  echo "=== boot thr=$THR (nolmc, graph, k3) ==="
  if [ "$THR" = "none" ]; then
    SRV=$R/serve_arc_b70.sh
  else
    SRV=$R/serve_arc_b70_p66_$THR.sh
    sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold '"$THR"' \\|' \
      $R/serve_arc_b70.sh > "$SRV"
    grep -c "long-prefill-token-threshold $THR" "$SRV" >/dev/null || { echo "FATAL: sed thr=$THR"; exit 1; }
  fi
  docker rm -f arc-test >/dev/null 2>&1
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
    bash "$SRV" > "$CT/boot_thr$THR.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot_thr$THR.out" || { echo "FATAL: launch thr=$THR"; cat "$CT/boot_thr$THR.out"; exit 1; }
  echo "LMC_OFF_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep -c LMCACHE' || echo 0) (must be 0)"
  wait_health
  cd $CT && python3 $CT/p66_solo_ttft.py $CT/p66_solo_thr$THR.json 2>&1 | tee $CT/p66_solo_thr$THR.txt
  docker exec arc-test cat /root/serve_full.log > "$CT/serve_thr$THR.log" 2>/dev/null
  docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1; sleep 10
  docker rm -f arc-test >/dev/null 2>&1; sleep 5
done

echo "=== cross-threshold table ==="
for THR in none 2048 4096 8192; do
  echo "-- thr=$THR:"; grep -E "TTFT=|warmup done" $CT/p66_solo_thr$THR.txt
done
echo "P66_GATE1: DONE"
