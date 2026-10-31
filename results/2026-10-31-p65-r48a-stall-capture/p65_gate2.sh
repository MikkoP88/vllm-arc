#!/bin/bash
# p65_gate2.sh — P6.5 r48b: fix-leg-1 (upstream knob long-prefill-token-
# threshold 2048) on cert3. Faces: corrected timeline harness (unique
# seeds, wave-2 coverage, 8k/32k/64k + cached) + run7n t0 throughput
# regression. Baseline = r48a capture (certified posture, threshold 0).
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
mkdir -p $CT

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48b.log 2>&1 &
}
trap w14_start EXIT

docker stop radiance-test >/dev/null 2>&1; sleep 2

echo "=== [1] rig-local serve copy with --long-prefill-token-threshold 2048 ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_lpt2048.sh
grep -n "long-prefill" $R/serve_arc_b70_lpt2048.sh || { echo "FATAL: sed failed"; exit 1; }

echo "=== [2] boot cert3 + LPT2048 (k3 + graphs + 25-var chain) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash $R/serve_arc_b70_lpt2048.sh \
  > $CT/boot_lpt.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_lpt.out || { echo "FATAL: serve launch failed"; cat $CT/boot_lpt.out; exit 1; }
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"
docker exec arc-test grep -c "long_prefill_token_threshold" /root/serve_full.log || true
docker exec arc-test grep -m1 -o "long_prefill_token_threshold[^,]*" /root/serve_full.log || true

echo "=== [3] stall-timeline capture (LPT2048 posture) ==="
cd $CT && python3 $CT/p65_timeline.py $CT/p65_timeline_lpt2048.json 2>&1 | tee $CT/p65_summary_lpt2048.txt

echo "=== [4] run7n t0 throughput regression (fresh-boot face) ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_lpt2048.out

echo "=== [5] engine log ==="
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48b.log 2>/dev/null
echo "P65_LEG_R48B: DONE"
