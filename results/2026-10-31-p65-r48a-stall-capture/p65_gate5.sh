#!/bin/bash
# p65_gate5.sh — P6.5 r48e: conviction leg for [vii] chunk-store path.
#   Boot B: certified posture MINUS the LMCache connector (ARC_LMC=0 — env
#   gate in serve_arc_b70.sh:165) PLUS --long-prefill-token-threshold 2048.
#   r48d convicted the chunk/mamba store path (decode stores exonerated:
#   SDC=False still froze, U32k 69.77 s). If the store path is the whole
#   story, no-connector + 2048 chunks must run CLEAN:
#     U32k TTFT ~16-18 s, bystander gaps ~1.0-1.1 s FLAT (pure chunk-step
#     bound), U64k ~31-35 s at full prefill rate, run7n unchanged.
#   Any residual growth with no connector = a second, scheduler/mamba-side
#   mechanism (next leg would target _mamba_block_aligned_split / lookahead).
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48e.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

echo "=== [B1] serve copy: threshold 2048 (connector stays OFF via ARC_LMC=0) ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_nolmc_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_nolmc_lpt.sh || { echo "FATAL: sed failed"; exit 1; }

echo "=== [B2] boot (ARC_LMC=0) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70_nolmc_lpt.sh > $CT/boot_nolmc_lpt.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_nolmc_lpt.out || { echo "FATAL: serve launch failed"; cat $CT/boot_nolmc_lpt.out; exit 1; }
docker exec arc-test grep -c "kv_transfer_config\|LMCacheEngine" /root/serve_full.log || echo "CONFIRMED: no connector in engine"
docker exec arc-test grep -m1 -o "long_prefill_token_threshold[^,]*" /root/serve_full.log || true
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"

echo "=== [B3] capture (U32k + U64k; C32k meaningless without connector) ==="
cd $CT && P65_FACES="U32k,U64k" python3 $CT/p65_timeline.py $CT/p65_tl_nolmc_lpt.json 2>&1 | tee $CT/p65_sum_nolmc_lpt.txt

echo "=== [B4] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_nolmc_lpt.out

docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48e.log 2>/dev/null
echo "P65_LEG_R48E: DONE"
