#!/bin/bash
# p65_gate4.sh — P6.5 r48d: store-queue-backpressure discriminator.
#   Mechanism under test [vii]: with --long-prefill-token-threshold 2048 the
#   per-step KV/mamba/decode store inflow marginally saturates the LMCache
#   save pipeline (measured drain ~2496 tok/s/worker during the r48cB freeze);
#   wait_for_save in the forward then stalls every step on the FIFO save
#   queue -> bystander gap grows 1.09->4.2 s over ~70 s; jam breaks when the
#   long prefill's last chunk stores.
#   Boot A (single variable): certified posture + threshold 2048 +
#   LMCACHE_SAVE_DECODE_CACHE=False (removes the 12-14 decode streams'
#   store inflow; chunk+mamba stores remain).
#     predicted [vii] TRUE : U32k TTFT ~20-30 s (vs 71.5), bystander p50
#                gap ~1.0-1.2 s flat (not growing), U64k full speed, run7n
#                unchanged.
#     predicted [vii] FALSE: freeze unchanged (~70 s, growing gaps) ->
#                chunk/mamba store path itself is the driver (next leg).
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48d.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

boot_and_wait() {  # $1=serve-script $2=bootlog
  docker rm -f arc-test >/dev/null 2>&1
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash "$1" > "$2" 2>&1
  grep ARC_SERVE_LAUNCHED "$2" || { echo "FATAL: serve launch failed"; cat "$2"; exit 1; }
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=== [A1] serve copy: threshold 2048 + SAVE_DECODE_CACHE=False ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
    -e 's/export LMCACHE_SAVE_DECODE_CACHE=True;/export LMCACHE_SAVE_DECODE_CACHE=False;/' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_sdcf_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_sdcf_lpt.sh || { echo "FATAL: sed threshold failed"; exit 1; }
grep -c "LMCACHE_SAVE_DECODE_CACHE=False" $R/serve_arc_b70_sdcf_lpt.sh || { echo "FATAL: sed sdc failed"; exit 1; }

echo "=== [A2] boot ==="
boot_and_wait $R/serve_arc_b70_sdcf_lpt.sh $CT/boot_sdcf_lpt.out
docker exec arc-test grep -m1 -o "save_decode_cache[^,]*" /root/serve_full.log || echo "WARN: sdc not in engine config dump"
docker exec arc-test grep -m1 -o "long_prefill_token_threshold[^,]*" /root/serve_full.log || true

echo "=== [A3] capture (U32k + U64k + C32k) ==="
cd $CT && P65_FACES="U32k,U64k,C32k" python3 $CT/p65_timeline.py $CT/p65_tl_sdcf_lpt.json 2>&1 | tee $CT/p65_sum_sdcf_lpt.txt

echo "=== [A4] run7n t0 regression ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_sdcf_lpt.out

docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48d.log 2>/dev/null
echo "P65_LEG_R48D: DONE"
