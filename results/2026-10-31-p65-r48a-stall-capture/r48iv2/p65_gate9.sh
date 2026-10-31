#!/bin/bash
# p65_gate9.sh — P6.5 r48i-v2 (deferral probe, RACE-FREE) + r48h (3rd try).
#
# r48i (gate8) verified env+file but behaved as NO-DEFER (330 store events,
# attn-size distribution identical to r48g): the docker cp raced the
# engine's first import (vllm serve imports the adapter at config
# validation ~T+2-4s; cp lands ~T+2s; forked children inherit the old
# module from memory). v2 eliminates the race: docker commit a probe image
# with the patched adapter baked in, boot the normal script against it.
#   Activation check (behavioral): the 1664-token attn-span store class
#   must go 118 -> ~0 in serve_full.
# r48h: thr2048 + ARC_LMC=0, U32k — the 32k [vii-a] cell. Attempts 1-2
#   died at container start after rm -f of a loaded engine; this script
#   kills the serve processes FIRST and drains before rm.
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
PROBE=vllm-arc:v0.30.0-b70-cert3-p65probe
ADP=/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48iv2.log 2>&1 &
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

graceful_rm() {
  docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1
  sleep 10
  docker rm -f arc-test >/dev/null 2>&1
  sleep 5
}

echo "=== [1] probe image (patched adapter baked in) ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  docker rm -f p65probe >/dev/null 2>&1
  docker run -d --name p65probe --entrypoint sleep $IMG 60 >/dev/null
  docker cp $CT/adapter_p65.py p65probe:$ADP
  docker exec p65probe python3 -m py_compile $ADP || { echo "FATAL: py_compile"; exit 1; }
  docker commit -m "p65 probe: chunk-set store deferral (arc p65 fork delta)" p65probe $PROBE >/dev/null
  docker rm -f p65probe >/dev/null 2>&1
fi
echo "PROBE_MARKER=$(docker run --rm --entrypoint grep $PROBE -c 'arc p65 fork delta' $ADP)"
docker run --rm --entrypoint md5sum $PROBE $ADP

echo "=== [2] boot probe image (thr2048 + DEFER env) ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
    -e 's|^  LMC_ENV="export LMCACHE_USE_EXPERIMENTAL_API=True;|  LMC_ENV="export LMC_DEFER_CHUNK_STORE=1; export LMCACHE_USE_EXPERIMENTAL_API=True;|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_defer.sh
grep -c "long-prefill" $R/serve_arc_b70_defer.sh || { echo "FATAL: sed threshold"; exit 1; }
grep -c "LMC_DEFER_CHUNK_STORE" $R/serve_arc_b70_defer.sh || { echo "FATAL: sed defer"; exit 1; }
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer.sh > $CT/boot_defer_r48iv2.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_defer_r48iv2.out || { echo "FATAL: serve launch failed"; cat $CT/boot_defer_r48iv2.out; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"

echo "=== [3] capture v5.1 (U32k FRESH + C32k replay + U64k FRESH, W2MT=4000) ==="
cd $CT && P65_FACES="U32k,C32k,U64k" P65_W2MT=4000 python3 $CT/p65_timeline.py $CT/p65_tl_defer2_v51.json 2>&1 | tee $CT/p65_sum_defer2_v51.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_defer2_v51.txt
echo "ACTIVATION: attn-1664 store count (must be ~0 if defer live):"
docker exec arc-test grep -c "Stored 1664 out of total 1664" /root/serve_full.log || echo 0

echo "=== [4] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_defer2.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48iv2.log 2>/dev/null

echo "=== [5] graceful teardown, then r48h (thr2048 + ARC_LMC=0) ==="
graceful_rm
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_nolmc_lpt.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70_nolmc_lpt.sh > $CT/boot_nolmc_r48h.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_nolmc_r48h.out || { echo "FATAL: serve launch failed"; cat $CT/boot_nolmc_r48h.out; exit 1; }
docker exec arc-test grep -c "kv_transfer_config\|LMCacheEngine" /root/serve_full.log || echo "CONFIRMED: no connector"
wait_health

echo "=== [6] capture v5.1 (U32k FRESH only) ==="
cd $CT && P65_FACES="U32k" python3 $CT/p65_timeline.py $CT/p65_tl_nolmc32k_v51.json 2>&1 | tee $CT/p65_sum_nolmc32k_v51.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_nolmc32k_v51.txt

docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48h.log 2>/dev/null
echo "P65_LEG_R48IV2H: DONE"
