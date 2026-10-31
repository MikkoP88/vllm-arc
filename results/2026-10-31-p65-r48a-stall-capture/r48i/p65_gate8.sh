#!/bin/bash
# p65_gate8.sh — P6.5 r48i (deferral probe leg) + r48h retry (nolmc 32k).
#
# r48i: cert3 + threshold 2048 + LMC_DEFER_CHUNK_STORE=1 + the installed
#   adapter patched in-container (docker cp of the arc p65 chunk-set store
#   deferral) BEFORE the engine's first lmcache import (~90 s into boot;
#   cp lands ~2 s after docker run). Pre-registered predictions
#   (VERDICT r48i section): U32k 19-25 s, C32k ~1.2 s, U64k 42-52 s,
#   bystander p50 <= 800 ms during U32k, ONE store row per long face
#   (vs 16), run7n 290-300/84-85.
# r48h: threshold 2048 + ARC_LMC=0, face U32k only — the 32k [vii-a]
#   isolation cell. First attempt (gate7 B1) died at container start (OCI
#   failure after boot A's rm -f under run7n load); this script drains
#   30 s between boots and re-checks.
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
ADP=/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48i.log 2>&1 &
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

md5sum $CT/adapter_installed.py $CT/adapter_p65.py

echo "=== [1] serve copy: threshold 2048 + DEFER env ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
    -e 's|^  LMC_ENV="export LMCACHE_USE_EXPERIMENTAL_API=True;|  LMC_ENV="export LMC_DEFER_CHUNK_STORE=1; export LMCACHE_USE_EXPERIMENTAL_API=True;|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_defer.sh
grep -c "long-prefill" $R/serve_arc_b70_defer.sh || { echo "FATAL: sed threshold failed"; exit 1; }
grep -c "LMC_DEFER_CHUNK_STORE" $R/serve_arc_b70_defer.sh || { echo "FATAL: sed defer failed"; exit 1; }

echo "=== [2] boot + immediate in-container adapter patch ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer.sh > $CT/boot_defer_r48i.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_defer_r48i.out || { echo "FATAL: serve launch failed"; cat $CT/boot_defer_r48i.out; exit 1; }
docker cp $CT/adapter_p65.py arc-test:$ADP || { echo "FATAL: docker cp failed"; exit 1; }
docker exec arc-test python3 -m py_compile $ADP || { echo "FATAL: py_compile failed on patched adapter"; exit 1; }
echo "PATCH_MARKER=$(docker exec arc-test grep -c 'arc p65 fork delta' $ADP)"

wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"
docker exec arc-test grep -m1 -o "long_prefill_token_threshold[^,]*" /root/serve_full.log || true

echo "=== [3] capture v5.1 (U32k FRESH + C32k replay + U64k FRESH under live bys) ==="
cd $CT && P65_FACES="U32k,C32k,U64k" P65_W2MT=3000 python3 $CT/p65_timeline.py $CT/p65_tl_defer_v51.json 2>&1 | tee $CT/p65_sum_defer_v51.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_defer_v51.txt
echo "STORE_ROWS=$(docker exec arc-test grep -c 'Storing KV cache' /root/serve_full.log || echo 0)"

echo "=== [4] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_defer.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48i.log 2>/dev/null

echo "=== [5] drain 30s, then r48h (thr2048 + ARC_LMC=0, U32k only) ==="
docker rm -f arc-test >/dev/null 2>&1
sleep 30
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_nolmc_lpt.sh
grep -c "long-prefill" $R/serve_arc_b70_nolmc_lpt.sh || { echo "FATAL: sed failed"; exit 1; }
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=0 \
  bash $R/serve_arc_b70_nolmc_lpt.sh > $CT/boot_nolmc_r48h.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_nolmc_r48h.out || { echo "FATAL: serve launch failed"; cat $CT/boot_nolmc_r48h.out; exit 1; }
docker exec arc-test grep -c "kv_transfer_config\|LMCacheEngine" /root/serve_full.log || echo "CONFIRMED: no connector in engine"
wait_health

echo "=== [6] capture v5.1 (U32k FRESH only) ==="
cd $CT && P65_FACES="U32k" python3 $CT/p65_timeline.py $CT/p65_tl_nolmc32k_v51.json 2>&1 | tee $CT/p65_sum_nolmc32k_v51.txt
grep "prompt sizes\|TTFT\|prompt_tokens=" $CT/p65_sum_nolmc32k_v51.txt

docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48h.log 2>/dev/null
echo "P65_LEG_R48IH: DONE"
