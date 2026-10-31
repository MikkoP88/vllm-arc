#!/bin/bash
# p65_gate12.sh — P6.5 r48l: LENGTH-GATED deferral validation (probe4).
#
# r48k proved the deferral live and winning at 64k (51.90 vs 56.10 s,
# p95 gap 1969->1598, 1664-class stores 122->0, one-shot 2x63232 finals)
# but REGRESSING 32k (27.06 vs 22.89 s): the ~0.5 GB one-shot store runs
# at ~0.24 GB/s contended, and r48h proved the per-chunk events it
# replaced are free at 32k (LMC == nolmc there). Fix: LMC_DEFER_MIN_TOKENS
# gate (prompt_len >= min), certified value 40960 — between the 32k face
# ([vii-a] = 0) and the 64k face ([vii-a] = +24.6 s @12 bys).
# Pre-registered predictions (VERDICT r48l):
#   - ZERO ARC_P65_DEFER rows for the U32k face; ~38 for U64k
#   - 1664-class stores RETURN for U32k (~19-22), absent for U64k;
#     one-shot ~63k store present
#   - U32k TTFT 22.5-23.5 s; W_U32k p50 ~700-900 ms (r48g/r48iv2 family)
#   - U64k TTFT 50.0-52.5 s; W_U64k p50 ~1100-1300, p95 <= 1700 ms
#   - run7n 290-300 / 83-84
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
PROBE=vllm-arc:v0.30.0-b70-cert3-p65probe4
LADP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48l.log 2>&1 &
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

echo "=== [1] probe4 image (gated lmcint deferral) ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  docker rm -f p65probe4 >/dev/null 2>&1
  docker run -d --name p65probe4 --entrypoint sleep $IMG 60 >/dev/null
  docker cp $CT/adapter_lmc_p65.py p65probe4:$LADP
  docker exec p65probe4 python3 -m py_compile $LADP || { echo "FATAL: py_compile"; exit 1; }
  docker commit -m "p65 probe4: length-gated chunk-store deferral" p65probe4 $PROBE >/dev/null
  docker rm -f p65probe4 >/dev/null 2>&1
fi
docker run --rm --entrypoint md5sum $PROBE $LADP
echo "MIN_TOKENS_MARKER=$(docker run --rm --entrypoint grep $PROBE -c LMC_DEFER_MIN_TOKENS $LADP)"

echo "=== [2] derive defer2 script (thr2048 + DEFER + MIN_TOKENS=40960) ==="
sed -e 's|^  --max-num-seqs \$ARC_MAXSEQS \\|  --max-num-seqs $ARC_MAXSEQS \\\n  --long-prefill-token-threshold 2048 \\|' \
    -e 's|^  LMC_ENV="export LMCACHE_USE_EXPERIMENTAL_API=True;|  LMC_ENV="export LMC_DEFER_CHUNK_STORE=1; export LMC_DEFER_MIN_TOKENS=40960; export LMCACHE_USE_EXPERIMENTAL_API=True;|' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_defer2.sh
grep -c "long-prefill" $R/serve_arc_b70_defer2.sh || { echo "FATAL: sed threshold"; exit 1; }
grep -c "LMC_DEFER_MIN_TOKENS" $R/serve_arc_b70_defer2.sh || { echo "FATAL: sed defer2"; exit 1; }

echo "=== [3] boot probe4 ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer2.sh > $CT/boot_defer_r48l.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_defer_r48l.out || { echo "FATAL: serve launch failed"; cat $CT/boot_defer_r48l.out; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"
echo "SRC_OVERRIDE=$(docker exec arc-test grep -c 'src-override OK' /root/serve_full.log || echo 0)"

echo "=== [4] capture v5.1 (U32k + U64k, W2MT=4000) ==="
cd $CT && P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $CT/p65_timeline.py $CT/p65_tl_defer5_v51.json 2>&1 | tee $CT/p65_sum_defer5_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_defer5_v51.txt

echo "=== [5] verdict ==="
echo "DEFER_ROWS=$(docker exec arc-test grep -c 'ARC_P65_DEFER' /root/serve_full.log || echo 0)"
echo "-- DEFER by prompt_len (only >=40960 must appear):"
docker exec arc-test sh -c "grep -o 'ARC_P65_DEFER req=[^ ]* in_len=[0-9]* prompt_len=[0-9]*' /root/serve_full.log | awk '{print \$NF}' | sort | uniq -c"
echo "STORE classes:"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head'

echo "=== [6] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_defer5.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48l.log 2>/dev/null
echo "P65_LEG_R48L: DONE"
