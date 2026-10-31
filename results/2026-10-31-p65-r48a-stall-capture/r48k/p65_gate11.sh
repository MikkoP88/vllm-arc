#!/bin/bash
# p65_gate11.sh — P6.5 r48k: the REAL deferral fix leg (probe3 image).
#
# r48j settled it: ROWLOG_LINES=0 in the lmcache-integration adapter while
# stores ran — the live adapter is the fork's lmcint copy
# (vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py), src-override-exec'd
# by lmcache_hma_connector._SrcOverrideFinder in place of vllm's vendored
# lmcache_integration. r48i/r48iv2/r48j patched the WRONG FILE (lmcache
# package's integration adapter — not in this boot's import path at all;
# its shared log strings made every liveness check pass).
# This leg patches the lmcint file (deferral hunk identical to the repo
# copy vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py + probe-only
# ARC_P65_DEFER warning per deferred row).
# Pre-registered predictions (VERDICT r48k):
#   - src-override OK lines name lmcint at boot
#   - ARC_P65_DEFER rows ~19 (U32k) + ~38 (U64k) per engine, then ONE
#     final store per face (a NEW ~30k/~60k-token store class appears;
#     1664-class collapses 122 -> ~0-4)
#   - U32k TTFT 21-23 s (32k [vii-a] ~= 0 per r48h; little change)
#   - U64k TTFT 56.10 -> 48-52 s; W_U64k p50 1238 -> 700-950 ms
#   - run7n 290-300 / 83-84
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
PROBE=vllm-arc:v0.30.0-b70-cert3-p65probe3
LADP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48k.log 2>&1 &
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

echo "=== [1] probe3 image (lmcint deferral baked in) ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  docker rm -f p65probe3 >/dev/null 2>&1
  docker run -d --name p65probe3 --entrypoint sleep $IMG 60 >/dev/null
  docker cp $CT/adapter_lmc_p65.py p65probe3:$LADP
  docker exec p65probe3 python3 -m py_compile $LADP || { echo "FATAL: py_compile"; exit 1; }
  docker commit -m "p65 probe3: lmcint chunk-set store deferral (the real adapter)" p65probe3 $PROBE >/dev/null
  docker rm -f p65probe3 >/dev/null 2>&1
fi
docker run --rm --entrypoint md5sum $PROBE $LADP
echo "DEFER_MARKER=$(docker run --rm --entrypoint grep $PROBE -c defer_chunk_store $LADP)"

echo "=== [2] boot probe3 (thr2048 + DEFER env) ==="
grep -c "LMC_DEFER_CHUNK_STORE" $R/serve_arc_b70_defer.sh || { echo "FATAL: defer script missing"; exit 1; }
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer.sh > $CT/boot_defer_r48k.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_defer_r48k.out || { echo "FATAL: serve launch failed"; cat $CT/boot_defer_r48k.out; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"
echo "SRC_OVERRIDE=$(docker exec arc-test grep -c 'src-override OK' /root/serve_full.log || echo 0)"

echo "=== [3] capture v5.1 (U32k + U64k, W2MT=4000 — r48iv2 A/B) ==="
cd $CT && P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $CT/p65_timeline.py $CT/p65_tl_defer4_v51.json 2>&1 | tee $CT/p65_sum_defer4_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_defer4_v51.txt

echo "=== [4] the fix verdict ==="
echo "DEFER_ROWS=$(docker exec arc-test grep -c 'ARC_P65_DEFER' /root/serve_full.log || echo 0)"
docker exec arc-test grep 'ARC_P65_DEFER' /root/serve_full.log | head -5
docker exec arc-test grep 'ARC_P65_DEFER' /root/serve_full.log | tail -3
echo "STORE classes:"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head'

echo "=== [5] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_defer4.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48k.log 2>/dev/null
echo "P65_LEG_R48K: DONE"
