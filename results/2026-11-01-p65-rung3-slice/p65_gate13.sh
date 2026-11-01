#!/bin/bash
# p65_gate13.sh — P6.13 rung-3 sliced-store A/B gate (PREPARED, NOT RUN;
# owner-attended-window item). Rig-local script, gate12 lineage: the
# driver session scp's the repo adapter to $CT/adapter_p73.py first, then
# executes this on the rig (rig manual §9; address lives in the untracked
# layer).
#
# Leg: cert4 + repo lmcint adapter (rung-2 deferral + rung-3
#      _lmc_slice_store_window) committed into probe image p73probe.
#      Posture = r48l's thr2048+LMC+DEFER+MIN_TOKENS=40960
#      + LMC_STORE_SLICE_TOKENS=4096.
# Bars + pre-registered decision rule: DESIGN.md §5
# (results/2026-11-01-p65-rung3-slice/). Predictions:
#   - store histogram: 63232-class == 0 for the U64k face;
#     ~38+ 3328-class windows (2 workers x 19); 1664-class returns for
#     U32k (below MIN_TOKENS => deferral+slicing never engage)
#   - U64k TTFT 47.5-50.5 s (one-shot leaves the pre-first-token step);
#     W_U64k max <= 600 ms; ~19 drain steps at base+~300 ms
#   - U32k family unchanged (22.5-23.5 s); run7n 290-300 / 83-84
#   - ADOPT iff max<=600 AND TTFT<=r48l bar AND run7n +-5% AND
#     integrity 5/5 AND U32k family unchanged; else RED, keep default-OFF.
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert4
PROBE=vllm-arc:v0.30.0-b70-cert4-p73probe
LADP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_p73.log 2>&1 &
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

echo "=== [0] input check ==="
md5sum $CT/adapter_p73.py
grep -c "_lmc_slice_store_window" $CT/adapter_p73.py || { echo "FATAL: repo adapter missing slice helper"; exit 1; }

echo "=== [1] probe image (cert4 + rung-3 adapter, docker-commit law) ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  docker rm -f p73probe >/dev/null 2>&1
  docker run -d --name p73probe --entrypoint sleep $IMG 60 >/dev/null
  docker cp $CT/adapter_p73.py p73probe:$LADP
  docker exec p73probe python3 -m py_compile $LADP || { echo "FATAL: py_compile"; exit 1; }
  docker commit -m "p73 probe: sliced final-span store (rung 3)" p73probe $PROBE >/dev/null
  docker rm -f p73probe >/dev/null 2>&1
fi
docker run --rm --entrypoint md5sum $PROBE $LADP
echo "SLICE_MARKER=$(docker run --rm --entrypoint grep $PROBE -c LMC_STORE_SLICE_TOKENS $LADP)"

echo "=== [2] derive p73 serve script (defer2 + slice env) ==="
sed -e 's|^  LMC_ENV="export LMC_DEFER_CHUNK_STORE=1;|  LMC_ENV="export LMC_DEFER_CHUNK_STORE=1; export LMC_STORE_SLICE_TOKENS=4096;|' \
  $R/serve_arc_b70_defer2.sh > $R/serve_arc_b70_p73.sh
grep -c "LMC_STORE_SLICE_TOKENS" $R/serve_arc_b70_p73.sh || { echo "FATAL: sed slice env"; exit 1; }

echo "=== [3] boot probe ==="
docker rm -f arc-test >/dev/null 2>&1
sleep 30   # r48h boot-drain law after a loaded engine
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_p73.sh > $CT/boot_p73.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_p73.out || { echo "FATAL: serve launch failed"; cat $CT/boot_p73.out; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep -E "LMC_DEFER|LMC_STORE_SLICE"')"
echo "SRC_OVERRIDE=$(docker exec arc-test grep -c 'src-override OK' /root/serve_full.log || echo 0)"

echo "=== [4] capture v5.1 (U32k + C32k + U64k, W2MT=4000) ==="
cd $CT && P65_FACES="U32k,C32k,U64k" P65_W2MT=4000 python3 $CT/p65_timeline.py $CT/p73_tl_v51.json 2>&1 | tee $CT/p73_sum_v51.txt
grep "prompt sizes\|TTFT" $CT/p73_sum_v51.txt

echo "=== [4b] C64k mid-drain replay (integrity face, manual timing) ==="
# fire a replay of the U64k prompt ~2 s after a FRESH U64k face's first
# token (inside the ~19-step drain). v5.1 has no trigger hook: run a
# one-off python that reuses the harness's prompt builder + SSE client,
# or add P65_REPLAY=U64k:+2 in a v5.2 if legged twice. Correct output +
# TTFT between C32k-class and U64k-class = PASS (partial recompute is
# the designed clean-miss path).
echo "(manual face - see DESIGN.md §5 row 'C64k mid-drain replay')"

echo "=== [5] verdict bars ==="
echo "-- STORE classes (63232 must be 0; ~38+ 3328; 1664 = U32k family):"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head'
echo "-- WARN/ERROR triage (fork-added must be 0):"
docker exec arc-test sh -c 'grep -cE " (ERROR|Traceback) " /root/serve_full.log || echo 0'

echo "=== [6] run7n t0 ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 2>&1 | tee $CT/run7n_t0_p73.out
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_p73.log 2>/dev/null
echo "P73_GATE13: DONE (analysis vs DESIGN.md §5 next; W14 restored by trap)"
