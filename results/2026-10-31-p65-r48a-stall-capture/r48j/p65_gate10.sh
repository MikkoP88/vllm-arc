#!/bin/bash
# p65_gate10.sh — P6.5 r48j: INSTRUMENTED deferral probe (probe2 image).
#
# r48i and r48iv2 both verified patch-on-disk + env-in-process yet the
# deferral never fired (1664-class attn stores 118/118/122). The hunk's
# condition terms all LOOK right per the code (tracker.prompt_len = full
# prompt, token_ids cumulative), so this leg instruments
# from_request_tracker directly: every PREFILL row logs its decision
# terms at WARNING level. Settles in one boot:
#   (a) rows absent from the log while stores continue -> chunk rows are
#       built by a different module (find it from the store-side);
#   (b) rows present, defer=False -> the logged terms show which one;
#   (c) rows present, defer=True while stores continue -> worker side
#       ignores can_save (patch the loops instead).
# Short leg: U32k face only, no run7n (posture proven in r48iv2/r48g).
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
PROBE=vllm-arc:v0.30.0-b70-cert3-p65probe2
ADP=/opt/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48j.log 2>&1 &
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

echo "=== [1] probe2 image (instrumented adapter baked in) ==="
if ! docker image inspect $PROBE >/dev/null 2>&1; then
  docker rm -f p65probe2 >/dev/null 2>&1
  docker run -d --name p65probe2 --entrypoint sleep $IMG 60 >/dev/null
  docker cp $CT/adapter_p65v2.py p65probe2:$ADP
  docker exec p65probe2 python3 -m py_compile $ADP || { echo "FATAL: py_compile"; exit 1; }
  docker commit -m "p65 probe2: instrumented chunk-store deferral" p65probe2 $PROBE >/dev/null
  docker rm -f p65probe2 >/dev/null 2>&1
fi
docker run --rm --entrypoint md5sum $PROBE $ADP
echo "DEFER_MARKER=$(docker run --rm --entrypoint grep $PROBE -c 'defer_chunk_store' $ADP)"
echo "ROWLOG_MARKER=$(docker run --rm --entrypoint grep $PROBE -c 'ARC_P65_ROW' $ADP)"

echo "=== [2] boot probe2 (thr2048 + DEFER env) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$PROBE ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_defer.sh > $CT/boot_defer_r48j.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot_defer_r48j.out || { echo "FATAL: serve launch failed"; cat $CT/boot_defer_r48j.out; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep LMC_DEFER')"

echo "=== [3] capture v5.1 (U32k FRESH only, W2MT default) ==="
cd $CT && P65_FACES="U32k" python3 $CT/p65_timeline.py $CT/p65_tl_defer3_v51.json 2>&1 | tee $CT/p65_sum_defer3_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_defer3_v51.txt

echo "=== [4] the instrumented verdict ==="
echo "ROWLOG_LINES=$(docker exec arc-test grep -c 'ARC_P65_ROW' /root/serve_full.log || echo 0)"
docker exec arc-test grep 'ARC_P65_ROW' /root/serve_full.log | head -40
echo "..."
docker exec arc-test grep 'ARC_P65_ROW' /root/serve_full.log | awk '{print $6}' | sort | uniq -c | sort -rn | head
echo "STORE classes:"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head'
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48j.log 2>/dev/null
echo "P65_LEG_R48J: DONE"
