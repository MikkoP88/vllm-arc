#!/bin/bash
# p614_l2_slice.sh — L2 step [4]+[5] RERUN (master plan P6.14): the p73
# slice face (defer2 + SLICE=4096) on p614c2. First attempt FATAL'd
# operationally: the L2 battery's `docker rm -f arc-test` raced the
# default-face serve's slow graph teardown -- the container died
# Exited(137) but was NOT removed, the p73 serve script's same-image
# freshness guard then reused the corpse and `docker exec -d` hit a
# stopped container (daemon: "container ... is not running", the only
# line in boot_p73face.out). Fix here: remove arc-test and WAIT until
# docker inspect says it is gone (up to 180 s), then the r48h 30 s
# drain, then boot. Default-face results (t0 300.27/84.53, t1.0
# 184.37/61.83, integrity 5/5, WARN class diff 0, flip 8.3 s,
# p48 28/28 + p73 11/11 image-native) are complete and banked in
# l2_default/ -- this script only adds the slice face.
# Trap restores W14 (recreate law: w14_boot.sh fullres ONLY).
set -uo pipefail
R=/root/build
CT=$R/p65/p614/l2_default
IMG=vllm-arc:v0.30.0-b70-p614c2
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p614l2s.log 2>&1 &
}
trap w14_start EXIT

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

echo "=== [4s] robust arc-test removal (rm-race fix) ==="
docker exec arc-test bash -c 'pkill -f "vllm serve"' >/dev/null 2>&1
sleep 10
for i in $(seq 1 36); do
  docker rm -f arc-test >/dev/null 2>&1
  docker inspect arc-test >/dev/null 2>&1 || { echo "ARC_TEST_GONE after ~$((i*5))s"; break; }
  sleep 5
done
docker inspect arc-test >/dev/null 2>&1 && { echo "FATAL: arc-test will not die"; exit 1; }
sleep 30   # r48h boot-drain law

docker stop radiance-test >/dev/null 2>&1
sleep 5

echo "=== [4] slice face boot (defer2 + SLICE=4096) ==="
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_p73.sh > "$CT/boot_p73face.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_p73face.out" || { echo "FATAL: slice face launch failed"; cat "$CT/boot_p73face.out"; exit 1; }
wait_health
echo "ENV_CHECK=$(docker exec arc-test bash -c 'tr "\0" "\n" < /proc/$(pgrep -f "vllm serve" | head -1)/environ | grep -E "LMC_DEFER|LMC_STORE"')"

echo "=== [5] slice capture (U32k + U64k, W2MT=4000) ==="
cd $CT && P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $R/p65/p65_timeline.py $CT/p65_tl_p614_v51.json 2>&1 | tee $CT/p65_sum_p614_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_p614_v51.txt
echo "-- STORE classes (63232 must be 0; 3328 = slice signature; 1664 = U32k family):"
docker exec arc-test sh -c 'grep -o "Stored [0-9]* out of total" /root/serve_full.log | sort | uniq -c | sort -rn | head -6'
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full_p73face.log" 2>/dev/null
echo "ERROR_TB=$(grep -cE 'Traceback|ERROR' "$CT/serve_full_p73face.log" || echo 0)"
echo "P614_L2_SLICE: DONE (W14 restored by trap)"
