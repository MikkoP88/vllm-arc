#!/bin/bash
# p65_gate.sh — P6.5 r48a capture leg on cert3 (plan §12 item 7 P6.5 (a))
R=/root/build
CT=/root/build/p65
IMG=vllm-arc:v0.30.0-b70-cert3
mkdir -p $CT

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  docker start radiance-test >/dev/null 2>&1
  nohup bash /root/build/w14_serve_only.sh > /root/build/w14_restore_r48a.log 2>&1 &
}
trap w14_start EXIT

docker stop radiance-test >/dev/null 2>&1; sleep 2

echo "=== [1] boot cert3 (k3 + graphs + 25-var chain) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash $R/serve_arc_b70.sh \
  > $CT/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $CT/boot.out || { echo "FATAL: serve launch failed"; cat $CT/boot.out; exit 1; }
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"

echo "=== [2] stall-timeline capture (host-side python3) ==="
cd $CT && python3 $CT/p65_timeline.py $CT/p65_timeline.json 2>&1 | tee $CT/p65_summary.txt

echo "=== [3] engine log + keylines ==="
docker exec arc-test cat /root/serve_full.log > $CT/serve_full_r48a.log 2>/dev/null
grep -E "Avg prompt throughput|Avg generation throughput|Running:|Waiting:" $CT/serve_full_r48a.log | tail -40 > $CT/engine_stats.txt || true
grep -cE "ERROR" $CT/serve_full_r48a.log || true
echo "P65_LEG: DONE"
