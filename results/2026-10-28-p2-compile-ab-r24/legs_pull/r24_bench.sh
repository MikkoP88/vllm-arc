#!/bin/bash
# r24 bench — compile A/B gate battery. ORDER MATTERS: survival flip payload
# FIRST (defect-b regression cell on the new posture), then spec gate
# (compile×spec acceptance law REVAL on 0.30), then stepprobe + run7n +
# matrix + onset.
set -eo pipefail
LEG=/root/build/legs/arc-p2-r24
NAME=arc-test
mkdir -p "$LEG"

echo "=== R24 0: survival (flip payload as FIRST traffic) ==="
t0=$(date +%s.%N)
code=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req.json)
t1=$(date +%s.%N)
echo "large-first: HTTP $code in $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" | tee "$LEG/survival.txt"
code2=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req_small.json 2>/dev/null) || code2=SKIP
echo "small-after: HTTP $code2" | tee -a "$LEG/survival.txt"

echo "=== R24 1: spec gate (compile x spec acceptance REVAL) ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt"

echo "=== R24 2: stepprobe ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

echo "=== R24 3: run7n x2 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_2.txt"

echo "=== R24 4: matrix + k1k ==="
docker cp /root/build/arc_pybench.py "$NAME":/root/arc_pybench.py
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r24 2>&1 | tee "$LEG/pymatrix.txt"
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r24k1 k1k 2>&1 | tee "$LEG/k1k_warm.txt"

docker exec "$NAME" bash -c 'grep -ac onset /root/serve_full.log' > "$LEG/onset_count.txt" || true
echo "onset=$(cat "$LEG/onset_count.txt")"
docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -4' | tee "$LEG/p10_stats.txt"
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_r24_final.log"
echo "R24_BENCH_DONE"
