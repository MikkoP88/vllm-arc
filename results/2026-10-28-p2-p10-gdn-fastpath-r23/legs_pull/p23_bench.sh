#!/bin/bash
# p10/r23 bench — step decomposition (stepprobe) + the r18 gate-cell battery
# for the A/B: run7n x2, 18-cell matrix + k1k re-pass, spec gate, onset count.
# Compare vs results/2026-10-28-p2-graph-p12b-r18-p2gates (product posture,
# pre-p10) and r22a (stepprobe before/after is the p10 budget evidence).
set -eo pipefail
LEG=/root/build/legs/arc-p2-r23
NAME=arc-test
mkdir -p "$LEG"

echo "=== R23 1/4: stepprobe (step walls c1/c8) ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

echo "=== R23 2/4: run7n pass 1 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"
echo "=== R23 3/4: run7n pass 2 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_2.txt"

echo "=== R23 4/4: matrix + k1k re-pass + spec gate + onset ==="
docker cp /root/build/arc_pybench.py "$NAME":/root/arc_pybench.py
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r23 2>&1 | tee "$LEG/pymatrix.txt"
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r23k1 k1k 2>&1 | tee "$LEG/k1k_warm.txt"
docker exec "$NAME" /opt/venv/bin/python /root/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt"
docker exec "$NAME" bash -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" || true
echo "onset=$(cat "$LEG/onset_count.txt")"
echo "--- p10 build stats (budget law) ---"
docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -8' | tee "$LEG/p10_stats.txt"
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_r23_final.log"
echo "R23_BENCH_DONE"
