#!/bin/bash
# r18 bench phase — §10 P2 rows: TPOT c8 <= 60 ms (18-cell matrix), p31
# acceptance gate (pos1>=0.5 hard, 0.85 target), run7n agg/solo bars.
# Suite order law (Appendix B): run7n x2 -> 18-cell matrix -> k1k warm
# re-pass -> spec gate. prewarm_v7 N/A at P2 (cache layer is p40+, P3).
set -o pipefail
LEG=/root/build/legs/arc-p2-r18
NAME=arc-test

echo "=== R18 BENCH 1/4: run7n pass 1 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"
echo "=== R18 BENCH 2/4: run7n pass 2 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_2.txt"

echo "=== R18 BENCH 3/4: 18-cell matrix + k1k warm re-pass ==="
docker cp /root/build/arc_pybench.py "$NAME":/root/arc_pybench.py
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r18 2>&1 | tee "$LEG/pymatrix.txt"
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r18k1 k1k 2>&1 | tee "$LEG/k1k_warm.txt"

echo "=== R18 BENCH 4/4: p31 spec-acceptance gate ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt"
echo "GATE_EXIT=$?"

echo "=== counts ==="
docker exec "$NAME" bash -c 'grep -ac "ZOMBIE_ONSET" /root/serve_full.log || true' | tee "$LEG/onset_count.txt"
echo "R18_BENCH_DONE"
