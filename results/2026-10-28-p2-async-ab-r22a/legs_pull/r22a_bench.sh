#!/bin/bash
# r22a bench — the r18 gate faces on the new posture (prefix caching OFF):
# run7n x2 -> 18-cell matrix -> k1k re-pass -> spec gate -> onset count.
# Documents BOTH sides of the A/B: TPOT c8 gain AND the c>=2 prefix-loss.
set -o pipefail
LEG=/root/build/legs/arc-p2-r22a
NAME=arc-test
mkdir -p "$LEG"

echo "=== R22A BENCH 1/4: run7n pass 1 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"
echo "=== R22A BENCH 2/4: run7n pass 2 ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_2.txt"

echo "=== R22A BENCH 3/4: 18-cell matrix + k1k re-pass ==="
docker cp /root/build/arc_pybench.py "$NAME":/root/arc_pybench.py
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r22a 2>&1 | tee "$LEG/pymatrix.txt"
docker exec "$NAME" /opt/venv/bin/python /root/arc_pybench.py r22ak1 k1k 2>&1 | tee "$LEG/k1k_warm.txt"

echo "=== R22A BENCH 4/4: p31 spec-acceptance gate ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt"
echo "GATE_EXIT=$?"

echo "=== counts ==="
docker exec "$NAME" bash -c 'grep -ac "ZOMBIE_ONSET" /root/serve_full.log || true' | tee "$LEG/onset_count.txt"
docker exec "$NAME" cp /root/serve_full.log /root/serve_r22a_final.log
echo "R22A_BENCH_DONE"
