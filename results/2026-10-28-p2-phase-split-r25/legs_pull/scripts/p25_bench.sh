#!/bin/bash
# r25 bench — LEAN diagnostic leg: phase-split evidence + distortion controls.
# 1) stepprobe (c1/c8 step walls under the timer — compare r23 62.9/77.4)
# 2) run7n x1 (solo distortion control — compare r23 59.07-59.33)
# 3) spec gate (correctness control — logits must be identical to product)
# 4) phase log pull ([arc-r25] lines with timestamps; last lines per leg are
#    pure-leg: ring = 96 steps ~ 6-7 s, legs run 15-20 s)
set -eo pipefail
LEG=/root/build/legs/arc-p2-r25
NAME=arc-test
mkdir -p "$LEG"

echo "=== R25 1/4: stepprobe (step walls c1/c8 under timer) ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

echo "=== R25 2/4: run7n (solo distortion control) ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"

echo "=== R25 3/4: spec gate (correctness control) ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt"

echo "=== R25 4/4: phase log pull ==="
docker exec "$NAME" bash -c 'grep -a "arc-r25" /root/serve_full.log' | tee "$LEG/phase_log.txt"
echo "--- p10 build stats (posture control) ---"
docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -6' | tee "$LEG/p10_stats.txt"
docker exec "$NAME" bash -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" || true
echo "onset=$(cat "$LEG/onset_count.txt")"
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_r25_final.log"
echo "R25_BENCH_DONE"
