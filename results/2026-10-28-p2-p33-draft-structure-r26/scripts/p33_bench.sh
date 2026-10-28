#!/bin/bash
# p33 probe leg (r26) battery — order law: gates first (correctness), then
# diagnostics; flip payload as FIRST traffic. Golden ref = the banked
# product-posture reference (/root/build/p25/golden_ref.json, captured
# timers-off in the p32 phase-R leg).
set -eo pipefail
NAME=arc-test
LEG=/root/build/legs/arc-p2-p33

echo "=== p33 1/6: survival (832-token flip payload as FIRST traffic) ==="
t0=$(date +%s.%N)
code=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req.json)
t1=$(date +%s.%N)
echo "large-first: HTTP $code in $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" | tee "$LEG/survival.txt"
code2=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req_small.json 2>/dev/null) || code2=SKIP
echo "small-after: HTTP $code2" | tee -a "$LEG/survival.txt"

echo "=== p33 2/6: spec gate (acceptance, gate (a)) ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt" || echo "SPEC_GATE_FAIL_EXIT=$?"

echo "=== p33 3/6: golden compare (gate (c), vs banked product ref) ==="
python3 /root/build/p25/p32_golden.py --out "$LEG/golden_run.json" 2>&1 | tee "$LEG/golden_run.txt"
python3 /root/build/p25/p32_golden.py --run "$LEG/golden_run.json" --ref /root/build/p25/golden_ref.json 2>&1 | tee "$LEG/golden_compare.txt" || echo "GOLDEN_FAIL_EXIT=$?"

echo "=== p33 4/6: run7n x1 (TPOT diagnostic, timer-on caveat) ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"

echo "=== p33 5/6: stepprobe (step walls c1/c8) ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

echo "=== p33 6/6: structure + split pull ==="
docker exec "$NAME" bash -c 'grep -a "arc-p33" /root/serve_full.log' | tee "$LEG/p33_structure.txt"
docker exec "$NAME" bash -c 'grep -a "arc-r25" /root/serve_full.log | tail -10' | tee "$LEG/split.txt"
docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -4' | tee "$LEG/p10_stats.txt"
docker exec "$NAME" bash -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" || true
echo "onset=$(cat "$LEG/onset_count.txt")"
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_r26_final.log" 2>/dev/null || docker cp "$NAME":/root/serve_full.log "$LEG/serve_r26_final.log"
echo "P33_BENCH_DONE"
