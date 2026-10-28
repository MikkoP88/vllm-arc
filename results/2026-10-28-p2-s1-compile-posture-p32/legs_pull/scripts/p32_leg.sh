#!/bin/bash
# p32 S1 leg driver — one composite arc leg:
#   phase R (reference): product posture (mode=none, TIMERS-OFF per law) ->
#                        golden-logits reference capture
#   phase S (S1):        W14 compile posture (FULL_AND_PIECEWISE) + timer ->
#                        survival -> spec gate -> golden compare -> run7n ->
#                        stepprobe -> split pull
# Radiance-test stopped FIRST (port-8000 law); W14 restored by p32_teardown.sh.
set -eo pipefail
NAME=arc-test
LEG=/root/build/legs/arc-p2-p32
mkdir -p "$LEG"

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker start "$NAME"
sleep 5
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true

echo "=== P32 phase R: reference boot (product mode=none, timers off) ==="
docker exec "$NAME" bash -c 'ls /root/p10cap/gdn_metadata_fastpath.py /root/p10cap/sitecustomize.py >/dev/null && echo P10CAP-OK'
docker exec "$NAME" bash -c 'rm -f /root/serve_full.log; rm -rf /root/p10cap/__pycache__'
docker exec -d "$NAME" bash /root/p23_serve.sh
echo "R_BOOT $(date +%H:%M:%S)"
ok=0
for i in $(seq 1 72); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "R_BOOT_DEATH:"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R_HEALTHY $(date +%H:%M:%S)" || { echo "R_HEALTH_TIMEOUT"; exit 5; }
python3 /root/build/p25/p32_golden.py --out "$LEG/golden_ref.json" 2>&1 | tee "$LEG/golden_ref.txt"
cp "$LEG/golden_ref.json" /root/build/p25/golden_ref.json

echo "=== P32 phase S: S1 boot (FULL_AND_PIECEWISE, W14 sizes, timer on) ==="
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2; rm -f /root/serve_full.log; rm -rf /root/p25cap/__pycache__'
docker exec -d "$NAME" bash /root/p32_s1_serve.sh
echo "S_BOOT $(date +%H:%M:%S) — piecewise capture phases, allow 20 min"
ok=0
for i in $(seq 1 120); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "S_BOOT_DEATH:"; docker exec "$NAME" tail -60 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "S_HEALTHY $(date +%H:%M:%S)" || { echo "S_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: compile mode / coercion / fastpath / timer / interlock ---"
docker exec "$NAME" bash -c 'grep -ao "CompilationMode\.[A-Z_]*" /root/serve_full.log | sort | uniq -c; grep -ac "coerced" /root/serve_full.log; grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "phase timer installed" /root/serve_full.log | head -2; grep -a "warmup interlock installed" /root/serve_full.log | head -1; grep -a "Capturing" /root/serve_full.log | tail -1 | head -c 200; echo'

echo "=== P32 S1 battery ==="
echo "--- 1/6: survival (832-token flip payload as FIRST traffic) ---"
t0=$(date +%s.%N)
code=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req.json)
t1=$(date +%s.%N)
echo "large-first: HTTP $code in $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" | tee "$LEG/survival.txt"
code2=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req_small.json 2>/dev/null) || code2=SKIP
echo "small-after: HTTP $code2" | tee -a "$LEG/survival.txt"
echo "--- 2/6: spec gate (p31 acceptance, gate (a)) ---"
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt" || echo "SPEC_GATE_FAIL_EXIT=$?"
echo "--- 3/6: golden compare (gate (c)) ---"
python3 /root/build/p25/p32_golden.py --out "$LEG/golden_run.json" 2>&1 | tee "$LEG/golden_run.txt"
python3 /root/build/p25/p32_golden.py --run "$LEG/golden_run.json" --ref /root/build/p25/golden_ref.json 2>&1 | tee "$LEG/golden_compare.txt" || echo "GOLDEN_FAIL_EXIT=$?"
echo "--- 4/6: run7n x1 (TPOT diagnostic, timer-on caveat) ---"
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"
echo "--- 5/6: stepprobe (step walls c1/c8) ---"
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"
echo "--- 6/6: phase split pull (gate (d)) ---"
docker exec "$NAME" bash -c 'grep -a "arc-r25" /root/serve_full.log | tail -12' | tee "$LEG/split.txt"
docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -4' | tee "$LEG/p10_stats.txt"
docker exec "$NAME" bash -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" || true
echo "onset=$(cat "$LEG/onset_count.txt")"
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_s1_final.log"
echo "P32_S1_BENCH_DONE"
