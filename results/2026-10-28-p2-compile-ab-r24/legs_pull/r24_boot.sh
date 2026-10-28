#!/bin/bash
# r24 boot — compile A/B: FULL_AND_PIECEWISE (W14-style: compiled+captured
# prefill pieces, decode FULL graphs, speculator coerced FULL by the
# defect-a patch, warmup interlock live). p10 fastpath stays deployed.
# Longer boot expected (piecewise capture phases).
set -eo pipefail
NAME=arc-test

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker start "$NAME"
sleep 5
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true
curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-DOWN

docker exec "$NAME" mkdir -p /root/build/legs/arc-p2-r24
docker cp /root/build/r24/r24_serve.sh "$NAME":/root/r24_serve.sh
docker exec "$NAME" bash -c 'rm -f /root/serve_full.log'
docker exec -d "$NAME" bash /root/r24_serve.sh
echo "R24_LAUNCHED $(date +%H:%M:%S) — polling health (max 20 min; piecewise capture is slow)"
ok=0
for i in $(seq 1 120); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -50 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R24_HEALTHY $(date +%H:%M:%S)" || { echo "R24_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: compile mode / piecewise / coercion / fastpath / interlock ---"
docker exec "$NAME" bash -c 'grep -ao "CompilationMode\.[A-Z_]*" /root/serve_full.log | sort | uniq -c | head; grep -ac "coerced" /root/serve_full.log; grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "warmup interlock installed" /root/serve_full.log | head -1; grep -a "Capturing" /root/serve_full.log | tail -1 | head -c 200; echo'
echo "R24_BOOT_DONE"
