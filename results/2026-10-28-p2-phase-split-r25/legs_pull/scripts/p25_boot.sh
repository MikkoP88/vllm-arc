#!/bin/bash
# r25 boot — product posture + p10 fastpath + r25 phase timer (diagnostic).
# Radiance-test stopped FIRST (port-8000 collision law); arc-test container
# reused; /root/p25cap staged with sitecustomize + timer + the already-
# validated fastpath file from /root/p10cap.
set -eo pipefail
NAME=arc-test

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker start "$NAME"
sleep 5
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true
curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-DOWN

docker exec "$NAME" mkdir -p /root/build/legs/arc-p2-r25 /root/p25cap
docker exec "$NAME" bash -c 'cp /root/p10cap/gdn_metadata_fastpath.py /root/p25cap/ && ls -la /root/p25cap/'
docker cp /root/build/p25/p25_serve.sh "$NAME":/root/p25_serve.sh
docker cp /root/build/p25/p25_sitecustomize.py "$NAME":/root/p25cap/sitecustomize.py
docker cp /root/build/p25/r25_phase_timer.py "$NAME":/root/p25cap/r25_phase_timer.py
docker exec "$NAME" bash -c 'rm -f /root/serve_full.log'
docker exec -d "$NAME" bash /root/p25_serve.sh
echo "R25_LAUNCHED $(date +%H:%M:%S) — polling health (max 12 min)"
ok=0
for i in $(seq 1 72); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -50 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R25_HEALTHY $(date +%H:%M:%S)" || { echo "R25_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: fastpath / phase timer / interlock ---"
docker exec "$NAME" bash -c 'grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "\[r25\]" /root/serve_full.log | head -4; grep -a "phase timer installed" /root/serve_full.log | head -2; grep -a "warmup interlock installed" /root/serve_full.log | head -1'
echo "R25_BOOT_DONE"
