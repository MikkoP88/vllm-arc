#!/bin/bash
# p33 probe leg (r26) boot — deploy /root/p33cap (p10 fastpath + r25 timer
# from p25cap + p33 probe/sitecustomize from host), stop radiance (port-8000
# law), boot product posture with probes, wait health. Run from RIG HOST.
set -eo pipefail
NAME=arc-test
LEG=/root/build/legs/arc-p2-p33
mkdir -p "$LEG" /root/p33cap

echo "=== p33 cap dir ==="
docker exec "$NAME" true 2>/dev/null || docker start "$NAME" >/dev/null
sleep 3
docker exec "$NAME" bash -c 'ls /root/p25cap/gdn_metadata_fastpath.py /root/p25cap/r25_phase_timer.py >/dev/null && echo P25CAP-OK'
docker exec "$NAME" mkdir -p /root/p33cap
docker exec "$NAME" bash -c 'cp /root/p25cap/gdn_metadata_fastpath.py /root/p25cap/r25_phase_timer.py /root/p33cap/ && echo CAP-COPIED'
docker cp /root/build/p33/p33_probe.py "$NAME":/root/p33cap/p33_probe.py
docker cp /root/build/p33/p33_sitecustomize.py "$NAME":/root/p33cap/sitecustomize.py
docker exec "$NAME" bash -c 'chmod +x /root/p33cap/*.py; ls -la /root/p33cap/'
docker cp /root/build/p33/p33_serve.sh "$NAME":/root/p33_serve.sh
docker exec "$NAME" chmod +x /root/p33_serve.sh

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2; rm -f /root/serve_full.log; rm -rf /root/p33cap/__pycache__' || true
docker exec -d "$NAME" bash /root/p33_serve.sh
echo "P33_BOOT $(date +%H:%M:%S) — product posture + probes, allow 8 min"
ok=0
for i in $(seq 1 72); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "P33_BOOT_DEATH:"
    docker exec "$NAME" tail -60 /root/serve_full.log
    exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "P33_HEALTHY $(date +%H:%M:%S)" || { echo "P33_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: probes / fastpath / interlock / captures ---"
docker exec "$NAME" bash -c 'grep -a "arc-p33" /root/serve_full.log | head -12; grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "phase timer installed" /root/serve_full.log | head -2; grep -a "warmup interlock installed" /root/serve_full.log | head -1; grep -a "speculator FULL-capture" /root/serve_full.log | head -2; grep -a "Capturing" /root/serve_full.log | tail -1 | head -c 200; echo'
echo "P33_BOOT_DONE"
