#!/bin/bash
# r29 boot — p14 prefill-parity + P2 close-out numbers leg. Product posture
# VERBATIM + profiler armed + p10 fastpath ONLY (timers OFF — TTFT/TPOT
# gates-leg law). Deploys /root/r29cap, clears /root/prof, stops radiance,
# boots, waits health. Run from RIG HOST.
set -eo pipefail
NAME=arc-test

echo "=== r29 cap dir ==="
docker exec "$NAME" true 2>/dev/null || docker start "$NAME" >/dev/null
sleep 3
docker exec "$NAME" bash -c 'ls /root/p25cap/gdn_metadata_fastpath.py >/dev/null && echo P25CAP-OK'
docker exec "$NAME" mkdir -p /root/r29cap
docker exec "$NAME" bash -c 'cp /root/p25cap/gdn_metadata_fastpath.py /root/r29cap/ && echo CAP-COPIED'
docker cp /root/build/r29/r29_sitecustomize.py "$NAME":/root/r29cap/sitecustomize.py
docker exec "$NAME" bash -c 'chmod +x /root/r29cap/*.py; ls -la /root/r29cap/'
docker cp /root/build/r29/r29_serve.sh "$NAME":/root/r29_serve.sh
docker exec "$NAME" chmod +x /root/r29_serve.sh
docker exec "$NAME" bash -c 'rm -rf /root/prof /root/r29cap/__pycache__; mkdir -p /root/prof; pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2; rm -f /root/serve_full.log' || true

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker exec -d "$NAME" bash /root/r29_serve.sh
echo "R29_BOOT $(date +%H:%M:%S) — product posture + profiler, allow 8 min"
ok=0
for i in $(seq 1 72); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "R29_BOOT_DEATH:"
    docker exec "$NAME" tail -60 /root/serve_full.log
    exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R29_HEALTHY $(date +%H:%M:%S)" || { echo "R29_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: fastpath / interlock / captures ---"
docker exec "$NAME" bash -c 'grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "warmup interlock" /root/serve_full.log | head -1; grep -a "Capturing" /root/serve_full.log | tail -1 | head -c 200; echo'
echo "R29_BOOT_DONE"
