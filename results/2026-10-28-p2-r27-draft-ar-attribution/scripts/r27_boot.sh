#!/bin/bash
# r27 boot — usage: r27_boot.sh <graphed|eager>
# Deploys /root/r27cap (p10 fastpath ONLY, timers-off law), clears /root/prof,
# stops radiance (port-8000 law), boots the phase posture + profiler-config,
# waits health. graphed = product posture verbatim (phase A, visibility check);
# eager = cudagraph_mode NONE (phase B, attribution). Run from RIG HOST.
set -eo pipefail
NAME=arc-test
MODE="${1:?graphed|eager}"
SERVE=/root/r27_serve.sh
[ "$MODE" = eager ] && SERVE=/root/r27_serve_eager.sh

echo "=== r27 cap dir (mode=$MODE) ==="
docker exec "$NAME" true 2>/dev/null || docker start "$NAME" >/dev/null
sleep 3
docker exec "$NAME" bash -c 'ls /root/p25cap/gdn_metadata_fastpath.py >/dev/null && echo P25CAP-OK'
docker exec "$NAME" mkdir -p /root/r27cap
docker exec "$NAME" bash -c 'cp /root/p25cap/gdn_metadata_fastpath.py /root/r27cap/ && echo CAP-COPIED'
docker cp /root/build/r27/r27_sitecustomize.py "$NAME":/root/r27cap/sitecustomize.py
docker exec "$NAME" bash -c 'chmod +x /root/r27cap/*.py; ls -la /root/r27cap/'
docker cp /root/build/r27/r27_serve.sh "$NAME":/root/r27_serve.sh
docker cp /root/build/r27/r27_serve_eager.sh "$NAME":/root/r27_serve_eager.sh
docker exec "$NAME" chmod +x /root/r27_serve.sh /root/r27_serve_eager.sh
docker exec "$NAME" bash -c "rm -rf /root/prof /root/r27cap/__pycache__; mkdir -p /root/prof; pkill -9 -f '[V]LLM::' 2>/dev/null; pkill -9 -f '[v]llm serve' 2>/dev/null; sleep 2; rm -f /root/serve_full.log" || true

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker exec -d "$NAME" bash "$SERVE"
echo "R27_BOOT_$(echo $MODE | tr a-z A-Z) $(date +%H:%M:%S) — allow 8 min"
ok=0
for i in $(seq 1 72); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "R27_BOOT_DEATH:"
    docker exec "$NAME" tail -60 /root/serve_full.log
    exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R27_HEALTHY_$(echo $MODE | tr a-z A-Z) $(date +%H:%M:%S)" || { echo "R27_HEALTH_TIMEOUT"; docker exec "$NAME" tail -30 /root/serve_full.log; exit 5; }
echo "--- posture: fastpath / interlock / spec-mode / captures ---"
docker exec "$NAME" bash -c 'grep -a "OK (pid" /root/serve_full.log | head -2; grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "warmup interlock" /root/serve_full.log | head -1; grep -aE "cuda_graph|CUDAGraph|Capturing" /root/serve_full.log | tail -2 | head -c 300; echo; grep -aiE "eager|graph" /root/serve_full.log | grep -ai "mode" | head -2'
echo "R27_BOOT_DONE"
