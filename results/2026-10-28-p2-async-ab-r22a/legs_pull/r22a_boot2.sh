#!/bin/bash
# r22a boot2 — drain the profiler-stuck engine, relaunch WITHOUT profiler
# (bench posture). Container/image unchanged (p12b2); serve re-exec only.
set -eo pipefail
NAME=arc-test

docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 3' || true
for i in $(seq 1 20); do
  curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 || { echo PORT-DOWN; break; }
  sleep 2
done
docker exec "$NAME" bash -c 'curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-CONFIRMED-DOWN'

docker cp /root/build/r22/r22a_serve_noprof.sh "$NAME":/root/r22a_serve_noprof.sh
docker exec -d "$NAME" bash /root/r22a_serve_noprof.sh
echo "R22A_BOOT2_LAUNCHED $(date +%H:%M:%S) — polling health (max 12 min)"
ok=0
for i in $(seq 1 72); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R22A_BOOT2_HEALTHY $(date +%H:%M:%S)" || { echo "R22A_BOOT2_HEALTH_TIMEOUT"; docker exec "$NAME" tail -20 /root/serve_full.log; exit 5; }
echo "--- posture: align lines (expect 0) / async warnings (expect none) ---"
docker exec "$NAME" bash -c 'grep -ac align /root/serve_full.log; grep -a -i "async scheduling" /root/serve_full.log || echo NO-ASYNC-DISABLE-WARNING'
echo "R22A_BOOT2_DONE"
