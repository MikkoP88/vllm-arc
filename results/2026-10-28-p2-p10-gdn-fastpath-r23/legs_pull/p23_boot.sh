#!/bin/bash
# p10/r23 boot — law: stop radiance-test BEFORE the arc leg (port-8000
# collision lesson, r22a). Start arc-test, run the mechanism probe on the
# idle GPU, stage /root/p10cap (patch + sitecustomize), launch serve,
# poll health, grep posture (fastpath install + warmup interlock + align).
set -eo pipefail
NAME=arc-test

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker start "$NAME"
sleep 5
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true
curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-DOWN

LEG=/root/build/legs/arc-p2-r23
docker exec "$NAME" mkdir -p /root/p10cap "$LEG"
docker cp /root/build/p23/gdn_metadata_fastpath.py "$NAME":/root/p10cap/gdn_metadata_fastpath.py
docker cp /root/build/p23/p10_sitecustomize.py "$NAME":/root/p10cap/sitecustomize.py
docker cp /root/build/p23/p10_probe.py "$NAME":/root/p10_probe.py
docker cp /root/build/p23/p23_serve.sh "$NAME":/root/p23_serve.sh
echo "--- mechanism probe (idle GPU, engine not yet launched) ---"
docker exec "$NAME" /opt/venv/bin/python /root/p10_probe.py 2>&1 | tee /root/build/p23/probe.txt | tail -12

docker exec -d "$NAME" bash /root/p23_serve.sh
echo "R23_LAUNCHED $(date +%H:%M:%S) — polling health (max 12 min)"
ok=0
for i in $(seq 1 72); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R23_HEALTHY $(date +%H:%M:%S)" || { echo "R23_HEALTH_TIMEOUT"; docker exec "$NAME" tail -20 /root/serve_full.log; exit 5; }
echo "--- posture: fastpath install / warmup interlock / align mode / p10 stats ---"
docker exec "$NAME" bash -c 'grep -a "p10" /root/serve_full.log | head -8; grep -ac "align" /root/serve_full.log || true; grep -a "warmup" /root/serve_full.log | head -3'
echo "R23_BOOT_DONE"
