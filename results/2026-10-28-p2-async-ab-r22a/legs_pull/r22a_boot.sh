#!/bin/bash
# r22a boot — arc-test (exited with r21b state) restarted on the SAME p12b2
# image, staged with the r22a serve script. Single-variable A/B vs r21b:
# prefix caching OFF (-> mamba_cache_mode 'none' -> async skip path active).
set -eo pipefail
IMG=vllm-arc:v0.30.0-b70-p12b2
NAME=arc-test

docker start "$NAME"
sleep 2
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true
docker exec "$NAME" bash -c 'curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-DOWN'

WANT=$(docker image inspect "$IMG" --format '{{.Id}}')
HAVE=$(docker inspect "$NAME" --format '{{.Image}}' 2>/dev/null || true)
if [ "$HAVE" != "$WANT" ]; then
  echo "IMAGE-MISMATCH want=${WANT:7:12} have=${HAVE:7:12} — recreate needed"
  exit 3
fi
echo "ARC_CONTAINER_CURRENT image=$IMG id=${WANT:7:12}"

docker cp /root/build/r22/r22a_serve.sh "$NAME":/root/r22a_serve.sh
docker cp /root/build/r18/req.json "$NAME":/root/req.json
docker cp /root/build/r18/req_small.json "$NAME":/root/req_small.json

for i in $(seq 1 15); do
  curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 || break
  sleep 2
done
docker exec -d "$NAME" bash /root/r22a_serve.sh
echo "R22A_SERVE_LAUNCHED $(date +%H:%M:%S) — polling health (max 12 min)"
ok=0
for i in $(seq 1 72); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R22A_HEALTHY $(date +%H:%M:%S)" || { echo "R22A_HEALTH_TIMEOUT"; docker exec "$NAME" tail -20 /root/serve_full.log; exit 5; }

echo "=== posture verification (the r22a gates) ==="
echo "--- mamba cache mode lines (EXPECT: 'none when prefix caching is disabled'; NO align line) ---"
docker exec "$NAME" bash -c 'grep -a -i "mamba cache mode" /root/serve_full.log || echo NO-MAMBA-LINE'
echo "--- async scheduling disable warnings (EXPECT none) ---"
docker exec "$NAME" bash -c 'grep -a -i "async scheduling" /root/serve_full.log || echo NO-ASYNC-DISABLE-WARNING'
echo "--- prefix caching state (EXPECT disabled mention or absence of enable) ---"
docker exec "$NAME" bash -c 'grep -a -i "prefix cach" /root/serve_full.log | head -3 || true'
echo "--- interlock + platform lines ---"
docker exec "$NAME" bash -c 'grep -a "interlock\|graph-safe output patch installed" /root/serve_full.log | tail -6'
echo "R22A_BOOT_DONE"
