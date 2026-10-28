#!/bin/bash
# r18 boot — recreate arc-test from p12b2 (fork ca75f1d8: first_request_warmup
# wired via platform _XPU_PATCH_MODULES) and launch the P2 posture serve
# (mode=none + FULL_DECODE_ONLY graphs + MTP k4, ARC_TRACE_FLIP=1 evidence).
set -eo pipefail
IMG=vllm-arc:v0.30.0-b70-p12b2
NAME=arc-test

# 1. drain old engine (r17b control, dead after flip) — proctitle kill
docker exec "$NAME" bash -c 'pkill -9 -f "[V]LLM::" 2>/dev/null; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 2' || true
docker exec "$NAME" bash -c 'curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 && echo PORT-STILL-UP || echo PORT-DOWN'

# 2. container freshness law: image changed -> recreate (same posture flags
#    as serve_arc_b70.sh: privileged + oneCCL universal TP2 posture)
WANT=$(docker image inspect "$IMG" --format '{{.Id}}')
HAVE=$(docker inspect "$NAME" --format '{{.Image}}' 2>/dev/null || true)
if [ "$HAVE" != "$WANT" ]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" \
    --privileged \
    --device /dev/dri --ipc host --network host \
    --shm-size 64m \
    -v /root/build/neo_cache:/root/.neo_cache \
    -v /root/build/triton_cache:/root/.triton_cache \
    -v /root/build/inductor_cache:/root/.inductor_cache \
    -v /models/qwen3.8-27b-fp8:/models/target:ro \
    -v "/root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro" \
    --entrypoint /bin/bash "$IMG" -c 'sleep infinity'
  echo "ARC_CONTAINER_FRESH image=$IMG id=${WANT:7:12}"
else
  echo "ARC_CONTAINER_CURRENT image=$IMG"
fi

# 3. stage artifacts
docker cp /root/build/r18/r18_serve.sh "$NAME":/root/r18_serve.sh
docker cp /root/build/r18/req.json "$NAME":/root/req.json
docker cp /root/build/r18/req_small.json "$NAME":/root/req_small.json
docker exec "$NAME" bash -c 'grep -c first_request_warmup /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/first_request_warmup.py 2>/dev/null || find /opt/venv -name first_request_warmup.py | head -2'

# 4. wait out old port, then launch
for i in $(seq 1 15); do
  curl -s -m 2 http://127.0.0.1:8000/health >/dev/null 2>&1 || break
  sleep 2
done
docker exec -d "$NAME" bash /root/r18_serve.sh
echo "R18_SERVE_LAUNCHED $(date +%H:%M:%S) — polling health (max 12 min)"
ok=0
for i in $(seq 1 72); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "BOOT_DEATH detected:"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R18_HEALTHY $(date +%H:%M:%S)" || { echo "R18_HEALTH_TIMEOUT"; docker exec "$NAME" tail -20 /root/serve_full.log; exit 5; }
docker exec "$NAME" bash -c 'grep -a "interlock installed\|ArcXpuPlatform" /root/serve_full.log | tail -8'
docker exec "$NAME" bash -c 'grep -ac "ZOMBIE_ONSET" /root/serve_full.log || true'
