#!/bin/bash
# r30 boot — p40 first LMCache leg. Prereq: image vllm-arc:v0.30.0-b70-p40
# already built (build_image.sh from the p40 branch archive with the
# lmcache-0.5.5+arc1 wheel in context). Recreates arc-test from that image
# (same run recipe as the p12b2 container: /dev/dri, host net, 64M shm,
# cache + model + template binds), deploys /root/r30cap, stops radiance,
# launches serve (kv-transfer + LMCACHE env), waits health.
set -eo pipefail
NAME=arc-test
IMG=vllm-arc:v0.30.0-b70-p41

docker image inspect "$IMG" >/dev/null || { echo "FATAL: image $IMG missing"; exit 1; }
docker exec "$NAME" true 2>/dev/null && docker stop "$NAME" >/dev/null 2>&1 || true
docker rm "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --privileged \
  --device /dev/dri --network host --shm-size 64M --ipc host \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  -v /root/build/inductor_cache:/root/.inductor_cache \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  --entrypoint bash "$IMG" -c "sleep infinity" >/dev/null
sleep 3
docker exec "$NAME" bash -c 'pip show lmcache 2>/dev/null | sed -n "2p"; python -c "import lmcache.xpu_ops; print(\"XPU_OPS_IMPORT_OK\")" 2>&1 | tail -1'
# lmcache-0.5.5 runtime deps missed by the p40 --no-deps set (upstream
# Requires-Dist; storage_backend fs_connector imports aiofile at boot,
# redis listed by metadata). Dockerfile dep line gains both at next build.
docker exec "$NAME" pip install --no-cache-dir -q aiofile redis && echo DEPS-OK

docker exec "$NAME" mkdir -p /root/r30cap
# canonical patches ship in the fork dist (vllm_gaudi/xpu/patches/) — the
# fresh container has no /root/p25cap (old writable layer was docker rm'd).
# Newer-than-image patches overlay from /root/build/r30/patches/ (leg-time
# hot path until the next image build bakes the dist).
docker exec "$NAME" bash -c 'cp /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/gdn_metadata_fastpath.py /opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/mamba_lmcache_glue.py /root/r30cap/ && echo CAP-COPIED-FROM-DIST'
for f in /root/build/r30/patches/*.py; do
  [ -f "$f" ] && docker cp "$f" "$NAME":/root/r30cap/ >/dev/null && echo "CAP-OVERLAY $(basename "$f")"
done
[ -d /root/build/r30/patches/lmcint ] && docker cp /root/build/r30/patches/lmcint "$NAME":/root/r30cap/ >/dev/null && echo "CAP-OVERLAY lmcint/"
docker cp /root/build/r30/r30_sitecustomize.py "$NAME":/root/r30cap/sitecustomize.py
docker cp /root/build/r30/r30_serve.sh "$NAME":/root/r30_serve.sh
docker exec "$NAME" chmod +x /root/r30_serve.sh
docker exec "$NAME" bash -c 'chmod +x /root/r30cap/*.py'
docker exec "$NAME" bash -c 'rm -rf /root/r30cap/__pycache__; pkill -9 -f "[V]LLM::" 2>/dev/null; sleep 2; rm -f /root/serve_full.log' || true

docker stop radiance-test >/dev/null 2>&1 && echo RADIANCE-STOPPED || echo RADIANCE-ALREADY-DOWN
docker exec -d "$NAME" bash /root/r30_serve.sh
echo "R30_BOOT $(date +%H:%M:%S) — product posture + LMCacheConnectorV1, allow 9 min"
ok=0
for i in $(seq 1 72); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  if docker exec "$NAME" sh -c 'grep -aqE "EngineCore (failed|died)|Traceback|Engine core initialization failed" /root/serve_full.log' 2>/dev/null; then
    echo "R30_BOOT_DEATH:"
    docker exec "$NAME" tail -80 /root/serve_full.log
    exit 4
  fi
  sleep 10
done
[ "$ok" = 1 ] && echo "R30_HEALTHY $(date +%H:%M:%S)" || { echo "R30_HEALTH_TIMEOUT"; docker exec "$NAME" tail -40 /root/serve_full.log; exit 5; }
echo "--- posture: fastpath / interlock / captures / lmcache / p41 glue ---"
docker exec "$NAME" bash -c 'grep -a "fastpath installed" /root/serve_full.log | head -2; grep -a "warmup interlock" /root/serve_full.log | head -1; grep -ai "lmcache" /root/serve_full.log | head -6; grep -a "arc-p41" /root/serve_full.log | head -4; grep -a "Capturing" /root/serve_full.log | tail -1 | head -c 150; echo'
echo "R30_BOOT_DONE"
