#!/bin/bash
# r20 profile driver — c8 1k-ctx steady DECODE window on the eager engine.
# Waits for actual request activity (not a fixed sleep) so the window is
# pure decode.
set -o pipefail
NAME=arc-test

echo "=== warm/vaccinate ==="
curl -s -m 60 -o /dev/null -w "warm small: %{http_code} %{time_total}s\n" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"hello","max_tokens":2,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions

echo "=== start c8 decode load (8 x 1k ctx, 400 out) ==="
sed 's/range(8192)/range(1024)/' /root/build/r19_load.py > /root/build/r20_load.py
docker cp /root/build/r20_load.py "$NAME":/root/r20_load.py
docker exec -d "$NAME" bash -c '/opt/venv/bin/python /root/r20_load.py > /root/r20_load.log 2>&1'

echo "=== wait for decode (request seen + 8s) ==="
for i in $(seq 1 40); do
  if docker exec "$NAME" sh -c 'grep -aq "POST /v1/completions" /root/serve_full.log'; then break; fi
  sleep 3
done
sleep 8

echo "=== profiler window (12s) ==="
curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- start $(date +%H:%M:%S)"
sleep 12
curl -s -m 60 -X POST http://127.0.0.1:8000/stop_profile; echo " <- stop $(date +%H:%M:%S)"
sleep 5
docker exec "$NAME" sh -c 'ls -la /root/prof2/ | tail -5; grep -c "200 OK" /root/r20_load.log 2>/dev/null || true'
echo R20_PROF_DONE
