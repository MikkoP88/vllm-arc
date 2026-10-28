#!/bin/bash
# r19 profile driver — c8 8k-ctx steady decode under a 10s profiler window.
set -o pipefail
LEG=/root/build/legs/arc-p2-r19-prof
NAME=arc-test
mkdir -p "$LEG"

echo "=== warm/vaccinate (first generate -> interlock) ==="
curl -s -m 60 -o /dev/null -w "warm small: %{http_code} %{time_total}s\n" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"hello","max_tokens":2,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions

echo "=== start c8 load (8 x 8k ctx, 400 out) ==="
docker cp /root/build/r19_load.py "$NAME":/root/r19_load.py
docker exec -d "$NAME" bash -c '/opt/venv/bin/python /root/r19_load.py > /root/r19_load.log 2>&1'
sleep 12

echo "=== profiler window ==="
curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- start $(date +%H:%M:%S)"
sleep 10
curl -s -m 30 -X POST http://127.0.0.1:8000/stop_profile; echo " <- stop $(date +%H:%M:%S)"
sleep 8
docker exec "$NAME" bash -c 'pkill -f "r19_load" 2>/dev/null; ls -la /root/prof/ | tail -6'
docker exec "$NAME" bash -c 'tail -12 /root/r19_load.log'
echo R19_PROF_DONE
