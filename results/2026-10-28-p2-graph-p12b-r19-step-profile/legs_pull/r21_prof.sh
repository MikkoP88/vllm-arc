#!/bin/bash
# r21 profile driver — decode window on the GRAPHED posture (r19 serve).
# Key difference vs r20 driver: window opens well after TTFT so it contains
# pure decode steps; analyzer counts graph replays vs eager kernels.
set -o pipefail
NAME=arc-test

curl -s -m 60 -o /dev/null -w "warm small: %{http_code} %{time_total}s\n" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"hello","max_tokens":2,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions

docker cp /root/build/r20_load.py "$NAME":/root/r21_load.py
docker exec -d "$NAME" bash -c '/opt/venv/bin/python /root/r21_load.py > /root/r21_load.log 2>&1'

for i in $(seq 1 40); do
  if docker exec "$NAME" sh -c 'grep -aq "POST /v1/completions" /root/serve_full.log'; then break; fi
  sleep 3
done
sleep 14   # TTFT (~2s at 1k ctx c8) + margin -> pure decode

curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- start $(date +%H:%M:%S)"
sleep 12
curl -s -m 60 -X POST http://127.0.0.1:8000/stop_profile; echo " <- stop $(date +%H:%M:%S)"
sleep 5
docker exec "$NAME" sh -c 'ls -la /root/prof/ | grep rank | tail -2'
echo R21_PROF_DONE
