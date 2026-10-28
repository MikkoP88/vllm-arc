#!/bin/bash
# r22a profile driver — decode window on the r22a posture (prefix caching OFF).
# Same pattern as r21: 8 streams, 1k ctx, window opens after TTFT -> pure decode.
set -o pipefail
NAME=arc-test

curl -s -m 60 -o /dev/null -w "warm small: %{http_code} %{time_total}s\n" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"hello","max_tokens":2,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions

docker cp /root/build/r20_load.py "$NAME":/root/r22a_load.py
docker exec -d "$NAME" bash -c '/opt/venv/bin/python /root/r22a_load.py > /root/r22a_load.log 2>&1'

for i in $(seq 1 40); do
  if docker exec "$NAME" sh -c 'grep -aq "POST /v1/completions" /root/serve_full.log'; then break; fi
  sleep 3
done
sleep 14

curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- start $(date +%H:%M:%S)"
sleep 12
curl -s -m 60 -X POST http://127.0.0.1:8000/stop_profile; echo " <- stop $(date +%H:%M:%S)"
sleep 5
docker exec "$NAME" sh -c 'ls -la /root/prof3/ | grep rank | tail -2'
echo R22A_PROF_DONE
