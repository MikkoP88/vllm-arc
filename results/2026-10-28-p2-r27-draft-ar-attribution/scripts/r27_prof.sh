#!/bin/bash
# r27 profile driver — usage: r27_prof.sh <phase> <window_s>
# c1 PURE-DECODE window (single 1k-ctx stream, 600 out, token-id prompt =
# golden generator law). Warm, load in background, wait for decode, window,
# stop, dump, copy traces to the leg dir. Profile leg law: LAST thing on the
# boot (aggregation-hang law). Phase A graphed 8s; phase B eager 15s
# (launch-bound wall ~0.5s/step -> ~28 steps).
set -o pipefail
NAME=arc-test
PHASE="${1:?phase}"
WIN="${2:-12}"
LEG=/root/build/legs/arc-p2-r27

echo "=== warm/vaccinate ==="
curl -s -m 60 -o /dev/null -w "warm small: %{http_code} %{time_total}s\n" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"hello","max_tokens":2,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions

if [ ! -f /root/build/r27/r27_load.py ]; then
cat > /root/build/r27/r27_load.py <<'EOF'
"""r27_load.py — ONE 1k-ctx stream, 600 out tokens, token-id prompt (seeded
random ids, seed 20261027 — golden generator law), streaming; keeps the
engine in steady c1 decode through the profiler window."""
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
rng = random.Random(20261027)
ids = [rng.randint(1000, 50000) for _ in range(1024)]
body = {"model": MODEL, "prompt": ids, "max_tokens": 600,
        "temperature": 0, "ignore_eos": True, "stream": True}
req = urllib.request.Request(
    URL, data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
ttft = None
n = 0
with urllib.request.urlopen(req, timeout=1800) as r:
    for line in r:
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        p = line[5:].strip()
        if p == b"[DONE]":
            break
        ch = json.loads(p)
        if ch.get("choices") and ch["choices"][0].get("text"):
            n += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
wall = time.perf_counter() - t0
print(f"stream 200 ttft={ttft:.2f}s chunks={n} wall={wall:.1f}s", flush=True)
print(f"LOAD_DONE tpot_ms={1000*(wall-ttft)/max(1,n-1):.1f}", flush=True)
EOF
fi
docker cp /root/build/r27/r27_load.py "$NAME":/root/r27_load.py
docker exec "$NAME" bash -c "echo > /root/serve_full.req 2>/dev/null; true"
docker exec -d "$NAME" bash -c '/opt/venv/bin/python /root/r27_load.py > /root/r27_load.log 2>&1'

echo "=== wait for decode (request seen + 8s) ==="
for i in $(seq 1 40); do
  if docker exec "$NAME" sh -c 'grep -aq "POST /v1/completions" /root/serve_full.log'; then break; fi
  sleep 3
done
sleep 8

echo "=== profiler window (${WIN}s, phase $PHASE) ==="
curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- start $(date +%H:%M:%S)"
sleep "$WIN"
curl -s -m 120 -X POST http://127.0.0.1:8000/stop_profile; echo " <- stop $(date +%H:%M:%S)"
sleep 10
docker exec "$NAME" bash -c "ls -la /root/prof/ | tail -6; cat /root/r27_load.log 2>/dev/null; grep -a 'SpecDecoding metrics' /root/serve_full.log | tail -2"
mkdir -p "$LEG/prof_$PHASE"
docker exec "$NAME" bash -c "cp /root/prof/* /tmp/ 2>/dev/null; true"
for f in $(docker exec "$NAME" sh -c 'ls /root/prof/ 2>/dev/null'); do
  docker cp "$NAME":/root/prof/"$f" "$LEG/prof_$PHASE/$f" >/dev/null 2>&1 && echo "pulled $f"
done
docker exec "$NAME" sh -c 'du -sh /root/prof 2>/dev/null; n=$(ls /root/prof | wc -l); echo prof_files=$n'
echo "R27_PROF_DONE phase=$PHASE"
