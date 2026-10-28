#!/bin/bash
# r29 battery — p14 TTFT ladder + P2 close-out numbers, profile LAST.
# 1/5 survival (flip payload first traffic)   — regression cell
# 2/5 spec gate                               — acceptance cell
# 3/5 TTFT ladder c1 8k/16k/32k/64k cold      — p14 bars 3944/8083/17447/41565
# 4/5 stepprobe c1/c8                         — TPOT gate numbers TIMERS-OFF
# 5/5 prefill profile window (LAST, law)      — one 8k prefill, eager-visible
set -o pipefail
NAME=arc-test
LEG=/root/build/legs/arc-p2-r29
mkdir -p "$LEG"

echo "=== r29 1/5: survival (832-token flip payload as FIRST traffic) ==="
{ code=$(curl -s -o /dev/null -w "%{http_code}" -m 120 -H 'Content-Type: application/json' \
  -d @/root/build/r18/req.json http://127.0.0.1:8000/v1/completions || echo 000); } 2>/dev/null
echo "large-first: HTTP $code"
{ code2=$(curl -s -o /dev/null -w "%{http_code}" -m 30 -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-fp8","prompt":"ping","max_tokens":4,"temperature":0}' \
  http://127.0.0.1:8000/v1/completions || echo 000); } 2>/dev/null
echo "small-after: HTTP $code2"
docker exec "$NAME" sh -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" 2>/dev/null || true

echo "=== r29 2/5: spec gate ==="
python3 /root/build/arc_spec_gate.py --json-out > "$LEG/spec_gate.json" 2> "$LEG/spec_gate.txt" || true
tail -2 "$LEG/spec_gate.txt" || true

echo "=== r29 3/5: TTFT ladder (c1, cold, max_tokens=1) ==="
cat > /root/build/r29/r29_ttft.py <<'EOF'
"""r29_ttft.py — c1 TTFT ladder, ctx 8k/16k/32k/64k, cold (unique seeded
token-id prompts per ctx — no prefix reuse), max_tokens=1 non-stream:
total request time ~= TTFT. Bars (D champion, plan App. B): 3944/8083/
17447/41565 ms."""
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
BARS = {8192: 3944, 16384: 8083, 32768: 17447, 65536: 41565}

for ctx, bar in BARS.items():
    rng = random.Random(20261028 * 100 + ctx)
    ids = [rng.randint(1000, 50000) for _ in range(ctx)]
    body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": 1,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        json.load(r)
    dt_ms = (time.perf_counter() - t0) * 1000
    print(f"TTFT ctx={ctx:6d} cold={dt_ms:8.1f} ms  bar={bar}  "
          f"delta={dt_ms/bar - 1:+.1%}", flush=True)
print("TTFT_LADDER_DONE", flush=True)
EOF
docker cp /root/build/r29/r29_ttft.py "$NAME":/root/r29_ttft.py
docker exec "$NAME" /opt/venv/bin/python /root/r29_ttft.py 2>&1 | tee "$LEG/ttft_ladder.txt"

echo "=== r29 4/5: stepprobe (step walls c1/c8, TIMERS-OFF posture) ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

echo "=== r29 5/5: prefill profile window (LAST — aggregation-hang law) ==="
curl -s -m 5 -X POST http://127.0.0.1:8000/start_profile; echo " <- prof start $(date +%H:%M:%S)"
cat > /root/build/r29/r29_prefill_req.py <<'EOF'
import json, random, time, urllib.request
rng = random.Random(99110011)
ids = [rng.randint(1000, 50000) for _ in range(8192)]
body = json.dumps({"model": "qwen3.8-27b-fp8", "prompt": ids,
                   "max_tokens": 2, "temperature": 0,
                   "ignore_eos": True}).encode()
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions", data=body,
    headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as r:
    json.load(r)
print(f"PREFILL_REQ_DONE {(time.perf_counter()-t0)*1000:.1f} ms", flush=True)
EOF
docker cp /root/build/r29/r29_prefill_req.py "$NAME":/root/r29_prefill_req.py
docker exec "$NAME" /opt/venv/bin/python /root/r29_prefill_req.py 2>&1 | tee "$LEG/prefill_req.txt"
curl -s -m 120 -X POST http://127.0.0.1:8000/stop_profile; echo " <- prof stop $(date +%H:%M:%S)"
sleep 10
docker exec "$NAME" bash -c 'ls -la /root/prof/ | tail -5'
for f in $(docker exec "$NAME" sh -c 'ls /root/prof/ 2>/dev/null'); do
  docker cp "$NAME":/root/prof/"$f" "$LEG/prof_prefill/$f" >/dev/null 2>&1 || { mkdir -p "$LEG/prof_prefill"; docker cp "$NAME":/root/prof/"$f" "$LEG/prof_prefill/$f" >/dev/null 2>&1; } && echo "pulled $f"
done
docker exec "$NAME" sh -c 'docker=true; true' 2>/dev/null
docker exec "$NAME" bash -c 'grep -a "SpecDecoding metrics" /root/serve_full.log | tail -1' || true
docker cp "$NAME":/root/serve_full.log "$LEG/serve_r29_final.log" >/dev/null 2>&1
echo "R29_BENCH_DONE"
