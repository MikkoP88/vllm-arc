#!/bin/bash
# r18 survival probe — §10 P2 gate row 1: "FULL graph + MTP boots and
# survives first request". The FIRST traffic after health is the exact r16/r17
# large flip payload (832 tok, the one that flipped 3/3 pre-fix boots).
# Expect (patch ca75f1d8): warmup interlock fires -> 200 -> no onset.
set -eo pipefail
LEG=/root/build/legs/arc-p2-r18
NAME=arc-test
mkdir -p "$LEG"

echo "=== R18 SURVIVAL: large-as-first (the r16/r17 flip payload) ==="
R1=$(curl -s -o /tmp/r18_first.json -w "%{http_code} %{time_total}" -m 600 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req.json \
  http://127.0.0.1:8000/v1/completions)
echo "large-first: HTTP $R1"

R2=$(curl -s -o /tmp/r18_small2.json -w "%{http_code} %{time_total}" -m 120 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req_small.json \
  http://127.0.0.1:8000/v1/completions)
echo "small2: HTTP $R2"

R3=$(curl -s -o /tmp/r18_large2.json -w "%{http_code} %{time_total}" -m 600 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req.json \
  http://127.0.0.1:8000/v1/completions)
echo "large2: HTTP $R3"

echo "=== interlock + onset evidence ==="
docker exec "$NAME" bash -c \
  'grep -a "warmup interlock installed\|warmup interlock first\|warmup stage\|warmup complete\|warmup FAILED" /root/serve_full.log | tail -10'
ONSET=$(docker exec "$NAME" bash -c 'grep -ac "ZOMBIE_ONSET" /root/serve_full.log || true')
echo "onset_count=$ONSET"
docker exec "$NAME" bash -c 'grep -a "Xe2 posture active\|cudagraph\|speculative" /root/serve_full.log | head -6' || true

{
  echo "large-first: $R1"; echo "small2: $R2"; echo "large2: $R3"
  echo "onset_count=$ONSET"
} > "$LEG/survival.txt"
docker exec "$NAME" cp /root/serve_full.log /root/serve_r18_survival.log
echo "R18_SURVIVAL_DONE"
