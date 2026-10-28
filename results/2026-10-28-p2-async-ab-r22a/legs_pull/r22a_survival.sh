#!/bin/bash
# r22a survival — regression cell held from r18: the exact 832-tok flip payload
# as FIRST traffic (graphs+MTP posture must survive; interlock fires first).
set -eo pipefail
LEG=/root/build/legs/arc-p2-r22a
NAME=arc-test
mkdir -p "$LEG"

echo "=== R22A SURVIVAL: large-as-first (832-tok flip payload) ==="
R1=$(curl -s -o /tmp/r22a_first.json -w "%{http_code} %{time_total}" -m 600 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req.json \
  http://127.0.0.1:8000/v1/completions)
echo "large-first: HTTP $R1"
R2=$(curl -s -o /tmp/r22a_small2.json -w "%{http_code} %{time_total}" -m 120 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req_small.json \
  http://127.0.0.1:8000/v1/completions)
echo "small2: HTTP $R2"
R3=$(curl -s -o /tmp/r22a_large2.json -w "%{http_code} %{time_total}" -m 600 \
  -H 'Content-Type: application/json' -d @/root/build/r18/req.json \
  http://127.0.0.1:8000/v1/completions)
echo "large2: HTTP $R3"

echo "=== interlock + onset evidence ==="
docker exec "$NAME" bash -c \
  'grep -a "warmup interlock installed\|warmup interlock first\|warmup stage\|warmup complete\|warmup FAILED" /root/serve_full.log | tail -10'
ONSET=$(docker exec "$NAME" bash -c 'grep -ac "ZOMBIE_ONSET" /root/serve_full.log || true')
echo "onset_count=$ONSET"

{
  echo "large-first: $R1"; echo "small2: $R2"; echo "large2: $R3"
  echo "onset_count=$ONSET"
} > "$LEG/survival.txt"
echo "R22A_SURVIVAL_DONE"
