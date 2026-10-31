#!/bin/bash
# r47c — p48 image gates (P6.2 lmcache 0.5.6+arc1 + P6.3 hygiene).
# Legs: (1) image-native import verify (p48_verify.py), (2) certified serve
# boot from p48 (25-var chain incl. LMCACHE_TRACK_USAGE=false), (3) run7n
# batteries k3 greedy t0 + t1.0 xhigh in the same session (±5% vs r46t
# baselines: greedy 299.9 agg steady / 84.5 solo; t1.0 186.2 / 61.1),
# (4) full WARN/ERROR hygiene audit boot+battery (p63 gate: zero
# fork-added), (5) LMCache 0.5.6 keylines (glue 0, joint, retrieval INFO).
set -uo pipefail
R=/root/build
CT=$R/r47c
IMG=vllm-arc:v0.30.0-b70-p48
mkdir -p "$CT"

w14_start() {
  docker start radiance-test >/dev/null 2>&1
  sleep 3
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_r47c.log 2>&1 &
}
trap w14_start EXIT
docker stop radiance-test >/dev/null 2>&1; sleep 2

echo "=== [1] image-native verify (throwaway GPU container) ==="
docker rm -f r47c-verify >/dev/null 2>&1
docker run -d --name r47c-verify --privileged --ipc host \
  --entrypoint /bin/bash "$IMG" -c 'sleep 600' >/dev/null
docker cp "$CT/p48_verify.py" r47c-verify:/tmp/p48_verify.py
docker exec -e LMCACHE_TRACK_USAGE=false r47c-verify /opt/venv/bin/python /tmp/p48_verify.py \
  > "$CT/p48_verify.out" 2>&1
tail -4 "$CT/p48_verify.out"
docker rm -f r47c-verify >/dev/null

echo "=== [2] certified serve boot from p48 (k3, graphs, 25-var chain) ==="
docker rm -f arc-test >/dev/null 2>&1
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 bash $R/serve_arc_b70.sh \
  > "$CT/serve_boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/serve_boot.out" || { echo "FATAL: serve launch failed"; cat "$CT/serve_boot.out"; exit 1; }
ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"
docker exec arc-test cat /root/serve_full.log > "$CT/serve_boot_engine.log" 2>/dev/null

echo "=== [3] run7n batteries (same serve session) ==="
B=$R/r46t/bench_run7n_t.py
python3 "$B" --temperature 0 --thinking xhigh > "$CT/k3_t0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/k3_t0_run7n.out"
python3 "$B" --temperature 1.0 --thinking xhigh > "$CT/k3_t1.0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/k3_t1.0_run7n.out"

echo "=== [4] full WARN/ERROR audit (boot + battery) ==="
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full.log" 2>/dev/null
grep -E "WARNING|ERROR" "$CT/serve_full.log" | grep -vE "lmcache.logging|filelock|asyncio" \
  > "$CT/serve_warn_error.txt" || true
wc -l "$CT/serve_warn_error.txt"
cat "$CT/serve_warn_error.txt"

echo "=== [5] LMCache 0.5.6 keylines ==="
echo -n "glue-unavailable count: "; grep -c "glue-unavailable" "$CT/serve_full.log" || true
echo -n "Retrieved INFO lines: "; grep -cE "Retrieved [0-9]+ tokens" "$CT/serve_full.log" || true
grep -m2 "LMC joint lookup" "$CT/serve_full.log" || echo "(no joint lines yet)"
grep -m2 "hit rate" "$CT/serve_full.log" || true
grep -m3 "p41c" "$CT/serve_full.log" || true

echo "R47C_GATES_DONE"
