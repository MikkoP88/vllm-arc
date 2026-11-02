#!/bin/bash
# p614_dbg_boot.sh — transient DEBUG boot (never banks, never ships):
# the p614c1 image + the instrumented adapter bind-mounted over
# site-packages. One flip request to fire the lagged store, then dump
# every [p614dbg] line. Trap restores W14 (recreate law).
set -uo pipefail
R=/root/build
CT=$R/p65/p614/dbg
ADP=/opt/venv/lib/python3.12/site-packages/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p614dbg.log 2>&1 &
}
trap w14_start EXIT

docker rm -f arc-test >/dev/null 2>&1
docker stop radiance-test >/dev/null 2>&1
sleep 30   # r48h boot-drain law

# one-variable serve script + the debug mount injected after --shm-size
sed -e 's|--kv-cache-dtype fp8_e4m3|--kv-cache-dtype turboquant_4bit_nc|' \
    -e "s|--shm-size 64g \\\\|--shm-size 64g \\\\\n    -v $R/p65/p614/dbg/vllm_v1_adapter.py:$ADP:ro \\\\|" \
    $R/serve_arc_b70.sh > $R/serve_arc_b70_tq4dbg.sh
n=$(grep -c "turboquant_4bit_nc" $R/serve_arc_b70_tq4dbg.sh)
m=$(grep -c "p614/dbg/vllm_v1_adapter.py" $R/serve_arc_b70_tq4dbg.sh)
echo "DBG_SCRIPT: tq=$n mount=$m (must be 1/1)"
[ "$n" = "1" ] && [ "$m" = "1" ] || { echo "FATAL: dbg script wrong"; exit 1; }

ARC_IMAGE=vllm-arc:v0.30.0-b70-p614c1 ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_tq4dbg.sh > "$CT/boot_dbg.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_dbg.out" || { echo "FATAL: launch failed"; cat "$CT/boot_dbg.out"; exit 1; }

ok=""
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ -n "$ok" ] || { echo "FATAL: no health"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
echo "HEALTH=OK after ~$((i*5))s"

python3 - << 'PYEOF' > "$CT/flip_dbg.out" 2>&1
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("FLIP:", r["choices"][0]["finish_reason"], f"{time.perf_counter()-t0:.1f}s")
PYEOF
cat "$CT/flip_dbg.out"

sleep 5   # let the deferred store fire
docker exec arc-test sh -c "grep -E 'p614dbg|r43kvfmt|r43xfer|ValueError: unsupported' /root/serve_full.log | head -40" > "$CT/dbg_lines.txt"
cat "$CT/dbg_lines.txt"
echo "P614_DBG: DONE (W14 restored by trap)"
