#!/bin/bash
# p615c_collapse_probe.sh — WHEN does the t1.0-xhigh solo collapse mode
# hit, and what does spec acceptance do in it (owner ask 2026-11-03)?
# One k8v4 default-posture leg on cert6; greedy flip; one throwaway
# t1.0 request to absorb first-sampled-request JIT; then
# collapse_trace.py = 10 streaming solo t1.0-xhigh requests with
# per-chunk (per-engine-step) timestamps + /metrics draft/accept
# counter deltas around every request.
set -uo pipefail
R=/root/build
CT=$R/p65/p615/p615c_collapse
IMG=vllm-arc:v0.30.0-b70-cert6
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p615.log 2>&1 &
}
trap w14_start EXIT

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

# 0) pre-existing arc-test law + corpse-port guard
docker inspect arc-test > "$CT/preexisting_arc-test.json" 2>/dev/null || rm -f "$CT/preexisting_arc-test.json"
docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 10
for _i in $(seq 1 36); do
  docker rm -f arc-test >/dev/null 2>&1
  docker inspect arc-test >/dev/null 2>&1 || break
  sleep 5
done
docker stop radiance-test >/dev/null 2>&1
for _j in $(seq 1 36); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || break
  sleep 5
done
sleep 30

# 1) serve builder: k8v4 one-variable (same gate as every p615 leg)
cp $R/serve_arc_b70.sh $R/serve_p615c.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p615c.sh
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p615c.sh)
f=$(grep -oE -- "--kv-cache-dtype [a-z0-9_]+" $R/serve_p615c.sh | grep -vc -- "--kv-cache-dtype turboquant_k8v4" || true)
echo "SUBSTITUTIONS: dtype_lines_for_preset=$n other_dtypes=$f"
[ "$n" = "1" ] && [ "$f" = "0" ] || { echo "FATAL: dtype substitution wrong"; exit 1; }
cp $R/serve_p615c.sh "$CT/serve_used.sh"

# 2) boot (default posture: ARC_MNBT unset -> 8192)
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_p615c.sh > "$CT/boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
wait_health
docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > "$CT/posture.txt" || true
cat "$CT/posture.txt"

# 3) greedy flip/warmup (first-traffic law)
python3 - > "$CT/flip_first.out" 2>&1 << 'PYEOF'
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("FLIP_FIRST:", r["choices"][0]["finish_reason"], f"{time.perf_counter()-t0:.1f}s")
PYEOF
cat "$CT/flip_first.out"

# 4) throwaway t1.0 request (absorb first-sampled-request JIT so the
#    instrumented runs see steady state, not compile spikes)
python3 - > "$CT/t10_warmup.out" 2>&1 << 'PYEOF'
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 1.0,
        "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
        "repetition_penalty": 1.0, "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
        "messages": [{"role": "user", "content": "Count from 1 to 1000."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("T10_WARMUP:", r["choices"][0]["finish_reason"],
      r.get("usage", {}).get("completion_tokens"),
      f"{time.perf_counter()-t0:.1f}s")
PYEOF
cat "$CT/t10_warmup.out"

# 5) the instrumented trace
python3 $R/p65/p615/collapse_trace.py 10 > "$CT/collapse_trace.out" 2>&1
cat "$CT/collapse_trace.out"

# 6) audit + save
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full.log" 2>/dev/null
E=$(grep -c "ERROR" "$CT/serve_full.log" || true)
T=$(grep -c "Traceback" "$CT/serve_full.log" || true)
echo "ERRORS=$E TRACEBACKS=$T (want 0/0)"
echo "P615C_COLLAPSE_PROBE: DONE (W14 restored by trap)"
