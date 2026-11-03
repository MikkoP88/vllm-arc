#!/bin/bash
# p616a_boot_smoke.sh — P6.16a BOOT+SMOKE (owner directive 2026-11-03:
# implement the upstream-direction fix inline; start-to-end).
# Fix A = upstream-config-native spec-decode sampling posture:
# draft_sample_method="probabilistic" on the pinned vllm 0.30.0 wheel,
# served from the CERT6 IMAGE UNCHANGED via serve-line sed only.
# Posture: k8v4 default (the p615c collapse dtype — direct A/B).
# PASS = boots + t0 solo in the Run-7n band (banked k8v4 88.3-88.6)
# + no t1.0 trace leg < 55 tok/s (banked p615c: 1 leg @26, 2 depth-dead).
set -uo pipefail
R=/root/build
CT=$R/p65/p616/p616a
IMG=vllm-arc:v0.30.0-b70-cert6
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p616.log 2>&1 &
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

# 1) serve builder — TWO gates vs the banked p615c leg (one-variable law
#    applied twice, both verified by count):
#    (a) dtype: fp8_e4m3 -> turboquant_k8v4 (same sed as every p615 leg)
#    (b) SPEC_FLAG: += "draft_sample_method":"probabilistic"
cp $R/serve_arc_b70.sh $R/serve_p616a.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p616a.sh
sed -i 's|num_speculative_tokens\\":$ARC_SPEC}|num_speculative_tokens\\":$ARC_SPEC,\\"draft_sample_method\\":\\"probabilistic\\"}|' $R/serve_p616a.sh
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p616a.sh)
f=$(grep -oE -- "--kv-cache-dtype [a-z0-9_]+" $R/serve_p616a.sh | grep -vc -- "--kv-cache-dtype turboquant_k8v4" || true)
m=$(grep -c "draft_sample_method" $R/serve_p616a.sh)
p=$(grep -c "probabilistic" $R/serve_p616a.sh)
echo "SUBSTITUTIONS: dtype_lines_for_preset=$n other_dtypes=$f spec_method_lines=$m probabilistic=$p"
[ "$n" = "1" ] && [ "$f" = "0" ] && [ "$m" = "1" ] && [ "$p" = "1" ] || { echo "FATAL: substitution wrong"; exit 1; }
grep -n 'SPEC_FLAG=' $R/serve_p616a.sh | grep -v '^.*SPEC_FLAG=""' | head -2
cp $R/serve_p616a.sh "$CT/serve_used.sh"

# 2) boot (default posture: ARC_MNBT unset -> 8192; graph law on)
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_p616a.sh > "$CT/boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed"; cat "$CT/boot.out"; exit 1; }
wait_health
docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > "$CT/posture.txt" || true
docker exec arc-test sh -c 'grep -m3 -i "draft_sample_method\|probabilistic\|SpeculativeConfig" /root/serve_full.log' > "$CT/spec_method.txt" 2>/dev/null || echo "spec-method line NOT found in serve log" > "$CT/spec_method.txt"
cat "$CT/posture.txt"
cat "$CT/spec_method.txt"

# 3) greedy flip (first-traffic law; t0 path — probabilistic drafting
#    must degenerate to argmax at temp 0, so this face should be
#    byte-identical in behavior)
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

# 4) throwaway t1.0 request (absorb first-sampled-request JIT — the
#    probabilistic gumbel path's first traffic)
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

# 5) t0 band check — the Run-7n exact battery (agg 4x256 x2 + solo x2,
#    temp 0 default face). Banked k8v4 probe band: solo 88.3-88.6 tok/s.
python3 $R/r70/bench_run7n_t5.py > "$CT/t0_run7n.out" 2>&1
cat "$CT/t0_run7n.out"

# 6) the A/B trace — 10 streaming solo t1.0-xhigh requests, byte-exact
#    banked t1.0 body, per-chunk timestamps + /metrics deltas (same
#    instrument as the banked p615c collapse trace)
python3 $R/p65/p615/collapse_trace.py 10 > "$CT/collapse_trace.out" 2>&1
cat "$CT/collapse_trace.out"

# 7) audit + save
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full.log" 2>/dev/null
E=$(grep -c "ERROR" "$CT/serve_full.log" || true)
T=$(grep -c "Traceback" "$CT/serve_full.log" || true)
echo "ERRORS=$E TRACEBACKS=$T (want 0/0)"
echo "P616A_BOOT_SMOKE: DONE (W14 restored by trap)"
