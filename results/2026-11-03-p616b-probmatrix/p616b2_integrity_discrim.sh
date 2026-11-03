#!/bin/bash
# p616b2_integrity_discrim.sh — P6.16b2: classify the tq4/k3v4
# integrity 4/5 flakes (seeds 404/505 = knife-edge prompts whose outputs
# differ across KV dtypes; probabilistic rehits landed on the other
# dtype's hash). Code-level proof says draft_sample_method CANNOT change
# temp-0 tokens (gumbel.py:117 plain argmax at temp 0;
# rejection_sampler_utils.py:571-585 accept-iff-argmax + argmax emit),
# so the flake is either (A) traffic-order/cache-churn environmental or
# (B) posture-correlated via allocator/layout side effects.
# Discriminating experiment, two boots:
#   BOOT 1 tq4 GREEDY (certified posture, NO spec-method sed) with the
#     EXACT p616b leg order (flip -> warmup -> run7n t0 -> run7n t1.0 ->
#     collapse_trace x10 -> spec gate) then integrity x2.
#     A flake here => (A) order/environment, probabilistic exonerated.
#   BOOT 2 tq4 PROBABILISTIC fresh boot, minimal traffic (flip -> warmup)
#     then integrity x2.
#     Clean here + flake in boot 1 => churn. Clean both => posture flake
#     is intermittent; flake only boot 2 => posture-correlated.
set -uo pipefail
R=/root/build
CT=$R/p65/p616/p616b2
IMG=vllm-arc:v0.30.0-b70-cert6
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p616b2.log 2>&1 &
}
trap w14_start EXIT

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

kill_arc() {
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
}

flip_warmup() {
  python3 - << 'PYEOF'
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("FLIP_FIRST:", r["choices"][0]["finish_reason"], f"{time.perf_counter()-t0:.1f}s")
body["max_tokens"] = 50; body["temperature"] = 1.0
body["top_p"] = 0.95; body["top_k"] = 20; body["min_p"] = 0.0
body["presence_penalty"] = 0.0; body["repetition_penalty"] = 1.0
body["chat_template_kwargs"] = {"enable_thinking": True, "reasoning_effort": "xhigh"}
body["messages"] = [{"role": "user", "content": "Count from 1 to 1000."}]
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
print("T10_WARMUP:", r["choices"][0]["finish_reason"],
      r.get("usage", {}).get("completion_tokens"), f"{time.perf_counter()-t0:.1f}s")
PYEOF
}

run_matrix_traffic() {
  python3 $R/r70/bench_run7n_t5.py --temperature 0 > "$CT/${1}_t0_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/${1}_t0_run7n.out"
  python3 $R/r70/bench_run7n_t5.py --temperature 1.0 --thinking xhigh > "$CT/${1}_t10_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/${1}_t10_run7n.out"
  python3 $R/p65/p615/collapse_trace.py 10 > "$CT/${1}_collapse_trace.out" 2>&1
  grep -c "^req" "$CT/${1}_collapse_trace.out" || true
  python3 $R/arc_spec_gate.py > "$CT/${1}_spec_gate.out" 2>&1
  tail -1 "$CT/${1}_spec_gate.out"
}

run_integrity() {
  python3 $R/r42/r42_integrity.py > "$CT/${1}_integrity_p${2}.out" 2>&1
  grep -E "seed|INTEGRITY|DETERMINISM" "$CT/${1}_integrity_p${2}.out"
}

boot_leg() {  # $1=tag $2=spec_extra_sed ("prob" or "none")
  kill_arc
  cp $R/serve_arc_b70.sh $R/serve_p616b2.sh
  sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_4bit_nc/" $R/serve_p616b2.sh
  if [ "$2" = "prob" ]; then
    sed -i 's|num_speculative_tokens\\":$ARC_SPEC}|num_speculative_tokens\\":$ARC_SPEC,\\"draft_sample_method\\":\\"probabilistic\\"}|' $R/serve_p616b2.sh
  fi
  n=$(grep -c -- "--kv-cache-dtype turboquant_4bit_nc" $R/serve_p616b2.sh)
  m=$(grep -c "draft_sample_method" $R/serve_p616b2.sh)
  p=$(grep -c "probabilistic" $R/serve_p616b2.sh)
  echo "SUBSTITUTIONS($1): dtype=$n spec_method=$m prob=$p (want 1 / $([ "$2" = "prob" ] && echo 1 || echo 0) / $([ "$2" = "prob" ] && echo 1 || echo 0))"
  [ "$n" = "1" ] || { echo "FATAL dtype"; exit 1; }
  cp $R/serve_p616b2.sh "$CT/serve_used_$1.sh"
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
    bash $R/serve_p616b2.sh > "$CT/${1}_boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/${1}_boot.out" || { echo "FATAL: launch failed ($1)"; cat "$CT/${1}_boot.out"; exit 1; }
  wait_health
}

# ============ BOOT 1: tq4 GREEDY, exact p616b traffic order ============
echo "=========== BOOT 1: tq4 GREEDY + matrix traffic order ==========="
boot_leg greedy none
flip_warmup 2>&1 | tee "$CT/greedy_flip_warmup.out"
run_matrix_traffic greedy
run_integrity greedy 1
run_integrity greedy 2
docker exec arc-test cat /root/serve_full.log > "$CT/greedy_serve_full.log" 2>/dev/null
E1=$(grep -c "ERROR" "$CT/greedy_serve_full.log" || true)
echo "BOOT1 greedy errors=$E1"

# ============ BOOT 2: tq4 PROBABILISTIC, minimal traffic ============
echo "=========== BOOT 2: tq4 PROBABILISTIC + minimal traffic ==========="
boot_leg prob prob
flip_warmup 2>&1 | tee "$CT/prob_flip_warmup.out"
run_integrity prob 1
run_integrity prob 2

docker exec arc-test cat /root/serve_full.log > "$CT/prob_serve_full.log" 2>/dev/null
E=$(grep -c "ERROR" "$CT/prob_serve_full.log" || true)
echo "P616B2: DONE (ERRORS in prob log=$E; W14 restored by trap)"
