#!/bin/bash
# p616b_matrix.sh — P6.16b MATRIX (owner directive 2026-11-03: all
# performance numbers retested; merge only on real impact; certified
# configs must not regress). Full 5-dtype matrix on the PROBABILISTIC
# draft posture (draft_sample_method=probabilistic, serve-line sed only)
# from the cert6 image UNCHANGED, one boot per dtype, one-variable law
# per boot vs the banked greedy faces (same image, same harnesses).
# Amended merge gates (master-plan P6.16a RESULT block, pre-registered
# BEFORE this leg): (i) zero collapse-mode t1.0 legs per TQ dtype
# (pos0 >= 80% every leg, worst/median wall <= 1.7x); (ii) t0 agg/solo
# within -1.5% of banked; (iii) TTFT within -3%; (iv) det 3/3 +
# integrity 5/5 + 0 new WARN class; (v) EV >= -10% per dtype.
# Battery per dtype: flip-first, run7n t0 (agg x2 + solo x2), run7n
# t1.0 xhigh (agg x2 + solo x2), collapse_trace x10 (TQ dtypes only),
# spec gate, integrity 5/5 + det 3/3, solo TTFT U32k/U64k, log audit.
set -uo pipefail
R=/root/build
CTROOT=$R/p65/p616/p616b
IMG=vllm-arc:v0.30.0-b70-cert6
BENCH=$R/r70/bench_run7n_t5.py
mkdir -p "$CTROOT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p616b.log 2>&1 &
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

# dtype list: tag|preset
LEG_LIST="fp8|fp8_e4m3 k8v4|turboquant_k8v4 tq4|turboquant_4bit_nc k3v4|turboquant_k3v4_nc 3bit|turboquant_3bit_nc"

for entry in $LEG_LIST; do
  TAG="${entry%%|*}"; PRESET="${entry##*|}"
  CT=$CTROOT/$TAG
  mkdir -p "$CT"
  echo "=================== P616B LEG $TAG ($PRESET) ==================="

  # 0) pre-existing arc-test law + corpse-port guard
  docker inspect arc-test > "$CT/preexisting_arc-test.json" 2>/dev/null || rm -f "$CT/preexisting_arc-test.json"
  kill_arc

  # 1) serve builder: spec-method sed ALWAYS; dtype sed unless fp8 default
  cp $R/serve_arc_b70.sh $R/serve_p616b.sh
  if [ "$PRESET" != "fp8_e4m3" ]; then
    sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype $PRESET/" $R/serve_p616b.sh
  fi
  sed -i 's|num_speculative_tokens\\":$ARC_SPEC}|num_speculative_tokens\\":$ARC_SPEC,\\"draft_sample_method\\":\\"probabilistic\\"}|' $R/serve_p616b.sh
  n=$(grep -c -- "--kv-cache-dtype $PRESET" $R/serve_p616b.sh)
  f=$(grep -oE -- "--kv-cache-dtype [a-z0-9_]+" $R/serve_p616b.sh | grep -vc -- "--kv-cache-dtype $PRESET" || true)
  m=$(grep -c "draft_sample_method" $R/serve_p616b.sh)
  p=$(grep -c "probabilistic" $R/serve_p616b.sh)
  echo "SUBSTITUTIONS: dtype=$n other=$f spec_method=$m probabilistic=$p"
  [ "$n" = "1" ] && [ "$f" = "0" ] && [ "$m" = "1" ] && [ "$p" = "1" ] || { echo "FATAL: substitution wrong for $TAG"; exit 1; }
  cp $R/serve_p616b.sh "$CT/serve_used.sh"

  # 2) boot
  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
    bash $R/serve_p616b.sh > "$CT/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/boot.out" || { echo "FATAL: launch failed ($TAG)"; cat "$CT/boot.out"; exit 1; }
  wait_health
  docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > "$CT/posture.txt" || true
  docker exec arc-test sh -c 'grep -m1 -o "draft_sample_method.: .probabilistic" /root/serve_full.log' > "$CT/spec_method.txt" 2>/dev/null || echo "spec-method NOT in log" > "$CT/spec_method.txt"
  cat "$CT/posture.txt"; cat "$CT/spec_method.txt"

  # 3) flip-first (t0 path law)
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

  # 4) t1.0 throwaway (JIT absorb for the sampled path)
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

  # 5) run7n t0 + t1.0 xhigh
  python3 "$BENCH" --temperature 0 > "$CT/t0_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/t0_run7n.out"
  python3 "$BENCH" --temperature 1.0 --thinking xhigh > "$CT/t10_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/t10_run7n.out"

  # 6) collapse trace (TQ dtypes only — amended gate (i) evidence)
  if [ "$PRESET" != "fp8_e4m3" ]; then
    python3 $R/p65/p615/collapse_trace.py 10 > "$CT/collapse_trace.out" 2>&1
    grep -c "^req" "$CT/collapse_trace.out" || true
  fi

  # 7) spec gate + integrity/det
  python3 $R/arc_spec_gate.py > "$CT/spec_gate.out" 2>&1
  tail -3 "$CT/spec_gate.out"
  python3 $R/r42/r42_integrity.py > "$CT/integrity.out" 2>&1
  tail -3 "$CT/integrity.out"

  # 8) solo TTFT faces
  python3 $R/r70/p70_solo_ttft.py "$CT/ttft_solo.json" U32k,U64k > "$CT/ttft_solo.out" 2>&1
  grep -i "ttft\|U32k\|U64k" "$CT/ttft_solo.out" | head -8

  # 9) log save + audit
  docker exec arc-test cat /root/serve_full.log > "$CT/serve_full.log" 2>/dev/null
  E=$(grep -c "ERROR" "$CT/serve_full.log" || true)
  T=$(grep -c "Traceback" "$CT/serve_full.log" || true)
  I5=$(grep -c "INTEGRITY 5/5 PASS" "$CT/integrity.out" || true)
  D3=$(grep -c "DETERMINISM 3/3 PASS" "$CT/integrity.out" || true)
  echo "LEG $TAG GATES: ERROR=$E Traceback=$T integrity_5of5=$I5 det_3of3=$D3"
done

echo "P616B_MATRIX: DONE (W14 restored by trap)"
