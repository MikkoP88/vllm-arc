#!/bin/bash
# p619d leg t9 — ALL-DTYPE GATE ON THE SHIPPING MODE (P6.19d, 2026-11-04).
# t8/t8e: first MQ draft (SoA-layout) CORRUPTION_SIGNAL — REP-LOOP —
# root-caused to the WRONG OPS TREE: the wheel has two TQ pairs
# (ops/triton_turboquant_* = slot-interleaved AoS, the engine's default
# XPU path; ops/turboquant_soa/* = FlyDSL-gated, never used here); the
# draft targeted SoA, validated against SoA, read the AoS cache as
# zeros (t8f: mq_out exactly [0]; MQVAL3c killer test). Fix v6 ports
# the kernel to the AoS layout (patches/tq_mq_aos.py; MQVAL5 4/4
# presets offline-exact vs the engine's own pair; MQVAL6 reproduces
# the engine dump nonzero+det). t8g (graphs, mode=mq): ZERO garble all
# classes, essay/code deterministic — but multi/simple identity flips
# [A,B,A]/[A,B,B] + annex length-cutoff empties → still not
# identity-stable → CORRUPTION_SIGNAL by the conservative gate.
# DECISION: cert8 ships SPLIT default (t5/t7-proven clean+deterministic
# all dtypes); mq stays opt-in env, documented experimental with its
# t8g evidence (agg8 289.23 vs split 240.86 = +20%, solo parity).
# This leg runs the all-dtype gate on the SHIPPING path: fix v6
# mounted, NO VLLM_ARC_TQ_VERIFY_MODE env (default split), four dtypes:
#   turboquant_k8v4 / k3v4_nc / 4bit_nc / 3bit_nc
#     gate: v6 install line mode=split mq_kernel=True; flip coherent;
#     longprobe full ZERO garble (identity faces recorded, not gating —
#     Class-A analog, annex evidence in t7/t8e)
#   fp8_e4m3 control
#     gate: fix install line ABSENT (turboquant_attn never imports on
#     fp8 — the P6.19 patch cannot engage; t7 lesson: demanding the
#     marker here was a gate bug, not an engine fault) + KV pool line
# Evidence: /root/build/p618/p619c/t9_<tag>/ ; W14 restored by trap.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert6
CT=$R/p618/p619c
D=$R/p618/p619d

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
}
w14_restore() {
  kill_arc
  docker start radiance-test >/dev/null 2>&1
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p619t9.log 2>&1 &
}
trap w14_restore EXIT

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health"; docker exec arc-test tail -40 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

run_dtype() {  # $1 = dtype string, $2 = short tag
  local DTYPE=$1 TAG=$2
  mkdir -p "$CT/t9_$TAG"
  echo "=================== P619D LEG t9_$TAG ($DTYPE spec3 GRAPH=1 FIX v6 split default) $(date +%H:%M:%S) ==================="
  kill_arc
  docker stop radiance-test >/dev/null 2>&1; sleep 20

  SP=/opt/venv/lib/python3.12/site-packages
  cp $R/serve_arc_b70.sh $R/serve_p619t9_$TAG.sh
  sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype $DTYPE/" $R/serve_p619t9_$TAG.sh
  sed -i "/-v \/root\/build\/inductor_cache:\/root\/.inductor_cache \\\\/a\    -v $D/tq_spec_fix.py:$SP/vllm_gaudi/xpu/patches/tq_spec_fix.py:ro \\\\\n    -v $D/tq_mq_aos.py:$SP/vllm_gaudi/xpu/patches/tq_mq_aos.py:ro \\\\\n    -v $D/platform.py:$SP/vllm_gaudi/xpu/platform.py:ro \\\\" $R/serve_p619t9_$TAG.sh
  n=$(grep -c -- "--kv-cache-dtype $DTYPE" $R/serve_p619t9_$TAG.sh)
  m=$(grep -c "tq_spec_fix.py:" $R/serve_p619t9_$TAG.sh)
  k=$(grep -c "tq_mq_aos.py:" $R/serve_p619t9_$TAG.sh)
  e=$(grep -c "VLLM_ARC_TQ_VERIFY_MODE" $R/serve_p619t9_$TAG.sh)
  echo "SUBSTITUTIONS dtype=$n fixmount=$m mqmount=$k envexplicit=$e"
  [ "$n" = "1" ] && [ "$m" = "1" ] && [ "$k" = "1" ] && [ "$e" = "0" ] || { echo "FATAL: substitution wrong"; exit 1; }
  cp $R/serve_p619t9_$TAG.sh "$CT/t9_$TAG/serve_used.sh"

  ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
    bash $R/serve_p619t9_$TAG.sh > "$CT/t9_$TAG/boot.out" 2>&1
  grep ARC_SERVE_LAUNCHED "$CT/t9_$TAG/boot.out" || { echo "FATAL: launch failed"; cat "$CT/t9_$TAG/boot.out"; exit 1; }
  wait_health | tee "$CT/t9_$TAG/health.txt"

  docker exec arc-test sh -c "grep -m1 'P619 tq_spec_fix' /root/serve_full.log; grep -E 'KV cache size|GPU KV cache' /root/serve_full.log | head -2" > "$CT/t9_$TAG/posture.txt" || true
  cat "$CT/t9_$TAG/posture.txt"
  if [ "$TAG" = "fp8" ]; then
    # Pool NUMBER is free-memory-state dependent (cert7: 400,739 vs
    # 405,491 across days, identical posture) — record, don't hard-gate.
    # The gate that matters: the fix CANNOT engage on fp8 (the patched
    # module never imports) and the boot is healthy.
    grep -q "XPU KV cache size" "$CT/t9_$TAG/posture.txt" || { echo "FATAL: no KV pool line"; exit 1; }
    grep -q "tq_spec_fix" "$CT/t9_$TAG/posture.txt" && { echo "UNEXPECTED: fix engaged on fp8"; exit 1; }
    echo "FP8 CONTROL OK: fix absent by construction, pool recorded above"
  else
    grep -q "v6 installed (active=True mode=split mq_kernel=True)" "$CT/t9_$TAG/posture.txt" || { echo "FATAL: fix v6 split default not installed"; exit 1; }
  fi

  python3 - > "$CT/t9_$TAG/flip_first.out" 2>&1 << 'PYEOF'
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 832, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
c = r["choices"][0]
txt = c["message"].get("content") or c["message"].get("reasoning_content") or ""
print("FLIP_FIRST:", c["finish_reason"], f"{time.perf_counter()-t0:.1f}s", "head:", repr(txt[:120]))
PYEOF
  cat "$CT/t9_$TAG/flip_first.out"

  ( cd "$CT/t9_$TAG" && python3 $R/p618/p618_longprobe.py "t9_$TAG" full > "$CT/t9_$TAG/longprobe_full.out" 2>&1 )
  grep -E "^\[|LONGPROBE" "$CT/t9_$TAG/longprobe_full.out"

  docker exec arc-test sh -c 'grep -cE "ERROR|Traceback" /root/serve_full.log || true' > "$CT/t9_$TAG/erraudit.txt" || true
  cat "$CT/t9_$TAG/erraudit.txt"
  docker cp arc-test:/root/serve_full.log "$CT/t9_$TAG/serve_full.log" >/dev/null 2>&1 || true
  echo "LEG t9_$TAG DONE $(date +%H:%M:%S)"
}

run_dtype turboquant_k8v4 k8v4
run_dtype turboquant_k3v4_nc k3v4
run_dtype turboquant_4bit_nc q4nc
run_dtype turboquant_3bit_nc q3nc
run_dtype fp8_e4m3 fp8

echo "ALL-DTYPE MQ-DEFAULT GATE COMPLETE $(date +%H:%M:%S)"
