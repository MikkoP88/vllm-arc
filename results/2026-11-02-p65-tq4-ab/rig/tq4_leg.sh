#!/bin/bash
# tq4_leg.sh — temporary one-variable A/B: --kv-cache-dtype
# fp8_e4m3 -> turboquant_4bit_nc on the cert5 image + certified 25-var
# chain. Everything else byte-identical to serve_arc_b70.sh. Baseline =
# the cert5 gate artifacts (2026-11-01-p65-cert5/, same image/rig).
# Trap restores W14 (recreate law: w14_boot.sh fullres ONLY).
set -uo pipefail
R=/root/build
CT=$R/p65/tq4
IMG=vllm-arc:v0.30.0-b70-cert5
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_tq4.log 2>&1 &
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

# 0) snapshot the pre-existing arc-test (not started by this leg) + stop it
docker inspect arc-test > "$CT/preexisting_arc-test.json" 2>/dev/null
docker rm -f arc-test >/dev/null 2>&1
docker stop radiance-test >/dev/null 2>&1
sleep 30   # r48h boot-drain law

# 1) build the ONE-VARIABLE serve script (verify exactly 1 substitution)
sed 's/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_4bit_nc/' \
  $R/serve_arc_b70.sh > $R/serve_arc_b70_tq4.sh
n=$(grep -c "turboquant_4bit_nc" $R/serve_arc_b70_tq4.sh)
f=$(grep -c "kv-cache-dtype fp8_e4m3" $R/serve_arc_b70_tq4.sh)
echo "SUBSTITUTIONS: tq=$n fp8_left=$f (must be 1/0)"
[ "$n" = "1" ] && [ "$f" = "0" ] || { echo "FATAL: sed did not produce a single-variable script"; exit 1; }
diff <(grep -oE "\-\-[a-z-]+ [^\\\\]+|export [A-Z_]+=" $R/serve_arc_b70.sh | head -80) \
     <(grep -oE "\-\-[a-z-]+ [^\\\\]+|export [A-Z_]+=" $R/serve_arc_b70_tq4.sh | head -80) \
  > "$CT/serve_diff.txt" || true
cat "$CT/serve_diff.txt"

# 2) boot + health
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash $R/serve_arc_b70_tq4.sh > "$CT/boot_tq4.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/boot_tq4.out" || { echo "FATAL: launch failed"; cat "$CT/boot_tq4.out"; exit 1; }
wait_health

# 3) KV pool + posture lines (the capacity claim: TQ 4bit ~ 1.9-2x pool)
docker exec arc-test sh -c 'grep -E "KV cache size|memory pool|Available KV|TurboQuant|kv_cache_dtype|mamba" /root/serve_full.log | head -20' \
  > "$CT/posture_lines.txt" || true
cat "$CT/posture_lines.txt"

# 4) flip-first (832)
python3 - << 'PYEOF' > "$CT/flip_first.out" 2>&1
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

# 5) run7n t0 + t1.0 xhigh (cert posture batteries)
B=$R/r46t/bench_run7n_t.py
python3 "$B" --temperature 0 > "$CT/tq4_t0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/tq4_t0_run7n.out"
python3 "$B" --temperature 1.0 --thinking xhigh > "$CT/tq4_t1.0_run7n.out" 2>&1
grep -h 'aggregate=\|solo200' "$CT/tq4_t1.0_run7n.out"

# 6) spec gate (MTP acceptance under 4bit KV)
python3 $R/arc_spec_gate.py > "$CT/spec_gate.out" 2>&1
tail -6 "$CT/spec_gate.out"

# 7) integrity + determinism
python3 $R/r42/r42_integrity.py > "$CT/integrity.out" 2>&1
tail -3 "$CT/integrity.out"

# 8) long-context faces (U32k + U64k, 12 bys — direct A/B vs cert5 gate)
cd $CT && P65_FACES="U32k,U64k" P65_W2MT=4000 python3 $R/p65/p65_timeline.py $CT/p65_tl_tq4_v51.json 2>&1 | tee $CT/p65_sum_tq4_v51.txt
grep "prompt sizes\|TTFT" $CT/p65_sum_tq4_v51.txt

# 9) save engine log + WARN/ERROR audit
docker exec arc-test cat /root/serve_full.log > "$CT/serve_full_tq4.log" 2>/dev/null
grep -E "WARNING|ERROR" "$CT/serve_full_tq4.log" | grep -vE "lmcache.logging|filelock|asyncio" > "$CT/serve_warn_error.txt" || true
wc -l "$CT/serve_warn_error.txt"
grep -cE "Traceback|ERROR" "$CT/serve_full_tq4.log" || echo 0
echo "TQ4_LEG: DONE (W14 restored by trap)"
