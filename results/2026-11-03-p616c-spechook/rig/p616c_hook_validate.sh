#!/bin/bash
# p616c_hook_validate.sh — P6.16c: validate the ARC_SPEC_METHOD serve hook
# (repo tools/serve_arc_b70.sh @ feat/arc-p616-spec-method-hook, commit
# f22acdbd) deployed to the rig BYTE-UNCHANGED (ZERO seds — unlike p616a/b,
# the hook IS the deployment; the flag enters via env only). Two boots, fp8
# default dtype, cert6 image, graph law, ARC_SPEC=3, W14 trap.
#
# STAGE 0 FLAG PROBE (no boot): extract the deployed script's SPEC block
#   (SPEC_METHOD_FLAG="" .. 2nd fi) and expand SPEC_FLAG for
#   unset / junk(typical) / probabilistic / spec-off(ARC_SPEC=0).
#   Gates: unset == junk byte-identical == the certified bytes
#   --speculative-config '{"method":"mtp","num_speculative_tokens":3}' ;
#   probabilistic appends exactly the p616b-validated string; spec-off empty.
# BOOT A default path (ARC_SPEC_METHOD unset = certified posture through the
#   hook): run7n t0 (bank agg#2 296.25, solo 84.51/84.41, gate ±1.5%), run7n
#   t1.0-xhigh greedy face (bank agg 186.48, solo 61.73/61.79), TTFT U32k/
#   U64k (bank 17.34/40.24), spec gate, integrity 5/5 + det 3/3, 0 ERROR/
#   Traceback, ps cmdline contains ZERO draft_sample_method.
# BOOT B opt-in path (ARC_SPEC_METHOD=probabilistic): same battery.
#   ps cmdline contains draft_sample_method; engine config dump parsed the
#   key; t1.0 face vs p616b prob bank (agg 241.10, solo 67.46/64.10; gates:
#   both legs >= 55 tok/s, pair >= greedy bank 61.73, agg >= 186.48); t0
#   in-band ±1.5% (posture-provably identical at temp 0); TTFT vs 17.37/
#   40.30; integrity + det; 0 errors.
set -uo pipefail
R=/root/build
CT=$R/p65/p616/p616c
IMG=vllm-arc:v0.30.0-b70-cert6
SCRIPT=$R/p65/p616/serve_arc_b70_p616c.sh
BENCH=$R/r70/bench_run7n_t5.py
mkdir -p "$CT"

w14_start() {
  docker stop arc-test >/dev/null 2>&1
  nohup bash $R/w14_boot.sh fullres > $R/w14_boot_p616c.log 2>&1 &
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

# ================= STAGE 0: flag probe (no boot) =================
echo "=================== P616C STAGE 0: FLAG PROBE ================="
awk '/^SPEC_METHOD_FLAG="";?$/{f=1} f{print} f && /^fi$/{c++; if(c==2) exit}' \
  "$SCRIPT" > "$CT/hook_block.sh"
wc -l "$CT/hook_block.sh"
cp "$CT/hook_block.sh" "$CT/hook_block_probe.sh"
cat >> "$CT/hook_block_probe.sh" <<'PEOF'
printf 'SPEC_FLAG=[%s]\n' "$SPEC_FLAG"
PEOF
ARC_SPEC=3 ARC_SPEC_METHOD=""     bash "$CT/hook_block_probe.sh" > "$CT/flag_unset.txt" 2>&1
ARC_SPEC=3 ARC_SPEC_METHOD=typical bash "$CT/hook_block_probe.sh" > "$CT/flag_junk.txt" 2>&1
ARC_SPEC=3 ARC_SPEC_METHOD=probabilistic bash "$CT/hook_block_probe.sh" > "$CT/flag_prob.txt" 2>&1
ARC_SPEC=0 ARC_SPEC_METHOD=probabilistic bash "$CT/hook_block_probe.sh" > "$CT/flag_specoff.txt" 2>&1
cat "$CT/flag_unset.txt" "$CT/flag_junk.txt" "$CT/flag_prob.txt" "$CT/flag_specoff.txt"
UJ=$(cmp -s "$CT/flag_unset.txt" "$CT/flag_junk.txt" && echo IDENTICAL || echo DIFFER)
CERT=$(grep -cF -- "--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":3}'" "$CT/flag_unset.txt" || true)
PROB=$(grep -cF ",\"draft_sample_method\":\"probabilistic\"}" "$CT/flag_prob.txt" || true)
OFF=$(grep -cF 'SPEC_FLAG=[]' "$CT/flag_specoff.txt" || true)
echo "PROBE GATES: unset_vs_junk=$UJ unset_is_certified_bytes=$CERT prob_appends_validated=$PROB specoff_empty=$OFF"
[ "$UJ" = IDENTICAL ] && [ "$CERT" = 1 ] && [ "$PROB" = 1 ] && [ "$OFF" = 1 ] \
  || { echo "FATAL: flag probe failed"; exit 1; }

# ================= battery (per boot) =================
battery() {  # $1=tag
  python3 - > "$CT/${1}_flip_first.out" 2>&1 << 'PYEOF'
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
  cat "$CT/${1}_flip_first.out"
  python3 - > "$CT/${1}_t10_warmup.out" 2>&1 << 'PYEOF'
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
  cat "$CT/${1}_t10_warmup.out"
  python3 "$BENCH" --temperature 0 > "$CT/${1}_t0_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/${1}_t0_run7n.out"
  python3 "$BENCH" --temperature 1.0 --thinking xhigh > "$CT/${1}_t10_run7n.out" 2>&1
  grep -h 'aggregate=\|solo200' "$CT/${1}_t10_run7n.out"
  python3 $R/arc_spec_gate.py > "$CT/${1}_spec_gate.out" 2>&1
  tail -3 "$CT/${1}_spec_gate.out"
  python3 $R/r42/r42_integrity.py > "$CT/${1}_integrity.out" 2>&1
  grep -E "seed|INTEGRITY|DETERMINISM" "$CT/${1}_integrity.out"
  python3 $R/r70/p70_solo_ttft.py "$CT/${1}_ttft_solo.json" U32k,U64k > "$CT/${1}_ttft_solo.out" 2>&1
  grep -i "ttft\|U32k\|U64k" "$CT/${1}_ttft_solo.out" | head -8
  docker exec arc-test sh -c "pgrep -af 'vllm serve'" > "$CT/${1}_ps_cmdline.txt" 2>/dev/null
  grep -c "draft_sample_method" "$CT/${1}_ps_cmdline.txt" || true
  docker exec arc-test cat /root/serve_full.log > "$CT/${1}_serve_full.log" 2>/dev/null
  E=$(grep -c "ERROR" "$CT/${1}_serve_full.log" || true)
  T=$(grep -c "Traceback" "$CT/${1}_serve_full.log" || true)
  I5=$(grep -c "INTEGRITY 5/5 PASS" "$CT/${1}_integrity.out" || true)
  D3=$(grep -c "DETERMINISM 3/3 PASS" "$CT/${1}_integrity.out" || true)
  echo "BOOT $1 GATES: ERROR=$E Traceback=$T integrity_5of5=$I5 det_3of3=$D3"
}

# ================= BOOT A: default path =================
echo "=================== P616C BOOT A: DEFAULT PATH (unset) ================="
docker inspect arc-test > "$CT/preexisting_arc-test.json" 2>/dev/null || rm -f "$CT/preexisting_arc-test.json"
kill_arc
cp "$SCRIPT" "$CT/serve_used.sh"
env -u ARC_SPEC_METHOD ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash "$SCRIPT" > "$CT/bootA.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/bootA.out" || { echo "FATAL: launch failed (bootA)"; cat "$CT/bootA.out"; exit 1; }
wait_health
docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > "$CT/bootA_posture.txt" || true
docker exec arc-test sh -c 'grep -m1 -o "draft_sample_method.: .probabilistic" /root/serve_full.log' > "$CT/bootA_spec_method.txt" 2>/dev/null || echo "spec-method NOT in log (expected, default path)" > "$CT/bootA_spec_method.txt"
cat "$CT/bootA_posture.txt"; cat "$CT/bootA_spec_method.txt"
battery bootA

# ================= BOOT B: opt-in path =================
echo "=================== P616C BOOT B: OPT-IN PATH (probabilistic) ================="
kill_arc
ARC_SPEC_METHOD=probabilistic ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 \
  bash "$SCRIPT" > "$CT/bootB.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/bootB.out" || { echo "FATAL: launch failed (bootB)"; cat "$CT/bootB.out"; exit 1; }
wait_health
docker exec arc-test sh -c 'grep -m1 -o "draft_sample_method.: .probabilistic" /root/serve_full.log' > "$CT/bootB_spec_method.txt" 2>/dev/null || echo "spec-method NOT in log (FATAL)" > "$CT/bootB_spec_method.txt"
cat "$CT/bootB_spec_method.txt"
grep -q probabilistic "$CT/bootB_spec_method.txt" || { echo "FATAL: engine did not parse probabilistic (bootB)"; exit 1; }
battery bootB

echo "P616C: DONE (W14 restored by trap)"
