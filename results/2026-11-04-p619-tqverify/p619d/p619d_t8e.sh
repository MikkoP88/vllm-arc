#!/bin/bash
# p619d leg t8e — EAGER mq attribution leg (P6.19d).
# t8 (mq under XPU graphs) = CORRUPTION_SIGNAL (4/4 classes deterministic
# REP-LOOP). MQVAL + MQVAL2 prove the kernel eager-correct at real shapes
# (seq 4096, sparse pages, both key paths). This leg splits the remaining
# hypothesis space:
#   eager serve CLEAN  -> pure graph-replay divergence (t4f class) ->
#                        disposition: split default, mq experimental.
#   eager serve GARBLES-> in-engine integration bug (metadata plumbing my
#                        offline harness doesn't reproduce) -> findable.
# Posture: cert6 + turboquant_k8v4 + spec3 + ARC_GRAPH=0 (eager) + fix v5
# env VLLM_ARC_TQ_VERIFY_MODE=mq. Gates: install line; flip; longprobe
# full; erraudit. NO tps/annex (t8 already banked those faces).
# Evidence: /root/build/p618/p619c/t8e/ ; W14 restored by trap.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert6
CT=$R/p618/p619c
D=$R/p618/p619d
mkdir -p "$CT/t8e"

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
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p619t8e.log 2>&1 &
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

echo "=================== P619D LEG t8e (k8v4 spec3 EAGER FIX v5 MODE=mq) $(date +%H:%M:%S) ==================="
kill_arc

SP=/opt/venv/lib/python3.12/site-packages
cp $R/serve_arc_b70.sh $R/serve_p619t8e.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p619t8e.sh
sed -i "/-v \/root\/build\/inductor_cache:\/root\/.inductor_cache \\\\/a\    -v $D/tq_spec_fix.py:$SP/vllm_gaudi/xpu/patches/tq_spec_fix.py:ro \\\\\n    -v $D/tq_mq_soa.py:$SP/vllm_gaudi/xpu/patches/tq_mq_soa.py:ro \\\\\n    -v $D/platform.py:$SP/vllm_gaudi/xpu/platform.py:ro \\\\" $R/serve_p619t8e.sh
# GRAPH_ENV="" (line 144) is the unconditional initializer; the graph
# export at line 150 only runs when ARC_GRAPH=1 — so with ARC_GRAPH=0 the
# mode env must ride line 144.
sed -i 's/^GRAPH_ENV=""/GRAPH_ENV="export VLLM_ARC_TQ_VERIFY_MODE=mq;"/' $R/serve_p619t8e.sh
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p619t8e.sh)
m=$(grep -c "tq_spec_fix.py:" $R/serve_p619t8e.sh)
k=$(grep -c "tq_mq_soa.py:" $R/serve_p619t8e.sh)
e=$(grep -c '^GRAPH_ENV="export VLLM_ARC_TQ_VERIFY_MODE=mq;"' $R/serve_p619t8e.sh)
echo "SUBSTITUTIONS dtype=$n fixmount=$m mqmount=$k modeenv=$e"
[ "$n" = "1" ] && [ "$m" = "1" ] && [ "$k" = "1" ] && [ "$e" = "1" ] || { echo "FATAL: substitution wrong"; exit 1; }
cp $R/serve_p619t8e.sh "$CT/t8e/serve_used.sh"

ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=0 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p619t8e.sh > "$CT/t8e/boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/t8e/boot.out" || { echo "FATAL: launch failed"; cat "$CT/t8e/boot.out"; exit 1; }
wait_health | tee "$CT/t8e/health.txt"

docker exec arc-test sh -c "grep -m1 'P619 tq_spec_fix v5 installed' /root/serve_full.log; grep -m1 'XPU KV cache size' /root/serve_full.log; grep -m1 -iE 'graph posture|cudagraph_mode' /root/serve_full.log" > "$CT/t8e/posture.txt" || true
cat "$CT/t8e/posture.txt"
grep -q "v5 installed (active=True mode=mq mq_kernel=True)" "$CT/t8e/posture.txt" || { echo "FATAL: fix v5 mq not installed"; exit 1; }
grep -q "VLLM_XPU_ENABLE_XPU_GRAPH=True" "$CT/t8e/posture.txt" && { echo "FATAL: graphs ON in supposed-eager leg"; exit 1; }
echo "EAGER POSTURE CONFIRMED (no VLLM_XPU_ENABLE_XPU_GRAPH=True)"

python3 - > "$CT/t8e/flip_first.out" 2>&1 << 'PYEOF'
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
cat "$CT/t8e/flip_first.out"

( cd "$CT/t8e" && python3 $R/p618/p618_longprobe.py "t8e" full > "$CT/t8e/longprobe_full.out" 2>&1 )
grep -E "^\[|LONGPROBE" "$CT/t8e/longprobe_full.out"

docker exec arc-test sh -c 'grep -cE "ERROR|Traceback" /root/serve_full.log || true' > "$CT/t8e/erraudit.txt" || true
cat "$CT/t8e/erraudit.txt"
docker cp arc-test:/root/serve_full.log "$CT/t8e/serve_full.log" >/dev/null 2>&1 || true

echo "LEG t8e COMPLETE $(date +%H:%M:%S)"
