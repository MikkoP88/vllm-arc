#!/bin/bash
# p619d leg t8f — IN-ENGINE MQ DIFFERENTIAL (P6.19d instrument leg).
# t8 (graphs) AND t8e (eager) both garble with mode=mq — eager too, so
# NOT replay-class: an engine-input delta my offline harness missed.
# This leg mounts the v5i instrument: the MQ branch additionally
# computes row 0 through the STOCK single-row kernel on the SAME live
# inputs and logs the rel-difference + every input's shape/dtype/stride
# + a tensor dump (/root/p619f_dump.pt) for offline replay.
# Posture: cert6 + turboquant_k8v4 + spec3 + EAGER + mode=mq.
# No longprobe (verdict already banked); one flip request is enough to
# fire the instrument (first 3 MQ calls).
# Evidence: /root/build/p618/p619c/t8f/ ; W14 restored by trap.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert6
CT=$R/p618/p619c
D=$R/p618/p619d
mkdir -p "$CT/t8f"

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
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p619t8f.log 2>&1 &
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

echo "=================== P619D LEG t8f (k8v4 spec3 EAGER FIX v5i INSTR) $(date +%H:%M:%S) ==================="
kill_arc
# t8f v1 incident: launched while radiance-test (restored by t8e's exit
# trap) still held the GPUs/port — the health probe answered W14's
# engine (cert7 port-collision class) and the posture gate FATAL'd on
# the empty arc log. W14 MUST be stopped before the arc boot (and the
# trap restores it at leg end).
docker stop radiance-test >/dev/null 2>&1
sleep 20
docker ps --format '{{.Names}}' | grep -q '^radiance-test$' && { echo "FATAL: radiance-test still up"; exit 1; }

SP=/opt/venv/lib/python3.12/site-packages
cp $R/serve_arc_b70.sh $R/serve_p619t8f.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p619t8f.sh
sed -i "/-v \/root\/build\/inductor_cache:\/root\/.inductor_cache \\\\/a\    -v $D/tq_spec_fixi.py:$SP/vllm_gaudi/xpu/patches/tq_spec_fix.py:ro \\\\\n    -v $D/tq_mq_soa.py:$SP/vllm_gaudi/xpu/patches/tq_mq_soa.py:ro \\\\\n    -v $D/platform.py:$SP/vllm_gaudi/xpu/platform.py:ro \\\\" $R/serve_p619t8f.sh
sed -i 's/^GRAPH_ENV=""/GRAPH_ENV="export VLLM_ARC_TQ_VERIFY_MODE=mq;"/' $R/serve_p619t8f.sh
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p619t8f.sh)
m=$(grep -c "tq_spec_fixi.py:" $R/serve_p619t8f.sh)
k=$(grep -c "tq_mq_soa.py:" $R/serve_p619t8f.sh)
e=$(grep -c '^GRAPH_ENV="export VLLM_ARC_TQ_VERIFY_MODE=mq;"' $R/serve_p619t8f.sh)
echo "SUBSTITUTIONS dtype=$n fixmount=$m mqmount=$k modeenv=$e"
[ "$n" = "1" ] && [ "$m" = "1" ] && [ "$k" = "1" ] && [ "$e" = "1" ] || { echo "FATAL: substitution wrong"; exit 1; }
cp $R/serve_p619t8f.sh "$CT/t8f/serve_used.sh"

ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=0 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p619t8f.sh > "$CT/t8f/boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/t8f/boot.out" || { echo "FATAL: launch failed"; cat "$CT/t8f/boot.out"; exit 1; }
wait_health | tee "$CT/t8f/health.txt"

docker exec arc-test sh -c "grep -m1 'kv-cache-dtype turboquant_k8v4' /root/serve_full.log; grep -m1 'P619 tq_spec_fix v5i installed' /root/serve_full.log; grep -m1 'XPU KV cache size' /root/serve_full.log" > "$CT/t8f/posture.txt" || true
cat "$CT/t8f/posture.txt"
grep -q "v5i installed (active=True mode=mq mq_kernel=True)" "$CT/t8f/posture.txt" || { echo "FATAL: v5i not installed"; exit 1; }

python3 - > "$CT/t8f/flip_first.out" 2>&1 << 'PYEOF'
import json, time, urllib.request
body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
        "max_tokens": 64, "messages": [{"role": "user", "content": "Count from 1 to 20."}]}
t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"}), timeout=600))
c = r["choices"][0]
txt = c["message"].get("content") or c["message"].get("reasoning_content") or ""
print("FLIP_FIRST:", c["finish_reason"], f"{time.perf_counter()-t0:.1f}s", "head:", repr(txt[:120]))
PYEOF
cat "$CT/t8f/flip_first.out"

sleep 5
docker exec arc-test sh -c "grep -E 'P619F mqdiff' /root/serve_full.log | head -8" > "$CT/t8f/mqdiff.txt" || true
echo "=== P619F LINES (TP0) ==="; cat "$CT/t8f/mqdiff.txt"
docker cp arc-test:/root/p619f_dump.pt "$CT/t8f/p619f_dump.pt" >/dev/null 2>&1 || echo "dump cp failed"
ls -la "$CT/t8f/p619f_dump.pt" 2>/dev/null || true
docker cp arc-test:/root/serve_full.log "$CT/t8f/serve_full.log" >/dev/null 2>&1 || true

echo "LEG t8f COMPLETE $(date +%H:%M:%S)"
