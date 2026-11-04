#!/bin/bash
# p619d leg t8 — MQ-MODE SERVE GATE + PERF A/B + CLASS-A ANNEX (P6.19d).
# Offline MQVAL proved the tq_mq_soa kernel byte-identical to the split
# dispatch on all four presets. This leg runs it IN the engine under
# XPU graphs (the launch pattern differs: 2 kernels + per-call allocs
# per request vs q_len single-row calls), same posture as t5/t6p:
#   cert6 + turboquant_k8v4 + ARC_SPEC=3 + ARC_GRAPH=1 + fix v5 with
#   VLLM_ARC_TQ_VERIFY_MODE=mq (tq_mq_soa mounted).
# Gates: v5 install line mode=mq mq_kernel=True; flip probe coherent;
# longprobe full ZERO garble; 0 ERROR/Traceback.
# ANNEX (after battery, same live engine): essay t0 x8 rounds + a
# different-seed t1 control — re-characterizes the identity faces seen
# in t7 (k3v4 essay t1-seed flip; q4nc essay t0 [A,A,B] no-garble) as
# Class-A/RNG-path semantics exactly like cert7's t1-seed annex.
# PERF A/B vs t6p split (solo 54.50 / agg8 240.86) and cert7 fp8
# (84.50 / 299.73): same battery shape.
# Evidence: /root/build/p618/p619c/t8/ ; W14 restored by trap.
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert6
CT=$R/p618/p619c
D=$R/p618/p619d
mkdir -p "$CT/t8g"

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
  nohup bash $R/w14_serve_only.sh > $R/w14_restore_p619t8g.log 2>&1 &
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

echo "=================== P619D LEG t8g (k8v4 spec3 GRAPH=1 FIX v6 MODE=mq) $(date +%H:%M:%S) ==================="
kill_arc
docker stop radiance-test >/dev/null 2>&1; sleep 20

SP=/opt/venv/lib/python3.12/site-packages
cp $R/serve_arc_b70.sh $R/serve_p619t8g.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p619t8g.sh
sed -i "/-v \/root\/build\/inductor_cache:\/root\/.inductor_cache \\\\/a\    -v $D/tq_spec_fix.py:$SP/vllm_gaudi/xpu/patches/tq_spec_fix.py:ro \\\\\n    -v $D/tq_mq_aos.py:$SP/vllm_gaudi/xpu/patches/tq_mq_aos.py:ro \\\\\n    -v $D/platform.py:$SP/vllm_gaudi/xpu/platform.py:ro \\\\" $R/serve_p619t8g.sh
sed -i 's/GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"/GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1; export VLLM_ARC_TQ_VERIFY_MODE=mq;"/' $R/serve_p619t8g.sh
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p619t8g.sh)
m=$(grep -c "tq_spec_fix.py:" $R/serve_p619t8g.sh)
k=$(grep -c "tq_mq_aos.py:" $R/serve_p619t8g.sh)
e=$(grep -c "VLLM_ARC_TQ_VERIFY_MODE=mq" $R/serve_p619t8g.sh)
echo "SUBSTITUTIONS dtype=$n fixmount=$m mqmount=$k env=$e"
[ "$n" = "1" ] && [ "$m" = "1" ] && [ "$k" = "1" ] && [ "$e" = "1" ] || { echo "FATAL: substitution wrong"; exit 1; }
cp $R/serve_p619t8g.sh "$CT/t8g/serve_used.sh"

ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p619t8g.sh > "$CT/t8g/boot.out" 2>&1
grep ARC_SERVE_LAUNCHED "$CT/t8g/boot.out" || { echo "FATAL: launch failed"; cat "$CT/t8g/boot.out"; exit 1; }
wait_health | tee "$CT/t8g/health.txt"

docker exec arc-test sh -c "grep -m1 'P619 tq_spec_fix v6 installed' /root/serve_full.log; grep -E 'KV cache size|GPU KV cache' /root/serve_full.log | head -2" > "$CT/t8g/posture.txt" || true
cat "$CT/t8g/posture.txt"
grep -q "v6 installed (active=True mode=mq mq_kernel=True)" "$CT/t8g/posture.txt" || { echo "FATAL: fix v6 mq not installed"; exit 1; }

python3 - > "$CT/t8g/flip_first.out" 2>&1 << 'PYEOF'
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
cat "$CT/t8g/flip_first.out"

( cd "$CT/t8" && python3 $R/p618/p618_longprobe.py "t8g" full > "$CT/t8g/longprobe_full.out" 2>&1 )
grep -E "^\[|LONGPROBE" "$CT/t8g/longprobe_full.out"

# ---- CLASS-A ANNEX (same live engine; cert7 t1-seed-annex protocol) ----
python3 - > "$CT/t8g/annex_essay.out" 2>&1 << 'PYEOF'
import hashlib, json, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"

def gen(seed, temp):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": temp,
            "max_tokens": 2048, "messages": [{"role": "user", "content":
            "Write a reflective essay on how decentralization changes the "
            "economics of large-scale computing infrastructure."}]}
    if seed is not None:
        body["seed"] = seed
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900))
    c = r["choices"][0]
    txt = c["message"].get("content") or c["message"].get("reasoning_content") or ""
    return hashlib.sha256(txt.encode()).hexdigest()[:12], len(txt), c["finish_reason"]

print("== essay t0 x8 (identity face re-characterization) ==")
h0 = None
flips = 0
for i in range(8):
    h, n, fin = gen(None, 0.0)
    if h0 is None:
        h0 = h
    if h != h0:
        flips += 1
    print(f"  round{i}: {h} chars={n} finish={fin} {'SAME' if h == h0 else 'FLIP'}")
print(f"t0: {8 - flips}/8 identical to round0")

print("== t1 seed control (seed 7 x3 vs seed 11 x3) ==")
s7 = [gen(7, 1.0)[0] for _ in range(3)]
s11 = [gen(11, 1.0)[0] for _ in range(3)]
print(f"  seed7 : {s7} identical={len(set(s7)) == 1}")
print(f"  seed11: {s11} identical={len(set(s11)) == 1}")
print(f"  cross-seed differ: {set(s7) != set(s11)}")
PYEOF
cat "$CT/t8g/annex_essay.out"

# ---- PERF A/B (same battery shape as t6p) ----
python3 - > "$CT/t8g/tps.out" 2>&1 << 'PYEOF'
import json, time, urllib.request, concurrent.futures as cf

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = ("Write a detailed step-by-step explanation of how a flash-decoding "
          "split-KV attention kernel processes a long context, covering "
          "tiling, online softmax and the final reduction.")

def one(i):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
            "max_tokens": 512, "seed": 7,
            "messages": [{"role": "user", "content": PROMPT}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.perf_counter() - t0
    u = r["usage"]
    return (u["completion_tokens"], dt, r["choices"][0]["finish_reason"])

for phase in ("solo1", "solo2"):
    tok, dt, fin = one(0)
    print(f"{phase}: tps={tok/dt:.2f} tok={tok} dt={dt:.1f}s finish={fin}")

for _ in range(2):
    one(0)

t0 = time.perf_counter()
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(one, range(8)))
wall = time.perf_counter() - t0
tot = sum(r[0] for r in res)
fins = "".join("S" if r[2] == "stop" else "L" for r in res)
print(f"agg8: tps={tot/wall:.2f} tok={tot} wall={wall:.1f}s finishes={fins} "
      f"per_stream={[f'{r[0]/r[1]:.1f}' for r in res]}")
PYEOF
cat "$CT/t8g/tps.out"

docker exec arc-test sh -c 'grep -cE "ERROR|Traceback" /root/serve_full.log || true' > "$CT/t8g/erraudit.txt" || true
cat "$CT/t8g/erraudit.txt"
docker cp arc-test:/root/serve_full.log "$CT/t8g/serve_full.log" >/dev/null 2>&1 || true

echo "LEG t8g COMPLETE $(date +%H:%M:%S)"
