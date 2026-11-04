#!/bin/bash
# p619e post-gate annexes — 2026-11-04
#   [A] cert8 fp8 boot + t1-seed REP-LOOP re-characterization
#       (p619e_t1seed_repchar.py: 4 rounds x 4 classes seed-1234 pairs,
#       texts banked, window detail; different-seed controls; t0 spots)
#   [B] mq annex RE-RUN with the FIXED env injection (the gate [4] FATAL:
#       the sed targeted a literal "-e VLLM_XPU_ENABLE_XPU_GRAPH=1" docker
#       flag that does not exist — the serve script builds GRAPH_ENV; the
#       fixed sed appends the export INSIDE GRAPH_ENV)
#   [C] W14 restore (explicit + EXIT trap) with a GENERATION probe —
#       health-only proved insufficient (2026-11-04 incident: a dual-launch
#       cripple passed /health + /v1/models, then 500'd and died)
R=/root/build
C=$R/p618/cert8
IMG=vllm-arc:v0.30.0-b70-cert8

w14_restore() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  docker stop radiance-test >/dev/null 2>&1
  sleep 5
  docker start radiance-test >/dev/null 2>&1
  (nohup bash $R/w14_serve_only.sh > $R/w14_restore_p619e_annex.log 2>&1 &)
}
trap w14_restore EXIT

kill_arc() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 10
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 10
}

wait_health() {
  ok=""
  for i in $(seq 1 240); do
    curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; return 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

flip_first() {
  local out=$1
  python3 - > "$out" 2>&1 << 'PYEOF' || true
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
print("FLIP_FIRST:", c["finish_reason"], f"{time.perf_counter()-t0:.1f}s", "head:", repr(txt[:100]))
PYEOF
}

tps_probe() {
  local out=$1
  python3 - > "$out" 2>&1 << 'PYEOF' || true
import json, threading, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
def call(max_tok):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
            "max_tokens": max_tok, "messages": [{"role": "user", "content":
            "Write a detailed technical essay about GPU memory hierarchies."}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    return time.perf_counter() - t0, r["usage"]["completion_tokens"], r["choices"][0]["finish_reason"]
for i in (1, 2):
    dt, n, fin = call(512)
    print(f"solo{i}: tps={n / dt:.2f} tok={n} dt={dt:.1f}s finish={fin}")
res = [None] * 8
def agg(i):
    dt, n, fin = call(512)
    res[i] = (dt, n)
ts = [threading.Thread(target=agg, args=(i,)) for i in range(8)]
t0 = time.perf_counter()
[t.start() for t in ts]; [t.join() for t in ts]
wall = time.perf_counter() - t0
tot = sum(n for _, n in res)
print(f"agg8: tps={tot / wall:.2f} tok={tot} wall={wall:.1f}s per_stream=" +
      str([f"{n / dt:.1f}" for dt, n in res]))
PYEOF
}

erraudit() {
  { echo "ERROR=$(docker exec arc-test sh -c "grep -c 'ERROR' /root/serve_full.log" 2>/dev/null)"
    echo "TRACEBACK=$(docker exec arc-test sh -c "grep -c 'Traceback' /root/serve_full.log" 2>/dev/null)"
  } > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
}

echo "=== [A] cert8 fp8 + t1-seed repchar $(date +%F\ %H:%M:%S) ==="
mkdir -p $C/repchar/run
docker stop radiance-test >/dev/null 2>&1
kill_arc
cp $R/serve_arc_b70.sh $R/serve_p619e_repchar.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p619e_repchar.sh > $C/repchar/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $C/repchar/boot.out || { echo "FATAL: fp8 boot failed"; exit 1; }
wait_health | tee $C/repchar/health.txt || exit 1
docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > $C/repchar/posture.txt || true
cat $C/repchar/posture.txt
docker exec arc-test sh -c "grep -c 'P619 tq_spec_fix' /root/serve_full.log" > $C/repchar/fix_engaged_count.txt 2>/dev/null || echo 0 > $C/repchar/fix_engaged_count.txt
echo "FIX_ENGAGED_ON_FP8=$(cat $C/repchar/fix_engaged_count.txt) (expect 0)"
cd $C/repchar/run
python3 $C/p619e_t1seed_repchar.py t1seed_repchar > $C/repchar/repchar.out 2>&1
grep -cE 'pair_identical": true' $C/repchar/repchar.out || true
grep -E '"cls"' $C/repchar/repchar.out | head -20
tail -3 $C/repchar/repchar.out
cd $C
docker exec arc-test cat /root/serve_full.log > $C/repchar/serve_full.log 2>/dev/null || true
erraudit $C/repchar

echo "=== [B] mq annex (fixed GRAPH_ENV injection) $(date +%F\ %H:%M:%S) ==="
kill_arc
mkdir -p $C/mqannex2
cp $R/serve_arc_b70.sh $R/serve_p619e_mq2.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p619e_mq2.sh
sed -i 's|GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"|GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1; export VLLM_ARC_TQ_VERIFY_MODE=mq;"|' $R/serve_p619e_mq2.sh
grep -q "VLLM_ARC_TQ_VERIFY_MODE=mq" $R/serve_p619e_mq2.sh || { echo "FATAL: mq env not injected (sed target drifted)"; exit 1; }
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p619e_mq2.sh)
[ "$n" = "1" ] || { echo "FATAL: dtype substitution wrong ($n)"; exit 1; }
cp $R/serve_p619e_mq2.sh $C/mqannex2/serve_used.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p619e_mq2.sh > $C/mqannex2/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $C/mqannex2/boot.out || { echo "FATAL: mq boot failed"; exit 1; }
wait_health | tee $C/mqannex2/health.txt || exit 1
docker exec arc-test sh -c "grep -m1 'P619 tq_spec_fix' /root/serve_full.log; grep -E 'KV cache size' /root/serve_full.log | head -1" > $C/mqannex2/posture.txt || true
cat $C/mqannex2/posture.txt
if grep -q "v6 installed (active=True mode=mq mq_kernel=True)" $C/mqannex2/posture.txt; then
  echo "MQ POSTURE: OK (mode=mq engaged)"
else
  echo "MQ POSTURE: line differs — see posture.txt"
fi
flip_first $C/mqannex2/flip_first.out
cat $C/mqannex2/flip_first.out
tps_probe $C/mqannex2/tps.out
cat $C/mqannex2/tps.out
docker exec arc-test cat /root/serve_full.log > $C/mqannex2/serve_full.log 2>/dev/null || true
erraudit $C/mqannex2

echo "=== [C] W14 restore + generation verify $(date +%F\ %H:%M:%S) ==="
w14_restore
sleep 90
ok=""
for i in $(seq 1 90); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
  sleep 10
done
echo "W14_HEALTH=$ok after ~$((i*10))s"
gen=$(curl -sf -m 240 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flashnext","stream":false,"temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"Say OK"}]}' | head -c 120)
[ -n "$gen" ] && echo "W14_GEN_VERIFIED: $gen" || echo "W14_GEN_FAILED"
neng=$(docker exec radiance-test sh -c 'ps aux | grep "vllm serve" | grep -v grep | wc -l')
echo "W14_ENGINES=$neng (expect 1)"
docker inspect arc-test >/dev/null 2>&1 && echo "ARC-TEST STILL UP (BAD)" || echo "arc-test removed OK"
echo "=================== P619E ANNEX COMPLETE $(date +%F\ %H:%M:%S) ==================="
