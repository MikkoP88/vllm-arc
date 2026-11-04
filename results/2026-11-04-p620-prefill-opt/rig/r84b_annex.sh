#!/bin/bash
# p620 r84b — cert9 post-gate annexes — 2026-11-04
#   [A] mq annex with the FIXED env injection (the r84 gate [5] FATAL is
#       the cert8-gate [4] FATAL again: the sed targeted a literal
#       "-e VLLM_XPU_ENABLE_XPU_GRAPH=1" docker flag that does not exist —
#       the serve script builds GRAPH_ENV; the fixed sed appends the export
#       INSIDE GRAPH_ENV; p619e_annex.sh [B] precedent)
#   [B] W14 restore + REAL generation verify (w14_boot.sh fullres — the
#       radiance-test container was REMOVED this session; docker start
#       will not work) — this is the gate's [6] leg, run here because the
#       gate exits at [A]'s FATAL site
R=/root/build
C=$R/p620/cert9/gate
IMG=vllm-arc:v0.30.0-b70-cert9

w14_restore() {
  docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
  sleep 8
  for _i in $(seq 1 36); do
    docker rm -f arc-test >/dev/null 2>&1
    docker inspect arc-test >/dev/null 2>&1 || break
    sleep 5
  done
  docker stop radiance-test >/dev/null 2>&1
  (nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p620b.log 2>&1 &)
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
def call(max_tok, seed=None):
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
  docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log)' \
    > "$1/erraudit.txt" 2>/dev/null || echo "ERROR=?" > "$1/erraudit.txt"
  cat "$1/erraudit.txt"
  grep -q "ERROR=0" "$1/erraudit.txt" || echo "ERRAUDIT: FAIL ($1)"
}

echo "=================== P620 R84B ANNEX START $(date +%F\ %H:%M:%S) ==================="

echo "=== [A0] k3v4 t1-seed re-characterization (r82/r82b + cert7/cert8 annex method) $(date +%F\ %H:%M:%S) ==="
# The r84 gate's k3v4 longprobe flagged ONE face: essay t1_seed_identical=False
# (all four classes t0_identical, zero garble, finish=stop everywhere). The
# documented sporadic t1-seed face -> re-characterize on a FRESH k3v4 boot:
# 4 rounds x 4 classes seed pairs + t0 spots + different-seed control.
kill_arc
docker stop radiance-test >/dev/null 2>&1
sleep 5
mkdir -p $C/k3v4_rechar
cp $R/serve_arc_b70.sh $R/serve_p620_k3r.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k3v4_nc/" $R/serve_p620_k3r.sh
grep -q "VLLM_ARC_TQ_VERIFY_MODE" $R/serve_p620_k3r.sh && { echo "FATAL: mode env leaked"; exit 1; }
n=$(grep -c -- "--kv-cache-dtype turboquant_k3v4_nc" $R/serve_p620_k3r.sh)
[ "$n" = "1" ] || { echo "FATAL: dtype substitution wrong ($n)"; exit 1; }
cp $R/serve_p620_k3r.sh $C/k3v4_rechar/serve_used.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p620_k3r.sh > $C/k3v4_rechar/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $C/k3v4_rechar/boot.out || { echo "FATAL: k3v4 rechar boot failed"; exit 1; }
wait_health | tee $C/k3v4_rechar/health.txt || exit 1
docker exec arc-test sh -c "grep -m1 'v6 installed' /root/serve_full.log" > $C/k3v4_rechar/posture.txt || true
cat $C/k3v4_rechar/posture.txt
flip_first $C/k3v4_rechar/flip_first.out
cat $C/k3v4_rechar/flip_first.out
python3 $R/p618/p618d_t1seed_char.py > $C/k3v4_rechar/t1seed_char.out 2>&1
grep -E "t1_seed_identical|t0\]|CTRL|DONE" $C/k3v4_rechar/t1seed_char.out
docker exec arc-test cat /root/serve_full.log > $C/k3v4_rechar/serve_full.log 2>/dev/null || true
erraudit $C/k3v4_rechar

echo "=== [A0b] q4nc identity re-characterization (cert8 gate recorded this dtype cleanest; r84 shows essay t0 A-B-A variants + t1-seed miss — same char battery) $(date +%F\ %H:%M:%S) ==="
kill_arc
mkdir -p $C/q4nc_rechar
cp $R/serve_arc_b70.sh $R/serve_p620_q4r.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_4bit_nc/" $R/serve_p620_q4r.sh
grep -q "VLLM_ARC_TQ_VERIFY_MODE" $R/serve_p620_q4r.sh && { echo "FATAL: mode env leaked"; exit 1; }
n=$(grep -c -- "--kv-cache-dtype turboquant_4bit_nc" $R/serve_p620_q4r.sh)
[ "$n" = "1" ] || { echo "FATAL: dtype substitution wrong ($n)"; exit 1; }
cp $R/serve_p620_q4r.sh $C/q4nc_rechar/serve_used.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p620_q4r.sh > $C/q4nc_rechar/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $C/q4nc_rechar/boot.out || { echo "FATAL: q4nc rechar boot failed"; exit 1; }
wait_health | tee $C/q4nc_rechar/health.txt || exit 1
docker exec arc-test sh -c "grep -m1 'v6 installed' /root/serve_full.log" > $C/q4nc_rechar/posture.txt || true
cat $C/q4nc_rechar/posture.txt
flip_first $C/q4nc_rechar/flip_first.out
cat $C/q4nc_rechar/flip_first.out
python3 $R/p618/p618d_t1seed_char.py > $C/q4nc_rechar/t1seed_char.out 2>&1
grep -E "t1_seed_identical|t0\]|CTRL|DONE" $C/q4nc_rechar/t1seed_char.out
docker exec arc-test cat /root/serve_full.log > $C/q4nc_rechar/serve_full.log 2>/dev/null || true
erraudit $C/q4nc_rechar

echo "=== [A] mq annex (fixed GRAPH_ENV injection) $(date +%F\ %H:%M:%S) ==="
kill_arc
docker stop radiance-test >/dev/null 2>&1
sleep 5
mkdir -p $C/mqannex2
cp $R/serve_arc_b70.sh $R/serve_p620_mq2.sh
sed -i "s/--kv-cache-dtype fp8_e4m3/--kv-cache-dtype turboquant_k8v4/" $R/serve_p620_mq2.sh
sed -i 's|GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1;"|GRAPH_ENV="export VLLM_XPU_ENABLE_XPU_GRAPH=1; export VLLM_ARC_TQ_VERIFY_MODE=mq;"|' $R/serve_p620_mq2.sh
grep -q "VLLM_ARC_TQ_VERIFY_MODE=mq" $R/serve_p620_mq2.sh || { echo "FATAL: mq env not injected (sed target drifted)"; exit 1; }
n=$(grep -c -- "--kv-cache-dtype turboquant_k8v4" $R/serve_p620_mq2.sh)
[ "$n" = "1" ] || { echo "FATAL: dtype substitution wrong ($n)"; exit 1; }
cp $R/serve_p620_mq2.sh $C/mqannex2/serve_used.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash $R/serve_p620_mq2.sh > $C/mqannex2/boot.out 2>&1
grep ARC_SERVE_LAUNCHED $C/mqannex2/boot.out || { echo "FATAL: mq boot failed"; exit 1; }
wait_health | tee $C/mqannex2/health.txt || exit 1
docker exec arc-test sh -c "grep -m1 'v6 installed' /root/serve_full.log; grep -E 'KV cache size' /root/serve_full.log | head -1" > $C/mqannex2/posture.txt || true
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

echo "=== [B] W14 restore + generation verify (w14_boot fullres) $(date +%F\ %H:%M:%S) ==="
kill_arc
(nohup bash $R/w14_boot.sh fullres > $R/w14_restore_p620_final.log 2>&1 &)
ok=""
for i in $(seq 1 180); do
  grep -q "W13H_BOOT_DONE" $R/w14_restore_p620_final.log 2>/dev/null && { ok=1; break; }
  sleep 10
done
echo "W14_BOOT_DONE=$ok after ~$((i*10))s"
grep -E "HEALTH_OK|FRANCE|BOOT_DONE" $R/w14_restore_p620_final.log || tail -15 $R/w14_restore_p620_final.log
gen=$(curl -sf -m 240 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flashnext","stream":false,"temperature":0,"max_tokens":16,"messages":[{"role":"user","content":"Say OK"}]}' | head -c 120)
[ -n "$gen" ] && echo "W14_GEN_VERIFIED: $gen" || echo "W14_GEN_FAILED"
neng=$(docker exec radiance-test sh -c 'ps aux | grep "vllm serve" | grep -v grep | wc -l')
echo "W14_ENGINES=$neng (expect 1)"
PO=$(docker ps --format '{{.Names}} {{.Ports}}' | grep 8000 || echo NONE)
echo "PORT_8000_OWNER=$PO (expect radiance-test, arc-test absent)"
docker inspect arc-test >/dev/null 2>&1 && echo "ARC-TEST STILL UP (BAD)" || echo "arc-test removed OK"
echo "=================== P620 R84B ANNEX COMPLETE $(date +%F\ %H:%M:%S) ==================="
