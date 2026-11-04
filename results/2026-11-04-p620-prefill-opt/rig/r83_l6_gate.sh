#!/bin/bash
# p620 r83 — L3 async-store GATE BATTERY on the bind-mount probe posture
# (cert8 image + vllm_v1_adapter.py [L1+L3] + step_trace.py [V2-fixed] +
# platform.py). The pre-registered HARD gates (plan P6.20 record):
#   integrity 5/5 + determinism 3/3 (r42 battery)
#   LONG-RETURN GATE: longprobe v? full + 3x churn->rehit, cross-cycle
#     single-hash, garble-empty, finish=stop  == the zero-KV/cached-
#     content proof under the async store (the decisive correctness gate)
#   run7n t0 in-family + t1.0-xhigh banked
#   spec gate
#   INSTRUMENT-V2 gate: worker traces contain sample_tokens/execute_model/
#     wfs/wfs_flush segments (the r81 V2 lesson closed)
#   p65 full faces on the gated posture (TTFT/gaps record + step-trace
#     mechanism capture: wfs_flush handoff vs wfs_body worker cost)
#   0 ERROR / WARN class audit
# NO W14 trap (radiance-test absent on this rig state; restore at end of
# all p620 rig work via w14_boot.sh fullres).
set -uo pipefail
R=/root/build
IMG=vllm-arc:v0.30.0-b70-cert8
SP=/opt/venv/lib/python3.12/site-packages
CTROOT=$R/p620/r83
P620=$R/p620/files
mkdir -p "$CTROOT"
cd "$CTROOT"

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
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && { ok=1; break; }
    sleep 5
  done
  [ -n "$ok" ] || { echo "FATAL: no health in 20 min"; docker exec arc-test tail -60 /root/serve_full.log; exit 1; }
  echo "HEALTH=OK after ~$((i*5))s"
}

churn() {
  local base=$1
  python3 > "$CTROOT/churn_$base.out" 2>&1 << PYEOF || true
import json, threading, time, urllib.request
def one(i, out):
    body = {"model": "qwen3.8-27b-fp8", "stream": False, "temperature": 0.0,
            "max_tokens": 256, "seed": $base + i,
            "messages": [{"role": "user", "content":
                f"Essay {i}: explain cache coherence in distributed systems, 200 words."}]}
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}), timeout=600))
    out.append((time.perf_counter() - t0, r["usage"]["completion_tokens"]))
res = []
ts = [threading.Thread(target=one, args=(i, res)) for i in range(4)]
[t.start() for t in ts]; [t.join() for t in ts]
print("CHURN 4x256 done:", [f"{d:.1f}s/{n}tok" for d, n in res])
PYEOF
  tail -1 "$CTROOT/churn_$base.out"
}

echo "=================== P620 R83 GATE START $(date +%F\ %H:%M:%S) ==================="

echo "=== [0] py_compile staged files inside the image (p618c law) ==="
docker run --rm --entrypoint /opt/venv/bin/python -v "$P620":/p620f "$IMG" \
  -m py_compile /p620f/vllm_v1_adapter.py /p620f/step_trace_v2.py /p620f/platform.py \
  && echo "PY_COMPILE: PASS" || { echo "PY_COMPILE: FAIL"; exit 1; }

echo "=== [1] boot L3 posture (async store ON, step trace ON) $(date +%H:%M:%S) ==="
kill_arc
docker run -d --name arc-test --privileged --network host --ipc host \
  --shm-size 64g \
  -v /models/qwen3.8-27b-fp8:/models/target:ro \
  -v /root/build/v127_stage/chat_template_qwen38_high.jinja:/root/chat_template_qwen38_high.jinja:ro \
  -v /root/build/neo_cache:/root/.neo_cache \
  -v /root/build/triton_cache:/root/.triton_cache \
  -v /root/build/inductor_cache:/root/.inductor_cache \
  -v "$P620/vllm_v1_adapter.py":$SP/vllm_gaudi/xpu/patches/lmcint/vllm_v1_adapter.py \
  -v "$P620/step_trace_v2.py":$SP/vllm_gaudi/xpu/patches/step_trace.py \
  -v "$P620/platform.py":$SP/vllm_gaudi/xpu/platform.py \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  --entrypoint /bin/bash "$IMG" -c 'sleep infinity'
sleep 2
cp $R/serve_arc_b70.sh serve_p620_r83.sh
sed -i "s|export VLLM_XPU_ENABLE_XPU_GRAPH=1;|export VLLM_XPU_ENABLE_XPU_GRAPH=1; export ARC_STEP_TRACE=1; export VLLM_ARC_LMC_ASYNC_STORE=1;|" serve_p620_r83.sh
grep -q "VLLM_ARC_LMC_ASYNC_STORE=1" serve_p620_r83.sh || { echo "FATAL: env inject failed"; exit 1; }
cp serve_p620_r83.sh serve_used.sh
ARC_IMAGE=$IMG ARC_NAME=arc-test ARC_GRAPH=1 ARC_SPEC=3 ARC_LMC=1 \
  bash serve_p620_r83.sh > boot.out 2>&1
grep ARC_SERVE_LAUNCHED boot.out || { echo "FATAL: launch failed"; cat boot.out; exit 1; }
wait_health | tee health.txt
docker exec arc-test sh -c "grep -m4 'step_trace INSTALLED' /root/serve_full.log" > posture.txt 2>&1 || true
cat posture.txt
[ -s posture.txt ] || { echo "FATAL: step_trace INSTALLED absent"; exit 1; }

python3 - > flip_first.out 2>&1 << 'PYEOF' || true
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
cat flip_first.out

echo "=== [1b] INSTRUMENT-V2 gate: segments present in worker traces ==="
docker exec arc-test /opt/venv/bin/python -c "
import json, glob
segs = set()
files = glob.glob('/tmp/arc_step_trace_r*.jsonl')
for f in files:
    for line in open(f):
        try:
            segs.add(json.loads(line)['seg'])
        except Exception:
            pass
print('FILES', sorted(files))
print('SEGS', sorted(segs))
need = {'sample_tokens', 'execute_model', 'wfs', 'wfs_flush'}
print('INSTRUMENT_V2', 'PASS' if need <= segs else 'FAIL')
" > instrument_v2.out 2>&1
cat instrument_v2.out
grep -q "INSTRUMENT_V2 PASS" instrument_v2.out || { echo "INSTRUMENT-V2: FAIL"; exit 1; }
docker exec arc-test sh -c "grep -m2 'store worker ACTIVE' /root/serve_full.log" > posture_l3.txt 2>&1 || true
cat posture_l3.txt
grep -q "store worker ACTIVE" posture_l3.txt || { echo "FATAL: L3 worker not ACTIVE"; exit 1; }
docker exec arc-test sh -c 'grep -E "KV cache size" /root/serve_full.log | head -1' > kvpool.txt || true
cat kvpool.txt

echo "=== [2] integrity + determinism (r42 battery) $(date +%H:%M:%S) ==="
python3 $R/r42/r42_integrity.py > integrity.out 2>&1
I5=$(grep -c "INTEGRITY 5/5 PASS" integrity.out || true)
D3=$(grep -c "DETERMINISM 3/3 PASS" integrity.out || true)
echo "A-gates: integrity_5of5=$I5 det_3of3=$D3"
{ [ "$I5" = "1" ] && [ "$D3" = "1" ]; } || { echo "A-GATES: FAIL"; cat integrity.out; exit 1; }

echo "=== [3] run7n t0 + t1.0-xhigh $(date +%H:%M:%S) ==="
python3 $R/r46t/bench_run7n_t.py --temperature 0 > k3_t0_run7n.out 2>&1
grep -h 'aggregate=\|solo200' k3_t0_run7n.out
python3 $R/r46t/bench_run7n_t.py --temperature 1.0 --thinking xhigh > k3_t1.0_run7n.out 2>&1
grep -h 'aggregate=\|solo200' k3_t1.0_run7n.out

echo "=== [4] spec gate $(date +%H:%M:%S) ==="
python3 $R/arc_spec_gate.py > spec_gate.out 2>&1
tail -5 spec_gate.out

echo "=== [5] LONG-RETURN GATE (zero-KV proof under async store) $(date +%H:%M:%S) ==="
python3 $R/p618/p618_longprobe.py p620l3 full > longprobe_full.out 2>&1
grep -E "^\[|LONGPROBE" longprobe_full.out
for cyc in 1 2 3; do
  churn $((100 * cyc))
  python3 $R/p618/p618_longprobe.py p620l3r$cyc rehit > longprobe_rehit$cyc.out 2>&1
  grep -E "^\[|LONGPROBE" longprobe_rehit$cyc.out
done

python3 - > longreturn_gate_summary.txt 2>&1 << 'PYEOF'
import json
fail = []
def load(fn):
    with open(fn) as f:
        return json.load(f)
full = load("longprobe_p620l3_full.json")
rehits = [load(f"longprobe_p620l3r{c}_rehit.json") for c in (1, 2, 3)]
classes = ["essay", "code", "multi", "simple"]
for cls in classes:
    v = full.get(cls, {})
    hashes = set(v.get("hashes", []))
    garble_all = list(v.get("garble", {}).values()) + [v.get("t1_garble", [])]
    for r in rehits:
        rv = r.get(cls, {})
        hashes |= set(rv.get("hashes", []))
        garble_all += list(rv.get("garble", {}).values())
    flat = [g for g in garble_all if g]
    if len(hashes) != 1:
        fail.append(f"{cls}: cross-cycle hash drift {sorted(hashes)}")
    if flat:
        fail.append(f"{cls}: garble flags {flat}")
    if not v.get("t0_identical"):
        fail.append(f"{cls}: t0 not identical (cold/rehit/rehit2)")
    if not v.get("t1_seed_identical"):
        fail.append(f"{cls}: t1-seed not reproducible")
    if any(fin != "stop" for fin in v.get("finish", [])):
        fail.append(f"{cls}: truncation finish={v.get('finish')}")
    print(f"[{cls}] hashes={sorted(hashes)} finish={v.get('finish')} "
          f"chars={v.get('chars')} t1_seed_identical={v.get('t1_seed_identical')} "
          f"garble_empty={not flat}")
verdicts = [full.get("VERDICT")] + [r.get("VERDICT") for r in rehits]
print("probe verdicts:", verdicts)
if any(x != "CLEAN" for x in verdicts):
    fail.append(f"probe verdicts not all CLEAN: {verdicts}")
print("LONGRETURN_GATE:", "PASS (100% CLEAN, 4 classes x cold/rehit/rehit2 + 3 churn cycles)" if not fail else f"FAIL {fail}")
PYEOF
cat longreturn_gate_summary.txt
grep -q "LONGRETURN_GATE: PASS" longreturn_gate_summary.txt \
  || { echo "LONG-RETURN GATE: FAIL"; exit 1; }

echo "=== [6] p65 full faces + trace capture $(date +%H:%M:%S) ==="
python3 $R/p65/p65_timeline.py timeline.json > timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
grep -A5 'long-stream faces' timeline_stdout.txt
docker exec arc-test sh -c 'ls -la /tmp/arc_step_trace_*' > trace_ls.txt 2>&1 || true
cat trace_ls.txt
mkdir -p traces
for f in $(docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null'); do
  docker cp "arc-test:$f" "traces/$(basename "$f")" >/dev/null 2>&1 || true
done
ls -la traces/

echo "=== [7] error/WARN audit $(date +%H:%M:%S) ==="
docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log); echo WARN_RAW=$(grep -c WARNING /root/serve_full.log)' > erraudit.txt 2>/dev/null || true
cat erraudit.txt
docker exec arc-test sh -c 'grep WARNING /root/serve_full.log' | sed -E 's/[0-9]+/N/g' | sort -u > warn_classes.txt
wc -l warn_classes.txt
docker exec arc-test cat /root/serve_full.log > serve_full.log 2>/dev/null || true
kill_arc

echo "=================== P620 R83 GATE DONE $(date +%F\ %H:%M:%S) ==================="
