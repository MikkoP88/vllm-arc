#!/bin/bash
# r30 battery — p40 first LMCache leg. Order law: correctness gates first,
# then LMCache gates, then perf diagnostics. NO profiler this leg.
# 1/6 survival (flip payload first traffic)      — regression cell
# 2/6 spec gate (acceptance)                     — regression cell
# 3/6 golden vs banked product ref               — regression cell (c)
# 4/6 LMCache gates (r30_lmc.py):
#      A warm-hit   8k re-send t2 <= 750 ms bar (D champion warm-hit)
#      B integrity  5 seeded prompts cold sha vs re-hit sha byte-identical
#      C churn      24 distinct 8k -> fill 100 GB lr_u + evict + re-hit both
#                  ends; serve log Stored/Retrieved counters > 0
# 5/6 run7n x1                                    — perf regression cell
# 6/6 stepprobe (timers-off TPOT reference face)
set -o pipefail
NAME=arc-test
LEG=/root/build/legs/arc-p3-r30
mkdir -p "$LEG"

echo "=== r30 1/6: survival (832-token flip payload as FIRST traffic) ==="
t0=$(date +%s.%N)
code=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req.json)
t1=$(date +%s.%N)
echo "large-first: HTTP $code in $(echo "$t1 $t0" | awk '{printf "%.2f", $1-$2}')s" | tee "$LEG/survival.txt"
code2=$(curl -s -o /dev/null -w "%{http_code}" -m 300 -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" -d @/root/build/r18/req_small.json 2>/dev/null) || code2=SKIP
echo "small-after: HTTP $code2" | tee -a "$LEG/survival.txt"
docker exec "$NAME" sh -c 'grep -ac "onset" /root/serve_full.log' > "$LEG/onset_count.txt" 2>/dev/null || true

echo "=== r30 2/6: spec gate (acceptance) ==="
python3 /root/build/arc_spec_gate.py --json-out "$LEG/spec_gate.json" 2>&1 | tee "$LEG/spec_gate.txt" || echo "SPEC_GATE_FAIL_EXIT=$?"

echo "=== r30 3/6: golden compare (vs banked product ref) ==="
python3 /root/build/p25/p32_golden.py --out "$LEG/golden_run.json" 2>&1 | tee "$LEG/golden_run.txt"
python3 /root/build/p25/p32_golden.py --run "$LEG/golden_run.json" --ref /root/build/p25/golden_ref.json 2>&1 | tee "$LEG/golden_compare.txt" || echo "GOLDEN_FAIL_EXIT=$?"

echo "=== r30 4/6: LMCache gates (warm-hit / integrity / churn) ==="
cat > /root/build/r30/r30_lmc.py <<'EOF'
"""r30_lmc.py — p40 LMCache gates. Runs IN the arc container (docker exec).
A: warm-hit — 8k seeded prompt max_tokens=1: t_cold (first sight, stores KV)
   then t_warm (re-send, retrieve path); gate t_warm <= 750 ms (D bar).
B: integrity — 5 seeded 8k prompts max_tokens=64 temp=0: cold sha vs re-hit
   sha must be byte-identical (retrieved-prefix generation == full recompute).
C: churn — 24 distinct 8k prompts max_tokens=1 to fill the 100 GB lr_u CPU
   cache and force eviction, then re-hit BOTH the most-recent (should hit) and
   the very first (should be evicted -> recompute-class time)."""
import hashlib
import json
import random
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3.8-27b-fp8"
OUT = {"warm_hit": {}, "integrity": [], "churn": {}}


def send(ids, max_tokens):
    body = json.dumps({"model": MODEL, "prompt": ids, "max_tokens": max_tokens,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        obj = json.load(r)
    dt = (time.perf_counter() - t0) * 1000
    text = obj["choices"][0]["text"]
    return dt, hashlib.sha256(text.encode()).hexdigest(), len(ids)


def ids_for(seed, n):
    rng = random.Random(seed)
    return [rng.randint(1000, 50000) for _ in range(n)]


# --- A: warm-hit timing ------------------------------------------------------
p0 = ids_for(20261028, 8192)
t_cold, sha_cold, _ = send(p0, 1)
time.sleep(2)
t_warm, sha_warm, _ = send(p0, 1)
OUT["warm_hit"] = {"t_cold_ms": round(t_cold, 1), "t_warm_ms": round(t_warm, 1),
                   "sha_match": sha_cold == sha_warm,
                   "bar_ms": 750.0,
                   "PASS": t_warm <= 750.0}
print(f"A warm-hit: cold={t_cold:.1f} ms  warm={t_warm:.1f} ms  "
      f"sha_match={sha_cold == sha_warm}  gate(t2<=750)={'PASS' if t_warm <= 750 else 'FAIL'}", flush=True)

# --- B: integrity (5 seeded prompts, cold vs re-hit, full text) --------------
for i in range(5):
    p = ids_for(30303000 + i, 8192)
    _, sha1, _ = send(p, 64)
    time.sleep(1)
    _, sha2, _ = send(p, 64)
    ok = sha1 == sha2
    OUT["integrity"].append({"i": i, "sha_cold": sha1, "sha_rehit": sha2, "match": ok})
    print(f"B integrity[{i}]: match={ok}  {sha1[:12]} vs {sha2[:12]}", flush=True)

# --- C: churn (24 distinct 8k, then re-hit both ends) -------------------------
churn_ids = [ids_for(40404000 + i, 8192) for i in range(24)]
tts = []
for i, p in enumerate(churn_ids):
    dt, _, _ = send(p, 1)
    tts.append(round(dt, 1))
OUT["churn"]["cold_ms"] = tts
print(f"C churn: 24 distinct 8k sent, cold ms={tts[:6]}...{tts[-3:]}", flush=True)
time.sleep(2)
t_recent, _, _ = send(churn_ids[23], 1)   # most recent under lr_u -> should hit
t_evict, _, _ = send(churn_ids[0], 1)     # oldest -> should be evicted
OUT["churn"]["rehit_recent_ms"] = round(t_recent, 1)
OUT["churn"]["rehit_evicted_ms"] = round(t_evict, 1)
print(f"C re-hit: most-recent={t_recent:.1f} ms (expect warm-class)  "
      f"oldest/evicted={t_evict:.1f} ms (expect cold-class)", flush=True)

with open("/root/r30_lmc.json", "w") as f:
    json.dump(OUT, f, indent=1)
ok_all = OUT["warm_hit"]["PASS"] and all(r["match"] for r in OUT["integrity"])
print("R30_LMC_DONE", "PASS" if ok_all else "CHECK-ABOVE", flush=True)
EOF
docker cp /root/build/r30/r30_lmc.py "$NAME":/root/r30_lmc.py
docker exec "$NAME" /opt/venv/bin/python /root/r30_lmc.py 2>&1 | tee "$LEG/r30_lmc.txt"
docker cp "$NAME":/root/r30_lmc.json "$LEG/r30_lmc.json" >/dev/null 2>&1 || true
echo "--- serve-log LMCache counters (Stored/Retrieved/evict) ---"
docker exec "$NAME" bash -c 'grep -aiE "stored|retrieved|evict" /root/serve_full.log | tail -12' | tee "$LEG/lmc_log_counters.txt"
docker exec "$NAME" bash -c 'grep -ac "Stored" /root/serve_full.log; grep -ac "Retrieved" /root/serve_full.log' | tee "$LEG/lmc_log_counts.txt" || true
echo "--- LMCStatsMonitor / usage telemetry presence (e04 slot) ---"
docker exec "$NAME" bash -c 'grep -aiE "statsmonitor|usage" /root/serve_full.log | tail -5' | tee "$LEG/lmc_statsmon.txt" || true

echo "=== r30 5/6: run7n x1 (perf regression cell) ==="
python3 /root/build/bench_run7n.py 2>&1 | tee "$LEG/run7n_1.txt"

echo "=== r30 6/6: stepprobe (timers-off TPOT reference face) ==="
docker exec "$NAME" /opt/venv/bin/python /root/r22a_stepprobe.py 2>&1 | tee "$LEG/stepprobe.txt"

docker exec "$NAME" bash -c 'grep -a "arc-p10" /root/serve_full.log | tail -4' | tee "$LEG/p10_stats.txt" || true
docker exec "$NAME" cp /root/serve_full.log "$LEG/serve_r30_final.log" 2>/dev/null || docker cp "$NAME":/root/serve_full.log "$LEG/serve_r30_final.log"
echo "R30_BENCH_DONE"
