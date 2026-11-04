#!/bin/bash
# p620 r82b — t1-seed re-characterization annex (cert7 disposition method)
# + the r82 legs the early LONGRETURN exit skipped.
#
# Context: r82's longprobe full flagged essay t1_seed_identical=False
# (t1_hashes 18ff99786c73 / bf7fec212f1f) -> probe v3 verdict
# CORRUPTION_SIGNAL; every t0 face GREEN (all 4 classes single-hash across
# cold/rehit/rehit2 + 3 churn cycles, garble empty, finish=stop). The
# cross-boot table (fp8 posture, same certified code) shows this is the
# documented sporadic seeded-temp-1.0 divergence face (cert7: simple flip,
# shipped with annex; probe verdict logic keys t1 drift as signal) —
# code/multi cross-boot hash drift is pre-existing (persistent-LMCache
# path), NOT L3. This leg re-characterizes ON THE SAME LIVE ENGINE
# (cert7 method): 8 rounds x 2 seeded essay t1 requests, t0 spot hashes
# (expect 648d66ae32c9 boot-stable), different-seed control (expect
# differs). Then: p65 full faces + trace fetch + error/WARN audit + kill.
set -uo pipefail
R=/root/build
CTROOT=$R/p620/r82
cd "$CTROOT"

echo "=== [r82b.1] essay t1-seed re-characterization $(date +%H:%M:%S) ==="
python3 - << 'PYEOF' > t1seed_rechar.out 2>&1
import importlib.util as iu
import json

spec = iu.spec_from_file_location("lp", "/root/build/p618/p618_longprobe.py")
lp = iu.module_from_spec(spec)
spec.loader.exec_module(lp)

essay = lp.PROMPTS["essay"]
MT1 = lp.MAXTOK_T1
MT0 = lp.MAXTOK_T0

# t0 spot check (boot-stable expected: 648d66ae32c9)
t0_hashes = []
for _ in range(2):
    r = lp.gen(essay, 0.0, MT0)
    t0_hashes.append(r["sha"])
    print(f"t0 spot: {r['sha']} finish={r['finish']} chars={r['chars']}", flush=True)

# 8 rounds x (a,b) seeded pairs
pairs = []
hash_counts = {}
for rnd in range(8):
    a = lp.gen(essay, 1.0, MT1, seed=1234)
    b = lp.gen(essay, 1.0, MT1, seed=1234)
    same = a["sha"] == b["sha"]
    pairs.append({"round": rnd, "a": a["sha"], "b": b["sha"], "same": same,
                  "finish": [a["finish"], b["finish"]],
                  "garble_a": lp.detect_garble("essay", a["combined"])})
    for h in (a["sha"], b["sha"]):
        hash_counts[h] = hash_counts.get(h, 0) + 1
    print(f"round {rnd}: a={a['sha']} b={b['sha']} same={same} "
          f"fin={a['finish']}/{b['finish']}", flush=True)

# different-seed control
c = lp.gen(essay, 1.0, MT1, seed=999)
print(f"seed999 control: {c['sha']}", flush=True)

n_same = sum(1 for p in pairs if p["same"])
n_stop = sum(1 for p in pairs if p["finish"] == ["stop", "stop"])
n_garble = sum(1 for p in pairs if p["garble_a"])
distinct = sorted(hash_counts.items(), key=lambda kv: -kv[1])
t0_stable = len(set(t0_hashes)) == 1
control_differs = c["sha"] not in hash_counts

print(f"\nRECHAR_SUMMARY: pairs_identical={n_same}/8 finish_stop={n_stop}/8 "
      f"garble_flags={n_garble} distinct_t1_hashes={len(hash_counts)} "
      f"dominant={distinct[0][1]}/16 t0_spot_stable={t0_stable} "
      f"t0_hash={t0_hashes[0]} seed999_control_differs={control_differs}",
      flush=True)

verdict = []
if n_garble == 0 and n_stop == 8 and t0_stable and control_differs and \
        distinct[0][1] >= 12:
    verdict = "KNOWN_FACE"
else:
    verdict = "INVESTIGATE"
print(f"RECHAR_VERDICT: {verdict} "
      "(sporadic seeded-t1 RNG-path divergence under probabilistic "
      "drafting — p618b Class-A semantics; NOT corruption/looping/dtype "
      "if KNOWN_FACE)", flush=True)

json.dump({"t0_hashes": t0_hashes, "pairs": pairs,
           "hash_counts": hash_counts, "control_seed999": c["sha"],
           "verdict": verdict},
          open("t1seed_rechar.json", "w"), indent=1)
PYEOF
cat t1seed_rechar.out
grep -q "RECHAR_VERDICT: KNOWN_FACE" t1seed_rechar.out \
  || { echo "RECHAR: INVESTIGATE — STOP"; exit 1; }

echo "=== [r82b.2] p65 full faces + trace capture $(date +%H:%M:%S) ==="
python3 $R/p65/p65_timeline.py timeline.json > timeline_stdout.txt 2>&1 || echo "timeline exit=$?"
grep -A5 'long-stream faces' timeline_stdout.txt
docker exec arc-test sh -c 'ls -la /tmp/arc_step_trace_*' > trace_ls.txt 2>&1 || true
cat trace_ls.txt
mkdir -p traces
for f in $(docker exec arc-test sh -c 'ls /tmp/arc_step_trace_* 2>/dev/null'); do
  docker cp "arc-test:$f" "traces/$(basename "$f")" >/dev/null 2>&1 || true
done
ls -la traces/

echo "=== [r82b.3] error/WARN audit $(date +%H:%M:%S) ==="
docker exec arc-test sh -c 'echo ERROR=$(grep -c ERROR /root/serve_full.log); echo TRACEBACK=$(grep -c Traceback /root/serve_full.log); echo WARN_RAW=$(grep -c WARNING /root/serve_full.log)' > erraudit.txt 2>/dev/null || true
cat erraudit.txt
docker exec arc-test sh -c 'grep WARNING /root/serve_full.log' | sed -E 's/[0-9]+/N/g' | sort -u > warn_classes.txt
wc -l warn_classes.txt
docker exec arc-test cat /root/serve_full.log > serve_full.log 2>/dev/null || true
docker exec arc-test bash -c "pkill -f 'vllm serve'" >/dev/null 2>&1
sleep 10
for _i in $(seq 1 36); do
  docker rm -f arc-test >/dev/null 2>&1
  docker inspect arc-test >/dev/null 2>&1 || break
  sleep 5
done
sleep 5
echo "=================== P620 R82B DONE $(date +%F\ %H:%M:%S) ==================="
