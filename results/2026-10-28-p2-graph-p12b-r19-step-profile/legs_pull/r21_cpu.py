"""r21_cpu.py — what CPU work occupies the 21ms pre-gather stalls?
Finds the big GPU gaps before VectorizedGatherKernel, then lists the CPU
ops (cpu_op + runtime events) that run INSIDE those gap windows."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    tr = json.load(f)

evs = tr["traceEvents"] if isinstance(tr, dict) else tr
kerns = sorted((e for e in evs
                if e.get("ph") == "X" and e.get("cat", "").lower() in
                ("kernel", "xpu_kernel", "gpu_op")), key=lambda e: e["ts"])
cpus = [e for e in evs if e.get("ph") == "X" and
        e.get("cat") in ("cpu_op", "user_annotation", "python_function")]

# merged GPU intervals + gap list
ivs = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in kerns)
merged = []
for s, t, n in ivs:
    if merged and s <= merged[-1][1]:
        if t > merged[-1][1]:
            merged[-1] = (merged[-1][0], t, merged[-1][2])
    else:
        merged.append((s, t, n))

stalls = []
for i in range(1, len(merged)):
    g0, g1 = merged[i - 1][1], merged[i][0]
    if g1 - g0 > 10_000 and "Gather" in merged[i][2]:
        stalls.append((g0, g1, merged[i][2]))
print(f"gather stalls >10ms: {len(stalls)}")

# CPU ops inside stall windows (worker process events share the clock)
agg = defaultdict(lambda: [0, 0.0])
for g0, g1, _ in stalls:
    seen = defaultdict(float)
    for e in cpus:
        if e["ts"] >= g0 and e["ts"] < g1:
            ov = min(g1, e["ts"] + e["dur"]) - e["ts"]
            if ov > 0:
                seen[e["name"][:60]] += ov
    for nme, us in seen.items():
        agg[nme][0] += 1
        agg[nme][1] += us

print(f"total stall time {sum(g1-g0 for g0,g1,_ in stalls)/1e6:.2f}s")
print("== CPU ops inside stall windows (self-overlap summed, top 20) ==")
for nme, (c, us) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:20]:
    print(f"  {us/1e6:7.3f}s  n={c:5d}  {nme}")

# any cuda/xpu sync markers in stalls?
rt = [e for e in evs if e.get("cat") in ("cuda_runtime", "runtime", "xpu_runtime")]
syn = defaultdict(lambda: [0, 0.0])
for g0, g1, _ in stalls:
    for e in rt:
        if e["ts"] >= g0 and e["ts"] < g1:
            nme = e["name"]
            if "Sync" in nme or "sync" in nme or "stream" in nme.lower():
                syn[nme][0] += 1
                syn[nme][1] += e["dur"]
print("== runtime sync/stream calls in stalls ==")
for nme, (c, us) in sorted(syn.items(), key=lambda kv: -kv[1][1])[:8]:
    print(f"  {us/1e6:7.3f}s  n={c:5d}  {nme}")
