"""r21b_stacks.py — name the python call site inside the per-step stall.
Uses python_function events (with_stack=true) overlapping the big gaps."""
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

ivs = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in kerns)
merged = []
for s, t, n in ivs:
    if merged and s <= merged[-1][1]:
        if t > merged[-1][1]:
            merged[-1] = (merged[-1][0], t, merged[-1][2])
    else:
        merged.append((s, t, n))

stalls = [(merged[i - 1][1], merged[i][0]) for i in range(1, len(merged))
          if merged[i][0] - merged[i - 1][1] > 10_000 and "Gather" in merged[i][2]]
stalls = stalls[:20]
print(f"stalls sampled: {len(stalls)}")

py = [e for e in evs if e.get("cat") == "python_function"]
print(f"python_function events: {len(py)}")

# Deepest python frames inside each stall window, aggregated
agg = defaultdict(float)
for g0, g1 in stalls:
    for e in py:
        if e["ts"] >= g0 and e["ts"] < g1:
            ov = min(g1, e["ts"] + e["dur"]) - e["ts"]
            if ov > 1000:  # frames >1ms inside the window
                agg[e["name"][:110]] += ov

print("== python frames >1ms inside stall windows (top 25) ==")
for nme, us in sorted(agg.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {us/1e6:7.3f}s  {nme}")
