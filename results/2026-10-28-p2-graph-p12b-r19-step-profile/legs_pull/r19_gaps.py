"""r19_gaps.py — decode-step gap attribution: where does the wall time go?
Buckets inter-kernel idle gaps by the kernel that FOLLOWS each gap (the
blocker), prints one representative step timeline (top events)."""
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
                ("kernel", "xpu_kernel", "gpu_op")),
               key=lambda e: e["ts"])
print(f"kernels={len(kerns)}")

# merge overlaps (multi-stream): simple interval-merge on (ts, ts+dur)
ivs = []
for e in kerns:
    ivs.append((e["ts"], e["ts"] + e["dur"], e["name"]))
ivs.sort()
merged = []
for s, t, n in ivs:
    if merged and s <= merged[-1][1]:
        if t > merged[-1][1]:
            merged[-1] = (merged[-1][0], t, merged[-1][2])
    else:
        merged.append((s, t, n))

gaps = []
for i in range(1, len(merged)):
    g = merged[i][0] - merged[i - 1][1]
    if g > 0:
        gaps.append((g, merged[i][2]))  # gap before kernel n

total_gap = sum(g for g, _ in gaps)
span = merged[-1][1] - merged[0][0]
print(f"span={span/1e6:.2f}s busy={span-total_gap:.2f}s gap_total={total_gap/1e6:.2f}s ({100*total_gap/span:.0f}%)")

after = defaultdict(lambda: [0, 0.0])
for g, n in gaps:
    key = n[:52]
    after[key][0] += 1
    after[key][1] += g
print("\n== idle gaps ATTRIBUTED TO THE NEXT KERNEL (top 14 by total) ==")
for n, (c, us) in sorted(after.items(), key=lambda kv: -kv[1][1])[:14]:
    print(f"  {us/1e6:7.3f}s  n={c:5d} avg={us/c/1000:7.2f}ms  -> {n}")

# gap size histogram
big = sorted((g for g, _ in gaps), reverse=True)
if big:
    print(f"\ngap p50={big[len(big)//2]/1000:.2f}ms p90={big[int(len(big)*0.9)]/1000:.2f}ms "
          f"max={big[0]/1000:.1f}ms; gaps>1ms: {sum(1 for g in big if g>1000)}")
