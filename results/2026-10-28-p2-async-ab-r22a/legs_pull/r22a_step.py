"""r22a_step.py — step decomposition on the r22a posture trace (prefix caching
OFF -> mamba 'none' -> async skip path). Reports, per the §10 P2 decomposition
row: step wall (window/steps), GPU busy fraction, Event.synchronize calls
(expect ZERO in-step), metadata-build CPU totals, get_output stats."""
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
py = [e for e in evs if e.get("cat") == "python_function"]

# step count via async_utils.record_batch frames (1 per engine step)
steps = [e for e in py if "async_utils.py(90): record_batch" in e["name"]]
print(f"steps (record_batch frames): {len(steps)}")

# merged GPU busy intervals
ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in kerns)
merged = []
for s, t in ivs:
    if merged and s <= merged[-1][1]:
        if t > merged[-1][1]:
            merged[-1] = (merged[-1][0], t)
    else:
        merged.append((s, t))
span = merged[-1][1] - merged[0][0] if merged else 0
busy = sum(t - s for s, t in merged)
print(f"window span {span/1e6:.2f}s  GPU busy {busy/1e6:.2f}s ({100*busy/span:.1f}%)")
if steps:
    n = len(steps)
    print(f"step wall = span/steps = {span/1e6/n*1000:.1f} ms  "
          f"GPU busy/step = {busy/1e6/n*1000:.1f} ms")

# gaps attributed to following kernel, top 8
gaps = []
for i in range(1, len(merged)):
    g = merged[i][0] - merged[i-1][1]
    if g > 1000:
        gaps.append((g, kerns[i]["name"][:60] if i < len(kerns) else "?"))
tot = sum(g for g, _ in gaps)
gaps.sort(reverse=True)
print(f"total gap time {tot/1e6:.2f}s; top gaps:")
for g, nme in gaps[:8]:
    print(f"  {g/1e3:8.2f} ms  before {nme}")

# Event.synchronize frames (THE r21b wall: p50 58 ms x78)
syncs = [e for e in py if "streams.py" in e["name"] and "synchronize" in e["name"]]
sagg = defaultdict(lambda: [0, 0.0])
for e in syncs:
    sagg[e["name"][:90]][0] += 1
    sagg[e["name"][:90]][1] += e["dur"]
print(f"\nEvent.synchronize python frames: {len(syncs)}")
for nme, (c, us) in sorted(sagg.items(), key=lambda kv: -kv[1][1])[:6]:
    ds = sorted(e["dur"] for e in syncs if e["name"][:90] == nme)
    print(f"  n={c:4d} tot={us/1e6:6.3f}s p50={ds[len(ds)//2]/1e3:7.2f}ms  {nme}")

# metadata path totals
md = defaultdict(float)
for e in py:
    n = e["name"]
    if ("build_attn_metadata" in n or "gdn_attn" in n or "prepare_attn" in n
            or "_prepare_decode_inputs" in n):
        md[n[:90]] += e["dur"]
print("\nmetadata-path frames (total):")
for nme, us in sorted(md.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {us/1e6:7.3f}s  {nme}")

# get_output totals
gos = [e for e in py if "async_utils.py(167): get_output" in e["name"]]
if gos:
    ds = sorted(e["dur"] for e in gos)
    n = len(ds)
    print(f"\nget_output: n={n} tot={sum(ds)/1e6:.3f}s p50={ds[n//2]/1e3:.2f}ms "
          f"p90={ds[(n*9)//10]/1e3:.2f}ms")
