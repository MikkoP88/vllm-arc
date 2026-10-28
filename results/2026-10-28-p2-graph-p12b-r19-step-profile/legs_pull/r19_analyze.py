"""r19_analyze.py — rank trace breakdown: kernel duration histograms, decode
span isolation (post-prefill), per-category sums inside the decode span."""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    tr = json.load(f)

evs = tr["traceEvents"] if isinstance(tr, dict) else tr
kerns = [e for e in evs
         if e.get("ph") == "X" and e.get("cat", "").lower() in
         ("kernel", "xpu_kernel", "gpu_op")]


def cat(name):
    n = name.lower()
    if "gemm" in n and ("fp8" in n or n.startswith("gemm_kernel")):
        return "gemm(fp8/gemm_kernel)"
    if "gemm" in n:
        return "gemm-other"
    if "allreduce" in n or "allgather" in n or "reducecopy" in n or "barrier" in n:
        return "collective"
    if "varlen_fwd" in n or "fa2" in n or "flash" in n:
        return "fa2-attn"
    if "gdn" in n or "chunk" in n or "gated_delta" in n:
        return "gdn"
    if "quant" in n:
        return "quant"
    if "rms_norm" in n or "layer_norm" in n:
        return "norm"
    if "silu" in n or "act_and_mul" in n:
        return "act"
    if "copy" in n or "memcpy" in n or "fill" in n or "elementwise" in n:
        return "copy/elementwise"
    if "rotary" in n or "rope" in n:
        return "rope"
    return "other"


if not kerns:
    print("NO KERNEL EVENTS; cats seen:",
          sorted({e.get("cat") for e in evs if e.get("cat")})[:20])
    sys.exit(0)

kerns.sort(key=lambda e: e["ts"])
t0, t1 = kerns[0]["ts"], kerns[-1]["ts"] + kerns[-1]["dur"]
print(f"kernels={len(kerns)} span={(t1-t0)/1e6:.2f}s")

# last fa2 (prefill marker) -> decode span
fa2_ts = [e["ts"] for e in kerns if "varlen_fwd" in e["name"]]
if fa2_ts:
    dec_start = max(fa2_ts) + 50_000  # 50ms after last prefill attn
else:
    dec_start = t0
dec = [e for e in kerns if e["ts"] >= dec_start]
print(f"decode span: {(t1-dec_start)/1e6:.2f}s, {len(dec)} kernels "
      f"(prefill excluded {(t1-t0)-(t1-de_start) if False else (dec_start-t0)/1e6:.2f}s)")

for label, evset in (("FULL-WINDOW", kerns), ("DECODE-SPAN", dec)):
    agg = defaultdict(lambda: [0, 0.0])
    for e in evset:
        c = cat(e["name"])
        agg[c][0] += 1
        agg[c][1] += e["dur"]
    total = sum(v[1] for v in agg.values()) or 1
    print(f"\n== {label} total_xpu={total/1e6:.3f}s ==")
    for c, (n, us) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"  {c:24s} {us/1e6:8.3f}s {100*us/total:5.1f}%  "
              f"n={n:5d} avg={us/max(1,n)/1000:8.1f}us")

# gemm duration histogram in decode span
g = sorted(e["dur"] for e in dec if cat(e["name"]).startswith("gemm"))
if g:
    import math
    print(f"\ndecode-span gemm calls={len(g)} min={g[0]/1000:.0f}us "
          f"p50={g[len(g)//2]/1000:.0f}us p90={g[int(len(g)*0.9)]/1000:.0f}us "
          f"max={g[-1]/1000:.0f}us")
    buckets = [0]*9
    for d in g:
        b = min(8, int(math.log2(max(1, d)) // 3))  # 2^3=8us buckets
        buckets[b] += 1
    for i, b in enumerate(buckets):
        if b:
            print(f"  gemm {2**(3*i)}-{2**(3*(i+1))}us: {b}")

# collective histogram in decode span
c_ = sorted(e["dur"] for e in dec if cat(e["name"]) == "collective")
if c_:
    print(f"decode-span collective calls={len(c_)} p50={c_[len(c_)//2]/1000:.0f}us "
          f"p90={c_[int(len(c_)*0.9)]/1000:.0f}us max={c_[-1]/1000:.0f}us")

# top 12 individual kernels by total time in decode span
byname = defaultdict(lambda: [0, 0.0])
for e in dec:
    byname[e["name"][:70]][0] += 1
    byname[e["name"][:70]][1] += e["dur"]
print("\n== top kernels (decode span) ==")
for nme, (n, us) in sorted(byname.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {us/1e6:7.3f}s n={n:4d} avg={us/n/1000:8.1f}us  {nme}")
