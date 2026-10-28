"""r29_prefill_analyze.py — single-8k-prefill attribution (p14). The profile
window contains exactly one 8192-token prefill (+2 decode steps): prefill is
EAGER on the product posture (FULL_DECODE_ONLY) so every kernel is visible.
Aggregates the whole rank trace by category (r27 categorizer), reports the
prefill span, per-category ms + %, AR count/total, top kernels, and the
band arithmetic vs the D bar (3944 ms).

usage: python3 r29_prefill_analyze.py [profdir]   (default /root/prof)
"""
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

BAR_MS = 3944.0
CTX = 8192


def cat(name):
    n = name.lower()
    if ("allreduce" in n or "all_gather" in n or "allgather" in n
            or "reducecopy" in n or "barrier" in n or "ccl" in n
            or "reduce_scatter" in n or "reducescatter" in n):
        return "collective"
    if "gemm" in n:
        return "gemm-fp8" if ("fp8" in n or n.startswith("gemm_kernel")) else "gemm"
    if "matmul" in n:
        return "gemm"
    if "varlen_fwd" in n or "fa2" in n or "flash" in n or "fmha" in n:
        return "fa-attn"
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


def analyze(path):
    print(f"\n########## {os.path.basename(path)} ##########")
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        tr = json.load(f)
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    ks = [e for e in evs
          if e.get("ph") == "X" and e.get("cat", "").lower() in
          ("kernel", "xpu_kernel", "gpu_op")]
    if not ks:
        print("NO KERNEL EVENTS")
        return
    ks.sort(key=lambda e: e["ts"])
    # prefill span = first kernel .. last varlen FA burst end (draft prefill
    # + decode tail after it is small; report both whole and prefill-only)
    vl = [e for e in ks if "varlen_fwd" in e["name"].lower()]
    if vl:
        pre_end = max(e["ts"] + e["dur"] for e in vl) + 3000
    else:
        pre_end = ks[-1]["ts"] + ks[-1]["dur"]
    pre = [e for e in ks if e["ts"] < pre_end]
    wall_ms = (pre[-1]["ts"] + pre[-1]["dur"] - pre[0]["ts"]) / 1000
    print(f"kernels total={len(ks)} prefill-span={len(pre)} "
          f"span wall={wall_ms:.0f} ms (bar {BAR_MS:.0f} — client TTFT adds "
          f"tokenize/queue/sampling)")

    for label, evs_ in (("PREFILL SPAN", pre),):
        a = defaultdict(lambda: [0, 0.0])
        byname = defaultdict(lambda: [0, 0.0])
        for e in evs_:
            c = cat(e["name"])
            a[c][0] += 1
            a[c][1] += e["dur"]
            byname[e["name"][:74]][0] += 1
            byname[e["name"][:74]][1] += e["dur"]
        tot = sum(v[1] for v in a.values()) or 1.0
        print(f"\n== {label}: content total {tot/1000:.0f} ms "
              f"({100*tot/(wall_ms*1000):.0f}% of span wall) ==")
        for c, (cnt, us) in sorted(a.items(), key=lambda kv: -kv[1][1]):
            print(f"  {c:16s} {us/1000:8.1f} ms {100*us/tot:5.1f}% "
                  f"n={cnt:5d} avg={us/cnt/1000:8.1f}u")
        coll = a.get("collective", (0, 0.0))
        print(f"\n  AR/collective: {coll[1]/1000:.1f} ms "
              f"({100*coll[1]/tot:.1f}% of content), {coll[0]} calls "
              f"(expect ~128 layer ARs + draft/misc; per-call "
              f"{coll[1]/max(1,coll[0])/1000:.0f}u)")
        print("\n== top kernels ==")
        for nm, (cnt, us) in sorted(byname.items(),
                                    key=lambda kv: -kv[1][1])[:14]:
            print(f"  {us/1000:8.1f} ms n={cnt:5d} avg={us/cnt/1000:8.1f}u  {nm}")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/root/prof"
    files = [f for f in sorted(glob.glob(os.path.join(d, "*.gz"))
                               + glob.glob(os.path.join(d, "*.json")))
             if "rank" in os.path.basename(f)]
    if not files:
        print(f"no rank traces in {d}")
        sys.exit(1)
    print(f"traces: {[os.path.basename(f) for f in files]}")
    for f in files:
        try:
            analyze(f)
        except Exception as exc:  # noqa: BLE001
            print(f"{f}: FAILED {exc!r}")
