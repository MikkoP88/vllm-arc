#!/usr/bin/env python3
"""trace_summary.py -- P6.35: decompose the p135 XPU chrome trace (rank1,
U32k-decode window, 32 execute_model calls) into kernel classes, per-step
device time, and device-idle gaps. Runs rig-side; emits a compact summary.
"""
import json
import re
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/root/build/p635/legA-20261110-053607/p135_prof_rank1.json"

CLS = [
    ("gemm", re.compile(r"gemm|Gemm|GEMM|matmul_kernel")),
    ("attention_gdn", re.compile(r"gdn|attention|attn|flash|paged|kv_cache|mqa|qkv")),
    ("allreduce", re.compile(r"allreduce|all_reduce|ccl|CCL|AllReduce|ar_kernel|col reduction")),
    ("copy_mem", re.compile(r"copy|memcpy|fill|memset|cat")),
    ("norm_act", re.compile(r"rms|layernorm|norm|silu|act|gelu|softmax")),
    ("sample", re.compile(r"sample|sampling|argmax|topk|topp|multinomial|softmax.*prob")),
    ("other_math", re.compile(r"reduce|elementwise|vectorized|unrolled|distribution")),
]


def classify(name):
    for c, rx in CLS:
        if rx.search(name):
            return c
    return "other"


def main():
    with open(PATH) as f:
        data = json.load(f)
    ev = data["traceEvents"]
    kern = [e for e in ev if e.get("ph") == "X"
            and e.get("cat") in ("kernel", "gpu_op", "xpu_kernel")]
    print("events total=%d kernels=%d" % (len(ev), len(kern)))
    if not kern:
        cats = defaultdict(int)
        for e in ev:
            cats[e.get("cat")] += 1
        print("cats:", dict(cats))
        return
    kern.sort(key=lambda e: e["ts"])

    # per-class totals
    tot = defaultdict(float)
    cnt = defaultdict(int)
    names = defaultdict(float)
    for e in kern:
        d = e.get("dur", 0) / 1e3  # us -> ms
        c = classify(e.get("name", ""))
        tot[c] += d
        cnt[c] += 1
        names[(c, e.get("name", "")[:80])] += d
    span = (kern[-1]["ts"] + kern[-1].get("dur", 0) - kern[0]["ts"]) / 1e6
    print("window span %.3f s; kernel sum %.3f s; busy %.1f%%" %
          (span, sum(tot.values()) / 1e3, 100 * sum(tot.values()) / 1e3 / span))
    for c in sorted(tot, key=tot.get, reverse=True):
        print("  %-13s %8.2f ms  n=%-6d  %5.1f%% of kernel time" %
              (c, tot[c], cnt[c], 100 * tot[c] / sum(tot.values())))

    print("\nTOP-20 kernels by total ms:")
    for (c, n), d in sorted(names.items(), key=lambda kv: -kv[1])[:20]:
        print("  %8.2f ms  %-12s %s" % (d, c, n))

    # device idle gaps
    gaps = []
    cur_end = None
    for e in kern:
        if cur_end is None:
            cur_end = e["ts"] + e.get("dur", 0)
            continue
        if e["ts"] > cur_end:
            gaps.append((e["ts"] - cur_end) / 1e3)
        cur_end = max(cur_end, e["ts"] + e.get("dur", 0))
    gaps.sort()
    gt = sum(gaps)
    print("\ndevice idle: total %.2f ms over %d gaps; top gaps (ms): %s" %
          (gt, len(gaps), [round(g, 2) for g in gaps[-12:]]))
    print("TRACE_SUMMARY_OK")


if __name__ == "__main__":
    main()
