#!/usr/bin/env python3
"""p135_trace_bins.py -- P6.35 leg J: segment the mixed prefill+decode XPU
trace. Bins kernel time per 100 ms; finds the steady-decode region (periodic
small bursts); reports per-step device busy/idle + kernel mix there; lists
top gaps with absolute ts so they can be matched to prefill vs decode.
Usage: p135_trace_bins.py TRACE.json [bin_ms]
"""
import json
import re
import sys
from collections import defaultdict

PATH = sys.argv[1]
BIN = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

CLS = [
    ("gemm", re.compile(r"gemm|Gemm|GEMM|matmul_kernel")),
    ("attention_gdn", re.compile(r"gdn|attention|attn|flash|paged|kv_cache|mqa|qkv|Chunk")),
    ("allreduce", re.compile(r"allreduce|all_reduce|ccl|CCL|AllReduce|ar_kernel|ReduceCopy|barrier|allgatherv")),
    ("copy_mem", re.compile(r"copy|memcpy|fill|memset|cat")),
    ("norm_act", re.compile(r"rms|layernorm|norm|silu|act|gelu|softmax")),
    ("sample", re.compile(r"sample|sampling|argmax|topk|topp|multinomial")),
    ("other_math", re.compile(r"reduce|elementwise|vectorized|unrolled|distribution|cutlass")),
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
    kern.sort(key=lambda e: e["ts"])
    t0 = kern[0]["ts"]

    # ---- 100 ms bins of kernel time ----
    bins = defaultdict(float)
    for e in kern:
        b = int((e["ts"] - t0) / (BIN * 1e3))
        bins[b] += e.get("dur", 0) / 1e3
    nb = max(bins) + 1
    print("bin (%.0f ms) kernel-ms per bin:" % BIN)
    for b in range(nb):
        v = bins.get(b, 0.0)
        print("  %6.2f s  %6.1f ms  %s" % (b * BIN / 1e3, v, "#" * int(v / 4)))

    # ---- merged busy intervals, top gaps with ts ----
    iv = []
    for e in kern:
        s, d = e["ts"], e.get("dur", 0)
        if iv and s <= iv[-1][1]:
            iv[-1][1] = max(iv[-1][1], s + d)
        else:
            iv.append([s, s + d])
    gaps = []
    for a, b in zip(iv, iv[1:]):
        gaps.append((b[0] - a[1], (a[1] - t0) / 1e6))
    gaps.sort(key=lambda g: -g[0])
    print("\ntop 25 gaps (ms @ offset_s):")
    for g, off in gaps[:25]:
        print("  %9.2f ms @ %7.3f s" % (g / 1e3, off))

    # ---- steady-decode region: find the densest periodic cluster of
    # ~30-45 ms periods. Use bursts: consecutive kernels separated < 8 ms
    # form a burst; burst period distribution ----
    bursts = []
    cur = [iv[0][0], iv[0][1], 1]
    for a, b in zip(iv, iv[1:]):
        if b[0] - a[1] < 8e3:  # <8 ms gap merges into same burst
            cur[1] = b[1]
            cur[2] += 1
        else:
            bursts.append(tuple(cur))
            cur = [b[0], b[1], 1]
    bursts.append(tuple(cur))
    print("\nbursts n=%d; showing bursts with dur 8-80 ms (decode-step size):" % len(bursts))
    dec = [(b, e, n) for s, e, n in [(x[0], x[1], x[2]) for x in bursts]
           for b in [s] if 8e3 <= (e - s) <= 80e3]
    small = [(s / 1e3, e / 1e3, n) for s, e, n in bursts if 8e3 <= (e - s) <= 80e3]
    print("  n_small_bursts=%d" % len(small))
    if len(small) >= 6:
        durs = [(e - s) for s, e, n in small]
        pers = [(small[i + 1][0] - small[i][0]) / 1e3 for i in range(len(small) - 1)]
        durs.sort(); pers.sort()
        print("  burst dur ms p50=%.2f p90=%.2f" % (durs[len(durs)//2], durs[int(len(durs)*0.9)]))
        print("  burst period ms p50=%.2f p90=%.2f" % (pers[len(pers)//2], pers[int(len(pers)*0.9)]))
        # kernel mix inside small bursts
        lo, hi = small[0][0], small[-1][1]
        tot = defaultdict(float); cnt = defaultdict(int)
        for e in kern:
            if lo <= e["ts"] <= hi:
                c = classify(e.get("name", ""))
                tot[c] += e.get("dur", 0) / 1e3
                cnt[c] += 1
        s_ = sum(tot.values())
        print("  kernel mix inside small-burst span (%.3f s span, %.3f s kernels):" % ((hi-lo)/1e6, s_/1e3))
        for c in sorted(tot, key=tot.get, reverse=True):
            print("    %-13s %8.2f ms n=%-5d %5.1f%%" % (c, tot[c], cnt[c], 100*tot[c]/s_))
        print("TRACE_BINS_OK")


if __name__ == "__main__":
    main()
