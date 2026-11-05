#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ar_summarize.py — P6.22 r94 offline analysis: ARC_AR_TRACE JSONL ->
per-(op, caller, size) collective histogram.

The census instrument (vllm_gaudi/xpu/patches/ar_trace.py) emits one
JSONL line per wrapped torch.distributed call. This tool folds them
into the table the P6.22 lane needs: which message sizes dominate the
TP2 allreduce family, from where, how often — the inputs to the r95
ceiling bench and the one-shot decision gate.

Completeness check (the free-function-wrap risk): the JSONL all_reduce
COUNT must match the profiler trace's c10d::allreduce_-linked kernel
call count (r67b: x792 per window) — a large deficit means some ARs go
through group-object methods and the wrap needs a vllm-symbol level.

Usage:
  python3 ar_summarize.py TRACE.jsonl [MORE.jsonl ...] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict


def human_bytes(n):
    if n is None:
        return "-"
    # n is already scaled to `unit` by the loop's last division — do NOT
    # divide again (attempt-1 defect: every unit >= KiB printed 1024x low)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} B" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} GiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    # (op, caller, bytes) -> [count, dt_total_ms, dt_max_ms]
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    op_totals = defaultdict(lambda: [0, 0.0])
    n_lines = 0
    for path in a.traces:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_lines += 1
                key = (rec.get("op", "?"), rec.get("caller", "?"),
                       rec.get("bytes"))
                slot = agg[key]
                dt = rec.get("dt_ms") or 0.0
                slot[0] += 1
                slot[1] += dt
                slot[2] = max(slot[2], dt)
                t = op_totals[rec.get("op", "?")]
                t[0] += 1
                t[1] += rec.get("bytes") or 0

    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])[:60]
    print(f"lines={n_lines} distinct(op,caller,bytes)={len(agg)}")
    print(f"{'op':<22s} {'caller':<58s} {'size':>10s} {'count':>7s} "
          f"{'dt_mean':>8s} {'dt_max':>8s} {'dt_tot':>9s}")
    for (op, caller, nbytes), (cnt, tot, mx) in rows:
        print(f"{op:<22s} {caller[:58]:<58s} "
              f"{human_bytes(nbytes):>10s} {cnt:>7d} "
              f"{tot / cnt:>8.3f} {mx:>8.3f} {tot:>9.1f}")
    print("--- op totals ---")
    for op, (cnt, by) in sorted(op_totals.items(),
                                key=lambda kv: -kv[1][1]):
        print(f"{op:<22s} count={cnt:>7d} bytes_tot={human_bytes(by)}")

    if a.json:
        out = {"lines": n_lines,
               "rows": [
                   {"op": k[0], "caller": k[1], "bytes": k[2],
                    "count": v[0], "dt_total_ms": round(v[1], 1),
                    "dt_max_ms": round(v[2], 3)}
                   for k, v in sorted(agg.items(),
                                      key=lambda kv: -kv[1][1])],
               "op_totals": {k: {"count": v[0], "bytes": v[1]}
                             for k, v in op_totals.items()}}
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
