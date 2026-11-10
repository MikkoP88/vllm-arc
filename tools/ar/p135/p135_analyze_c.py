#!/usr/bin/env python3
"""p135_analyze_c.py -- P6.35 leg C/F analysis: decode-step decomposition
from ts-prefixed steps logs (spec3 vs spec0) + TTFT alignment.
ASCII-only output (cp1252 console law).

Usage: python p135_analyze_c.py <steps_spec3.log> <steps_spec0.log>
       [face_spec3.json] [face_spec0.json]
"""
import json
import re
import sys
from collections import defaultdict

PAT = re.compile(r"^(\d+\.\d+) (\S+)(.*)$")

# structural tags excluded from the flat per-tag table
STRUCT = {"STEP", "ADDR", "PROF_START", "PROF_DONE", "PROF_STOP_ERR",
          "PROF_ARM_ERR", "XRP_WRAP_TARGET"}


def parse(path):
    steps = []  # each: {tag: [ms...], wall, ts_open}
    cur = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = PAT.match(line)
        if not m:
            continue
        ts, tag, rest = float(m.group(1)), m.group(2), m.group(3)
        if tag in STRUCT and tag != "STEP":
            continue
        ms = re.search(r"(\d+\.\d+)", rest)
        if tag == "STEP":
            if cur is not None:
                cur["wall"] = float(ms.group(1))
                cur["ts_open"] = ts
                steps.append(cur)
            cur = defaultdict(list)
        elif cur is not None and ms:
            cur[tag].append(float(ms.group(1)))
    if cur is not None and "wall" in cur:
        steps.append(cur)
    return steps


def summarize(path, name):
    steps = parse(path)
    dec = [g for g in steps if g.get("EXEC") and g.get("wall", 1e9) < 300]
    n = len(dec)
    if not n:
        print("== %s: no decode steps" % name)
        return
    walls = sorted(g["wall"] for g in dec)
    print("== %s decode-ish steps n=%d wall p50=%.2f mean=%.2f p90=%.2f" % (
        name, n, walls[n // 2], sum(walls) / n, walls[int(n * .9)]))
    # dynamic tag discovery
    tags = sorted({t for g in dec for t in g
                   if t not in ("wall", "ts_open")})
    rows = []
    for t in tags:
        vals = [g.get(t) for g in dec]
        present = [v for v in vals if v]
        if not present:
            continue
        cnt = sum(len(v) for v in present)
        tot = sum(sum(v) for v in present)
        mx = sum(max(v) for v in present)
        rows.append((t, cnt / n, tot / n, mx / n))
    for t, cnt, tot, mx in rows:
        print("  %-5s lines/step=%.1f sum/step=%7.2f max/step=%7.2f" % (
            t, cnt, tot, mx))
    # engine residual: wall minus engine-thread serial regions (rank-max view)
    res = [g["wall"] - max(g.get("WEX", [0])) - max(g.get("WSM", [0])) -
           max(g.get("SCHED", [0])) for g in dec if g.get("WEX") and g.get("WSM")]
    if res:
        res = sorted(res)
        print("  engine residual (STEP - WEXmax - WSMmax - SCHED): "
              "p50=%.2f mean=%.2f ms" % (res[len(res) // 2], sum(res) / len(res)))
    # replay split: RFG wall vs XRP wall (same calls, different layers)
    if any("RFG" in g for g in dec) and any("XRP" in g for g in dec):
        rfg = [max(g["RFG"]) for g in dec if g.get("RFG")]
        xrpm = [max(g["XRP"]) for g in dec if g.get("XRP")]
        xrps = [sum(g["XRP"]) for g in dec if g.get("XRP")]
        if rfg and xrpm:
            rfg.sort(); xrpm.sort(); xrps.sort()
            print("  replay split: RFG max/step p50=%.2f | XRP max/step p50=%.2f"
                  " | XRP sum/step p50=%.2f" % (
                      rfg[len(rfg) // 2], xrpm[len(xrpm) // 2],
                      xrps[len(xrps) // 2]))


def ttft_align(steps_path, face_path, name):
    try:
        face = json.load(open(face_path))
    except Exception:
        return
    addr_ts = None
    first_sched = None
    first_exec = None
    first_step_done = None
    for line in open(steps_path, encoding="utf-8", errors="replace"):
        m = PAT.match(line)
        if not m:
            continue
        ts, tag, rest = float(m.group(1)), m.group(2), m.group(3)
        if tag == "ADDR" and addr_ts is None:
            addr_ts = ts
        elif addr_ts is not None:
            if tag == "SCHED" and first_sched is None and "ntok=8" in rest:
                first_sched = ts
            if tag == "EXEC" and first_exec is None and "ntok=7" in rest:
                first_exec = ts
            if tag == "STEP" and first_step_done is None and ts > addr_ts + 1:
                first_step_done = ts
            if first_step_done:
                break
    t_send = face.get("t_send_epoch")
    if not (t_send and addr_ts):
        print("== %s TTFT: insufficient ts" % name)
        return
    sched_ms = ((first_sched or addr_ts) - addr_ts) * 1e3 if first_sched else -1
    exec_ms = (((first_exec or addr_ts) - addr_ts) * 1e3 - sched_ms
               ) if first_exec else -1
    print("== %s TTFT decomposition (send->ADDR %.0f ms | ADDR->SCHED %.0f ms | "
          "SCHED->EXEC %.0f ms | exec->first-STEP %.0f ms | client TTFT %.0f ms)" % (
              name, (addr_ts - t_send) * 1e3, sched_ms, exec_ms,
              ((first_step_done or addr_ts) - addr_ts) * 1e3
              if first_step_done else -1,
              face.get("ttft_s", -1) * 1e3))


if __name__ == "__main__":
    s3, s0 = sys.argv[1], sys.argv[2]
    f3 = sys.argv[3] if len(sys.argv) > 3 else None
    f0 = sys.argv[4] if len(sys.argv) > 4 else None
    summarize(s3, "SPEC3")
    print()
    summarize(s0, "SPEC0")
    if f3:
        print()
        ttft_align(s3, f3, "SPEC3")
    if f0:
        ttft_align(s0, f0, "SPEC0")
