#!/usr/bin/env python3
"""r81_analyze.py — P6.20 r81 L3 A/B/A/B analyzer.

Per phase (l3on1/off1/l3on2/off2) x per pid-file:
  - segment dt distributions (schedule, update_states, prep_inputs,
    execute_model, sample_tokens, draft, wfs, wfs_flush [handoff],
    wfs_body [arc-lmc-store worker], wfs_total, get_finished)
  - step-class split (execute_model>500ms = chunk-ish steps)
Then a cross-phase summary table of the decision metrics:
  wfs_total / wfs_flush / wfs_body / sample_tokens p50+p99 and the
  per-face TTFTs from each timeline_stdout.txt.
"""
import json
import os
import re
import sys
from collections import defaultdict

SEGS = ("schedule", "update_states", "prep_inputs", "execute_model",
        "sample_tokens", "draft", "wfs", "wfs_flush", "wfs_body",
        "wfs_total", "get_finished")


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round(p / 100.0 * (len(vals) - 1)))))
    return vals[k]


def dist(vals):
    if not vals:
        return "n=0"
    return (f"n={len(vals)} p50={pct(vals,50):.1f} p90={pct(vals,90):.1f} "
            f"p99={pct(vals,99):.1f} max={max(vals):.1f}")


def load_trace(path):
    segs = defaultdict(list)
    steps = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            seg = rec.get("seg")
            if not seg:
                continue
            segs[seg].append(rec["dt_ms"])
            steps[rec.get("step")][seg] = rec["dt_ms"]
    return segs, steps


def analyze_phase(tag, ctdir):
    print(f"\n================ phase {tag} ================")
    ttf = {}
    so = os.path.join(ctdir, "timeline_stdout.txt")
    if os.path.exists(so):
        for line in open(so):
            m = re.match(r"\s+(\w+): TTFT=\s*([\d.]+)s", line)
            if m:
                ttf[m.group(1)] = float(m.group(2))
    print(f"TTFT: {ttf}")
    for name in sorted(os.listdir(ctdir)):
        if not name.startswith("arc_step_trace_"):
            continue
        segs, steps = load_trace(os.path.join(ctdir, name))
        print(f"  [{name}] steps={len(steps)}")
        for seg in SEGS:
            if seg in segs:
                print(f"    {seg:16s} {dist(segs[seg])}")
        chunk = [s for s, d in steps.items() if d.get("execute_model", 0) > 500]
        rest = [s for s in steps if s not in chunk]
        for label, group in (("chunk(em>500ms)", chunk), ("decode/rest", rest)):
            if not group:
                continue
            fl = [steps[s]["wfs_flush"] for s in group if "wfs_flush" in steps[s]]
            wt = [steps[s]["wfs_total"] for s in group if "wfs_total" in steps[s]]
            em = [steps[s]["execute_model"] for s in group
                  if "execute_model" in steps[s]]
            print(f"    [{label}] n={len(group)} em {dist(em)} | "
                  f"wfs_total {dist(wt)} | wfs_flush {dist(fl)}")


def main():
    ctroot = sys.argv[1] if len(sys.argv) > 1 else "/root/build/p620/r81"
    for tag in ("l3on1", "off1", "l3on2", "off2"):
        d = os.path.join(ctroot, tag)
        if os.path.isdir(d):
            analyze_phase(tag, d)
    print("\n(Judgment: wfs_total p99 in l3on phases should collapse to the "
          "handoff cost while wfs_body carries the store; TTFT/gap deltas "
          "come from the timeline tables.)")


if __name__ == "__main__":
    main()
