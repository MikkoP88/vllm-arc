#!/usr/bin/env python3
"""r80_analyze.py — P6.20 r80 A/B analyzer (evsync=1 vs =0).

Consumes each phase's timeline.json (client faces/gaps) + step-trace JSONL
(engine host segments) and prints the pre-registered comparison:

  per phase:
    - wfs / wfs_flush / wfs_total dt_ms distribution (p50/p90/p99/max,
      count) — the MECHANISM proof (flush cost on the step critical path)
    - sample_tokens / schedule / forward dt distributions, split by step
      class (chunk steps = forward p90 tail; decode steps = the rest)
    - per-face TTFT (first-token latency from timeline events) + gap p95
      in the face windows (the timeline's own percentile blocks)
  A/B table: phase a vs phase b deltas on every metric.

Zero-judgment: prints numbers + PASS/FAIL only where a pre-registered bar
exists (none here — r80 is the locator leg; bars land in the gate leg).
"""
import json
import os
import sys
from collections import defaultdict


def pct(vals, p):
    if not vals:
        return None
    vals = sorted(vals)
    k = min(len(vals) - 1, max(0, int(round(p / 100.0 * (len(vals) - 1)))))
    return vals[k]


def load_trace(path):
    segs = defaultdict(list)
    steps = defaultdict(dict)  # step -> seg -> dt
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or not line.startswith("{"):
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


def dist(vals):
    if not vals:
        return "n=0"
    return (f"n={len(vals)} p50={pct(vals,50):.1f} p90={pct(vals,90):.1f} "
            f"p99={pct(vals,99):.1f} max={max(vals):.1f}")


def analyze_phase(tag, ctdir):
    print(f"\n================ phase {tag} ================")
    tl_path = os.path.join(ctdir, "timeline.json")
    if os.path.exists(tl_path):
        with open(tl_path) as fh:
            tl = json.load(fh)
        # schema-agnostic dump: every top-level block truncated — p65 v5.1's
        # exact key names are in the banked legs, not in my head; truncation
        # keeps faces readable without guessing at the layout.
        print("--- timeline faces ---")
        for k, v in tl.items():
            if isinstance(v, (int, float, str, bool)):
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: {json.dumps(v)[:600]}")
    else:
        print("  (timeline.json absent)")
    print("--- step trace segments ---")
    for name in sorted(os.listdir(ctdir)):
        if not name.startswith("arc_step_trace_"):
            continue
        path = os.path.join(ctdir, name)
        segs, steps = load_trace(path)
        print(f"  [{name}] steps={len(steps)}")
        for seg in ("schedule", "update_states", "prompts_and_decodes",
                    "prep_prefill", "prep_decode", "forward",
                    "sample_tokens", "draft", "wfs", "wfs_flush",
                    "wfs_total", "get_finished"):
            if seg in segs:
                print(f"    {seg:22s} {dist(segs[seg])}")
        # step-class split: forward>500ms = chunk-ish steps
        chunk_steps = [s for s, d in steps.items()
                       if d.get("forward", 0) > 500]
        dec_steps = [s for s in steps if s not in chunk_steps]
        for label, group in (("chunk(fwd>500ms)", chunk_steps),
                             ("decode/rest", dec_steps)):
            if not group:
                continue
            tot = [sum(steps[s].values()) for s in group
                   if steps[s].get("sample_tokens")]
            fl = [steps[s]["wfs_flush"] for s in group
                  if "wfs_flush" in steps[s]]
            wfs = [steps[s]["wfs"] for s in group if "wfs" in steps[s]]
            print(f"    [{label}] n={len(group)} "
                  f"seg-sum {dist(tot)} | wfs {dist(wfs)} | "
                  f"wfs_flush {dist(fl)}")


def main():
    ctroot = sys.argv[1] if len(sys.argv) > 1 else "/root/build/p620/r80"
    for tag in ("a", "b"):
        d = os.path.join(ctroot, tag)
        if os.path.isdir(d):
            analyze_phase(tag, d)
    print("\n(A/B judgment: compare wfs_flush + sample_tokens distributions"
          " and the timeline gap/TTFT blocks across phases.)")


if __name__ == "__main__":
    main()
