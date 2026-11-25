#!/usr/bin/env python3
"""r239b gemma agg8 honest-aggregate re-derivation (revalidation 2026-11-25).

Recomputes the agg8 face from the BANKED raw artifact (serve.log engine
telemetry) after the client-side tps_mean was found to be a numerator
artifact. Run from the repository root:

  python results/2026-11-24-p647-r239-record-lanes/reval_agg8.py

Method: vLLM logs "Avg generation throughput" every 10 s (tokens
generated in the interval / interval). Integrating each 10-s bucket's
token count over the client-side rep windows (spans 78.4 / 77.4 / 78.4 s
from gemma_faces.txt, barriers anchored on the log's own Running-count
transitions) yields per-rep totals; the run=8 buckets give the
full-concurrency steady aggregate.
"""
import re

LOG = "results/2026-11-24-p647-r239-record-lanes/r239b/gemma/serve.log"

pat = re.compile(
    r"11-24 (14:\d\d:\d\d).*?Avg generation throughput: ([\d.]+) "
    r"tokens/s, Running: (\d+) reqs")

rows = []
for line in open(LOG, encoding="utf8", errors="replace"):
    m = pat.search(line)
    if m:
        t = m.group(1)
        h, mn, s = t.split(":")
        secs = int(h) * 3600 + int(mn) * 60 + int(s)
        rows.append((secs, float(m.group(2)), int(m.group(3))))

win = [(t, g, r) for (t, g, r) in rows if t >= 14 * 3600 + 24 * 60]
for t, g, r in win:
    print("t={:02d}:{:02d}:{:02d} gen={:6.1f} run={} tokens_in_bucket={:6.0f}"
          .format(t // 3600, (t % 3600) // 60, t % 60, g, r, g * 10))

# rep barriers: barrier1 ~14:24:36 (last solo bucket ends 14:24:35);
# spans from the banked gemma_faces.txt lines (78.4 / 77.4 / 78.4 s).
b1 = 14 * 3600 + 24 * 60 + 36.0
b2 = b1 + 78.4
b3 = b2 + 77.4
b4 = b3 + 78.4
for i, (lo, hi) in enumerate([(b1, b2), (b2, b3), (b3, b4)], 1):
    total = 0.0
    for t, g, r in win:
        ov = max(0.0, min(t, hi) - max(t - 10, lo))
        total += g * ov
    print("rep{} [{:.1f}s span] integrated tokens = {:.0f}  "
          "-> honest wall aggregate = {:.1f} tok/s"
          .format(i, hi - lo, total, total / (hi - lo)))

print()
print("full-concurrency buckets (run=8):",
      [g for (t, g, r) in win if r == 8])
print("per-stream steady (agg/8 at run=8):",
      [round(g / 8, 1) for (t, g, r) in win if r == 8])
