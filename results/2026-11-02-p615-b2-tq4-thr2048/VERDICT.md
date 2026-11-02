# p615 B2 — TQ4 + thr2048: stalls under the 1.5 s bar, at +25/+20 % TTFT (2026-11-02)

One-variable leg off the cert6 TQ4 face: `turboquant_4bit_nc`,
MNBT 8192, `--long-prefill-token-threshold 2048` added (the p65
long-context profile's chunking knob). Chunks confirmed **1664-class
× 406** store lines (1 TQ page) + 10× 8320 aggregate-tail stores.
Battery clean: flip 10.2 s, integrity 5/5 + determinism 3/3,
0 ERROR / 0 Traceback, KV 757,641 (2.89×, thr does not change the
workspace).

## Result (`rig/p65_sum.txt`)

| Metric | TQ4 default (cert6) | fp8 true default (B1) | **TQ4 + thr2048 (B2)** |
|---|---|---|---|
| U32k / U64k TTFT | 15.85 / 33.98 s | 17.25 / 39.77 s | **21.60 / 47.78 s** |
| W_U32k_start p50 / p95 / max | 2729 / 3227 / 3329 ms | 3986 / 4299 ms | **1017 / 1159 / 1863 ms** |
| W_U64k_start p50 / p95 / max | 3188 / 3780 / 3918 ms | 5295 / 5623 ms | **1096 / 1329 / 1472 ms** |
| stalls (>2×b95) U32k / U64k | 6 / 11 | 36 / 60 | 102 / 190 |
| end windows p95 | 58 / 54 ms | 59–65 ms | 46 / 54 ms (clean) |

Stall reduction vs TQ4 default **−64 % / −65 %**, vs fp8 true default
**−71 % / −75 %** — and vs the historical thr2048-profile fp8 face
(1273–1304 / 1574–1604 ms) TQ4+thr2048 is at parity or better with
**2.89× the KV capacity**. The stall count roughly triples (36→102,
60→190): finer chunks = more, shorter stalls, exactly as the taxonomy
predicts (stall ≈ tokens-per-chunk / prefill-rate; the window p50 IS
one 1664-token chunk step now).

## Disposition vs the pre-registered rule

- Gaps clause: **PASS** — p95 1159/1329 ms ≤ ~1.5 s on both faces.
- TTFT clause: **FAIL** — 21.60/47.78 s vs the same-harness fp8-default
  family 17.25/39.77 s (B1, same p65 timeline wave) = +25/+20 %.
- Integrity clause: PASS (5/5 + 3/3).

So B2 is NOT the strict "low-stall profile" (the TTFT cost is the
chunk-granularity tradeoff itself: 406 chunk steps vs 114 means 406
prefix re-reads per long prompt). B3 passes TTFT but misses the gaps
bar (2.0 s) — the TQ page size (1664) quantizes the ladder to
1664/3328/6656, so no knob lands between B3's stalls and B2's TTFT.
Disposition is a documented Pareto menu, not a single face — see the
P6.15 close-out. B4 (fp8 + MNBT 4096) and B5 (TQ4 + MNBT 2048, tests
whether the B2 TTFT cost is the thr mechanism or the chunk size)
complete the matrix.

Redaction: serve_full.log +132 (8×rig +48, 12×node +84),
preexisting_arc-test.json +5 (1×host) — arithmetic-exact, raw-ref
grep 0 — `results/REDACTION-2026-11-02-p615.md`. W14 restored by trap.
