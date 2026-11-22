# VERDICT — r221a engine-shape probe: the offline instrument CANNOT reproduce the `_64` e2e penalty — hot-L2 hypothesis (2026-11-22)

Leg: arc7 Phase-8 r221a — offline A/A of `_64` vs `_16` under the
FULL engine call shape, hunting the r220b/r220d e2e inversion's
mechanism. Wheel `62f234c8d0aa89c4aec3e8d2c6fa22ad` (same as r220).

## What was set (all previously-existing r212d knobs, unset in r220a)

`P204_BTWIDTH=4100` (engine block-table width), `P204_BTPAD=stale`
(stale beyond-fill entries), `P204_POOLBLKS=11281` (engine pool
over-allocation), `P204_MAXSK=cap` (capacity-priced extent 262400
with early-exit — already set in r220a), `P204_ESHAPE=1`; plus a
second pass at `NCAP=4` (the n1 engine face's real row count:
1 request × 4 draft positions — the r220a cells ran NCAP=32).

## Results — `_64` still wins everywhere offline

```
ncap=32 (engine shape):
    fmt    ctx |      e64 |      e16 |  64vs16 | e16 vs tight
  mxfp8   1024 |    374.5 |    798.3 |  -53.1% |       +14.3%
  mxfp8  32768 |    682.0 |    754.3 |   -9.6% |        -2.3%
  mxfp4   1024 |    277.6 |    623.6 |  -55.5% |       +18.9%
  mxfp4  32768 |    503.6 |    608.5 |  -17.2% |        +0.9%
  mxk8v4  1024 |    333.9 |    762.2 |  -56.2% |       +15.1%
  mxk8v4  32768 |    603.9 |    694.3 |  -13.0% |        -3.4%

ncap=4 (engine shape + engine row count):
  mxfp8  32768 |    686.8 |    770.7 |  -10.9% |        -0.2%
  mxfp4  32768 |    507.8 |    623.2 |  -18.5% |        +3.3%
  mxk8v4  32768 |    608.5 |    718.4 |  -15.3% |        -0.0%
```

Canaries clean (≤ 0.001953). Side observation: the engine SHAPE costs
`_16` +14..20% at 1K (table indirection / extent overhead dominated
at short ctx) and ~0 at 32K — the shape knobs are real and the probe
is now engine-faithful in tables, pool, extent, and row count.

## Verdict

The `_64` penalty does NOT reproduce offline under ANY combination of
the available shape knobs. Combined with the r220 correction (the 16
layer-calls are SEQUENTIAL — cross-call SLM contention ruled out) and
r220d (no split depth rescues it), the surviving hypothesis is
**HOT-L2 blindness**: the probe's 50× replay re-reads the same KV
pages (L2-resident; mxfp4 KV @32K ≈ 8 MB per call), while the
engine's layer-interleaved steps always read KV cold from HBM (L2
evicted by weight streaming between attention layers). A hot-cache
probe cannot see latency-hiding structure — and `_16`'s 4×-higher WG
count (4× more outstanding loads) is exactly a latency-hiding
advantage that cold HBM would reward and hot L2 erases.

## Next (r221b definition, not run)

Cold-graph harness knob: interleave a large dummy-buffer sweep
(≥ L2 size) between native calls INSIDE the captured graph, then
re-A/A. If `_64`'s win inverts cold → mechanism confirmed and the
tile lever reopens with a fix frame (restore latency-hiding: more
WGs or deeper async prefetch under the wide tile). If `_64` still
wins cold → the residual is engine-only beyond offline reach and the
lever stays closed by the e2e arbiter.
