# VERDICT — r221d knob isolation: BTWIDTH is the SOLE carrier of the engine-shape cost; the mechanism is auto-split over-pricing at short ctx (2026-11-22)

Leg: arc7 Phase-8 r221d — one-at-a-time isolation of the r221a side
finding (engine shape costs the SHIPPING `_16` route +14-20% @1K,
~0 @32K). Wheel `62f234c8d0aa89c4aec3e8d2c6fa22ad` (same as r220-r221b),
m0, `P204_ESHAPE=1 P204_MAXSK=cap` in every cell (the r220a base).
Cells: base / +bt (`P204_BTWIDTH=4100`) / +pad (+`P204_BTPAD=stale`) /
+pool (`P204_POOLBLKS=11281`) / all — at ctx {1024, 8192}.

## Result — R221D_PASS, canaries clean (0.0) on all 10 cells

```
    fmt    ctx |     base |      +bt |     +pad |    +pool |      all
  mxfp8   1024 |    698.5 |  +99.2   |  +99.2   |   +0.5   |  +99.5
  mxfp8   8192 |    502.3 |  -99.9   |  -99.8   |   -0.5   | -100.0
  mxfp4   1024 |    524.2 |  +99.3   |  +99.2   |   +0.1   |  +99.4
  mxfp4   8192 |    377.9 |   +1.0   |   +1.0   |   -0.9   |   +1.6
  mxk8v4  1024 |    662.0 |  +98.9   |  +99.1   |   -0.1   |  +100.1
  mxk8v4  8192 |    464.6 |  -71.6   |  -71.7   |   -0.3   |  -72.0
```

(us/call deltas vs base; full rows in run.log.)

- **pool is a NO-OP everywhere** (±0.5 µs): the 11281-block
  over-allocation theory is dead — the walk never touches pool extent
  (per-row early-exit law, consistent with r212f).
- **pad adds NOTHING over bt** (bt ≈ pad ≈ all in every cell): the
  stale-entry theory is dead; only the table WIDTH matters.
- **bt is FLAT +~99 µs @1K across all three formats** (99.2/99.3/98.9):
  a constant additive cost, independent of format and of the walk —
  NOT a table-indirection read cost. At 8192 the same knob is
  −99.9/−71.6 (mxfp8/mxk8v4) and ~0 (mxfp4).

## Mechanism

The flat format-independent ±99 µs is the auto-split heuristic's
block-table-width term (the "341 bw" first stage of the r219b chain):
pricing at the engine's 4100-wide capacity table picks the ~128-deep
split ladder even when the TRUE walk is 1K. The extra splits'
work-groups launch, read their row limits, and early-exit — but the
REDUCE still pays for them: +99 µs of reduce overhead on a walk whose
real cost is ~50-100 µs. At 8K the deep splits genuinely help
(−100/−72 µs on mxfp8/mxk8v4: more parallelism on a real 8K walk);
mxfp4 @8K is unmoved because its auto pick is already deep at that
face (its BN=8 tune prices differently).

So the r221a "+14-20% @1K engine-shape cost" = over-split reduce
overhead at short ctx, priced by capacity (bt_width 4100 →
max_seqlen_k 262400). In the ENGINE this is REAL and structural: the
capture-freeze law means the bucket-cap estimator cannot pass the true
short ctx at capture, and replay runs no python.

## Lever priced (deferred to the post-cert loop)

A reduce-side skip-empty-splits surgery in the wheel (the reduce
kernel skips splits whose rows all early-exited, so over-pricing at
short ctx costs launch but not reduce) would reclaim ~99 µs/call × 16
full-attention layer-calls ≈ 1.6 ms/step at 1K-class ctx. The e2e
battery faces are 4K+ where the probe says the knob is neutral-to-
positive; the crossover sits between 1K and 8K, unmapped. NOT gated
before cert26: C++ reduce surgery = new wheel + full re-validation
(value gates, A/A, battery) for a short-ctx-only win — recorded as
the r222 candidate on the open-lever list instead.

## Artifacts

`p221d_{base,bt,pad,pool,all}_{1024,8192}.json` (probe JSONs),
`run.log` (full report + cell exits + W14 restore), VERDICT, SHA256SUMS.
Offline leg — zero address hits (no redaction needed).
