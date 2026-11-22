# VERDICT — r220b e2e A/A: the `_64` tile REGRESSES under engine load (2026-11-22)

Leg: arc7 Phase-8 r220b — e2e A/A of `ARC_MXKV_P64NATIVE=1` (p64) vs
env-unset (ctl, the shipping `_16` sub-tile frame) on the SAME wheel
`62f234c8d0aa89c4aec3e8d2c6fa22ad`, mxfp4, n1, faces c8192/c16384/
c32768, one boot per cell, XPU graphs on, splits=auto (→128).

## Result — GATE_PASS (cells valid), LEVER VERDICT NEGATIVE

```
 cell |    ctx |     itl |  vs ctl | vs posture |    tps |  chunks | qual | errs
  p64 |   8192 |    48.7 |         |     +0.50% |   71.3 | ~   56 | OK | 0
  p64 |  16384 |    53.8 |         |     +5.12% |   63.9 | ~   57 | OK | 0
  p64 |  32768 |    64.0 |         |    +12.67% |   55.8 | ~   55 | OK | 0
  ctl |   8192 |    48.3 |         |     -0.35% |   68.0 | ~   54 | OK | 0
  ctl |  16384 |    51.0 |         |     -0.48% |   68.7 | ~   56 | OK | 0
  ctl |  32768 |    56.4 |         |     -0.64% |   67.0 | ~   52 | OK | 0
 p64 vs ctl @c8192: +0.86%
 p64 vs ctl @c16384: +5.63%
 p64 vs ctl @c32768: +13.40%
  p64 @32K vs fp8: 1.51x (posture 1.34x)
```

- p64 itl is WORSE, monotone in ctx, +13.40% at 32K — far outside the
  0.03-0.13% A/A spread class. Quality healthy on both cells
  (chunks 52-57 ≤ 80, 0 errors) — consistent with r220c value-pass.
- ctl tracks the banked ns128/auto posture within -0.64% — same-wheel
  behavior-identity of the r220a3 frame edits holds e2e too.

## The inversion, decomposed

Offline (r220a) says `_64` is 86 µs/call FASTER at 32K; e2e says
~475 µs/call SLOWER (7.6 ms/tok over 16 calls/step). ~560 µs/call of
cost appears only under engine load. Mechanism hypothesis: the `_64`
WG stages a 4× taller V image in SLM (DIRECTBF16 V), so per-WG SLM
quadruples; the engine runs 16-row shapes concurrently (16 calls/
step) and residency collapses, while the offline probe's single call
never sees the contention. The r213 probe/e2e sign-inversion class,
now with a concrete occupancy mechanism.

## Sequel

r220d brackets the split depth (p64 + forced ARC_MXKV_NSPLIT 64/256
at 32K): if neither face rescues ctl (56.4), the inversion is
fundamental occupancy and the tile lever closes e2e-negative (the
value + frame laws stand); the remaining r221 road would be V
sub-staging (16-row ping-pong V slices keeping the 4-SG K walk), not
split tuning.

## Posture

Unchanged: `_16` sub-tile frame, splits auto (→128). P64NATIVE stays
env-gated OFF (probe instrument only).
