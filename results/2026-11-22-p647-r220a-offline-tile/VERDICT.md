# VERDICT — r220a offline ladder: the `_64` tile halves short-ctx per-call time offline (2026-11-22)

Leg: arc7 Phase-8 r220a — graphtime µs/call of the
`ARC_MXKV_P64NATIVE` route vs the shipping `_16` sub-tile frame on
the SAME wheel (`62f234c8d0aa89c4aec3e8d2c6fa22ad`), engine geometry
(D=256 HREP=6 page 64, NCAP=32, ESHAPE=1, REPLAY=50).

## Timing — `_64` vs `_16` (same wheel)

```
    fmt    ctx |      _64 |      _16 |   delta | vs r212m3 base
   mxfp8   1024 |    375.3 |    698.5 |  -46.3% |  -43.0%
   mxfp8  32768 |    685.8 |    772.1 |  -11.2% |  -49.5%
   mxfp4   1024 |    277.7 |    524.3 |  -47.0% |  -53.3%
   mxfp4  32768 |    501.2 |    603.2 |  -16.9% |  -51.3%
  mxk8v4   1024 |    333.8 |    662.0 |  -49.6% |      n/a
  mxk8v4  32768 |    610.1 |    718.6 |  -15.1% |      n/a
```

Lane decomposition at 32K under the native tile (probe cells; canaries
vacuous by the harness law — r220c bank is the value instrument):

```
  mxfp8: K    83.5  Vg   -86.4  floor   241.7  full   685.8
  mxfp4: K    53.3  Vg    81.4  floor   194.4  full   501.2
  mxk8v4: K   129.3  Vg   -41.0  floor   203.0  full   610.1
```

## Reading

- The kv_tile 16→64 walk halves per-call time at 1K and takes another
  11-17% at 32K in the ISOLATED single-call probe — the r218 residual
  pools (floor 5.6 + union 8.7 ms/step) were the parallelism-starved
  walk.
- m0 canaries clean (0.000122-0.000244 @32K) but weak —
  graph-vs-eager self-reference class.
- The `_16` control cells double as same-wheel behavior-identity
  proof for the r220a3 frame edits on the shipping route.

## CAUTION (the r220b sequel)

This offline win does NOT transfer to the engine: r220b e2e A/A shows
p64 +0.86/+5.63/+13.40% itl vs ctl (worse) — the r213 probe/e2e
sign-inversion class. The single-call probe cannot see the residency
contention of 16 concurrent engine calls against a 4× taller per-WG
SLM V image. See the r220b and r220d banks before citing this table
as a perf claim.
