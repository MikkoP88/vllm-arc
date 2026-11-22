# VERDICT — r220c value gate: the multi-SG `_64` MX route is value-correct (2026-11-22)

Leg: arc7 Phase-8 r220c — eager value validation of the
`ARC_MXKV_P64NATIVE` multi-SG route after the r220a3 de-bake fix.

Wheel `62f234c8d0aa89c4aec3e8d2c6fa22ad` (build22; mainloop
`a02286184bdb07a088e0321a22bb0111`, dequant
`99300a75205df72e1a03d6f9b70f1702`).

Instrument: `r220c_verify.py` — cross-policy (`_64` vs certified
`_16`, same wheel, env-flipped per call), determinism (two `_64`
runs), cross-impl (vs Triton v2), NaN-prefill unwritten-row counts.
Eager only (no graph capture) at the engine geometry
(D=256 HREP=6 page 64).

## Gate v2 (excess band) — PASS, all 6 cells

```
     fmt    ctx |     16-64 |       det |     64-v2 |     16-v2 |       exc |  nan16  nan64 | verdict
   mxfp8   1024 |  0.000977 |  0.000000 |  0.001827 |  0.001827 |  0.000000 |      0      0 | OK
   mxfp4   1024 |  0.000977 |  0.000000 |  0.050452 |  0.050452 |  0.000000 |      0      0 | OK
  mxk8v4   1024 |  0.000977 |  0.000000 |  0.051015 |  0.051015 |  0.000000 |      0      0 | OK
   mxfp8  32768 |  0.000244 |  0.000000 |  0.000172 |  0.000174 |  0.000002 |      0      0 | OK
   mxfp4  32768 |  0.000488 |  0.000000 |  0.006216 |  0.006186 |  0.000030 |      0      0 | OK
  mxk8v4  32768 |  0.000244 |  0.000000 |  0.004201 |  0.004323 |  0.000122 |      0      0 | OK
R220C_GATE_V2_PASS
```

- `d_16_64` ≤ 0.000977 everywhere (was 0.379 uniform deviation on
  r220a, and seq-1 NaN on the e2m1-V lanes) — the double-bake
  diagnosis was correct; removing the anchor bakes fixed it.
- v2-excess ≤ 1.2e-4: the `_64` route tracks the certified `_16`
  route's own v2 relationship to quant-noise precision.

## Band history (why v2)

Run 2's log ends `R220C_GATE_FAIL` under gate v1
(`d_16_v2 <= 2e-3` absolute): the e2m1 lanes' native-vs-v2 floor
(0.05 @1K, 0.004-0.006 @32K) is quant-noise-scale and present
IDENTICALLY on the certified `_16` route — v1's band was
e4m3-calibrated. The gate took the excess form
`abs(d_64_v2 - d_16_v2) <= 2e-3` and was RECOMPUTED from these banked
artifacts (gate-recompute law; no cells re-run). The v2 table above is
that recomputation.

## Consumers

Value-correctness of the `_64` route is a precondition for the
r220a timing ladder and the r220b e2e A/A (both banked alongside);
the e2e verdict on the tile lever lives in the r220b bank.
