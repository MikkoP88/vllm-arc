# VERDICT — P6.47 P-I r251: the group-64 E8M0 offline band — zero cost on the gating operand; the lane is CLOSED-DEFERRED on target-relevance (2026-11-26)

**What**: P-I improvement cycle 2 of 3 — the group-64 E8M0 lever
(the corrected lever map's open candidate #1). The offline band
oracle (`tools/ar/p147_mxkv/p251_group64_oracle.py`, run in the
cert28 container, CPU-only, the certified wheel's own `mx_quant`
at torch 2.13.0+xpu) measured the round-trip error of the certified
encoders at OCP group 32 vs a widened group 64, per operand
element-format, on three populations (neutral iid gaussian;
heavy-tailed per-block amplitudes; adversarial adjacent-block 4x
amax alternation — the legal worst case for scale sharing).

## The band (12 cells, all D in {128, 256})

| population | fmt (operand class) | rmse ratio g64/g32 | max-err ratio | 32-blocks coarsened |
|---|---|---|---|---|
| gauss | mxfp8 (e4m3) | **1.000** | 1.000 | 7.8–8.0 % |
| gauss | mxfp4 (e2m1) | 1.031–1.032 | 1.000 | 8.5–8.7 % |
| heavy | mxfp8 (e4m3) | **1.000** | 1.000 | 46.0–46.4 % |
| heavy | mxfp4 (e2m1) | 1.061–1.067 | 1.000 | 46.4 % |
| adverse | mxfp8 (e4m3) | **1.000** | 1.000 | 50.0 % |
| adverse | mxfp4 (e2m1) | 1.181–1.183 | 1.000 | 50.0 % |

**Mechanism (why e4m3 is immune and e2m1 is not).** Both element
formats are floating: RN relative error is scale-free for both. The
difference is the small-element floor. e4m3's exponent range
reaches 2^-9 of the scale — its floor never binds, so a coarser
shared E8M0 scale costs literally zero (ratio 1.000 at every
population, even where 50 % of blocks were coarsened). e2m1's
smallest nonzero grid step is 0.5 of the scale — doubling a block's
scale doubles the magnitude below which elements round to zero,
so rmse rises in proportion to the coarsened fraction (gauss +3 %,
heavy +6 %, adverse +18 %). No new max-error class anywhere
(max ratio 1.000 — the top-magnitude grid step is unchanged).

**Where the cost lands on the champion lane.** mxk8v4 is e4m3 K +
e2m1 V: the coarsening cost is **zero on K — the acceptance-gating
operand — and +3–18 % rmse on V — the operand r238 measured
acceptance-free**. The lever's named risk ("coarser scales, K-side
risk") is measured empty for the champion lane; it would only exist
for the mxfp4-K lane, which already carries the e2m1-K acceptance
tax (r238) and is not the champion.

## The other two legs (already closed by facts, re-confirmed)

- **Weights side: vendor-locked.** The PWAL layout is `[k/32, n]`
  e8m0, the `fp4_gemm_w4a16`/`w4a8` weight-decomposition route
  reads that layout, and OCP MX defines the 32-element block —
  the "halves scale bytes everywhere" framing in the r247 lever
  map was the second framing slip (scale bytes are 4–6 % of a
  slot; halving them is 2–3 %, not "everywhere" — and on the
  weights side, not ours to change).
- **Capacity: +2.0 % pool on the champion lane** (400 → 392 B at
  D=256; mxfp4 272 → 264 = +2.9 %; mxfp8 528 → 520 = +1.5 %).

## Disposition: CLOSED-DEFERRED (viable, but nothing harvests it before cert29)

The band is good news banked: a group-64 KV lane on the champion
mix would cost zero on the gating operand. But the disposition
closes on target-relevance and sequencing, not quality:

1. **No binding face.** No qwen target face in any recent bank was
   pool-limited (the sustained c32768 n4 sequence served clean in
   r249; the KV_CAP refusal class was the gemma D=512 lane —
   target-only law). A +2 % pool gain has no face to harvest.
2. **The wiring surface is exactly the code cert29 must freeze.**
   The scale-offset law runs through the Triton store/decode
   kernels (v2/v5), the native kernel, `mx_layout` offsets, the
   engine pool arithmetic, AND the lmcint packed-row geometry that
   P-H2 (r249) just fixed and validated byte-true — a group-64
   rewrite re-opens the interop surface days after certifying it.
3. **Layout compatibility.** Group-64 abandons the OCP MX
   32-element block — a fork-proprietary cache shape, fine on this
   rig but a compatibility line worth naming.

**Reopen trigger (banked)**: any qwen face refused or degraded by
KV-pool pressure (KV_CAP class on the target), or a post-cert29
window where the interop surface is open anyway — the band in this
bank is the enabling evidence (K-side zero, V-side bounded), and
the acceptance probe is the mandated next rung at that point.

## NEXT RUNG (P-I cycle 3)

mxfp4-W4A8 weights lane: `fp4_gemm_w4a8` ships in the wheel, the
lane is hardware-legal (xe2 verdict), per-token activation quant
avoids the p630 per-call tax class. Measure decode + ITL vs the
W4A16 incumbent at the target faces.
