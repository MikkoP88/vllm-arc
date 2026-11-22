# r203b — permuted V frame (u32 gather) + lean e4m3 decode

**VERDICT: PASS — mxk8v4 BEATS the certified v2 Triton decoder at 32K
(1.408 vs 1.575 ms, 1.11x); mxfp8 -73% cumulative; all correctness
gates green (synth-v2 x2, faces 42/42, bitwise 0/1024 x3).**

- Wheel: `arc7mx-r203b` md5 `4f0891ded3cedb914f83c84f0027512e`
  (r203a `214861d1` was the e2m1 scatter half-window bug — see below).
- Sources: `mx_dequant.hpp` `6db53097`, `mx_paged_decode_mainloop.hpp`
  `7a1e0818`, `p200c_vlane_debug.py` `e4039fc5` (A-stage inverse
  permutation mirror).

## What changed

1. **PERMUTED V FRAME**: the staging scatter transposes the v-axis
   within each VV-step byte window (e4m3: phys(c) = (c&7)*4 + (c>>3);
   e2m1 per-16-byte window: phys(b) = (b&3)*4 + (b>>2)) so each
   (k-row, WI)'s four payload bytes land as ONE aligned u32 —
   dequant_vslm reads 8 u32s per VV step instead of 32 scattered byte
   transactions (the r202-diagnosed ~0.9 ms V-dequant slice).
   Byte-in-u32 j = (i&1) + 2*(i>>4); e2m1 nibble phase = t1&1.
2. **LEAN e4m3 DECODE**: branchless bit assembly (normals
   s|(e+120)<<23|m<<20; subnormals m*2^-9 via float multiply; sign
   select keeps -0.0 bitwise). Cutlass's generic convert carried a
   NaN check + subnormal normalize while-loop per byte. In-cache law
   excludes the NaN band (saturating quantizer).

## r203a bug (caught by the gates, fixed in r203b)

The e2m1 scatter fed the pair-spanning vb (0..31) to phys2 instead of
the step-local vb&15 — the high half of each copy window overflowed
the next VV step's window. Symptoms: A_BAD 572 + C 760 on mxk8v4,
bitwise 500/1024 on BOTH e2m1-V lanes, e4m3/mxfp8 fully CLEAN —
clean format attribution. One-line fix (`bl = vb & 15`).

## 32K device ms (mode0: r200h -> r201b -> r203b)

| fmt | r200h | r201b | r203b | v2 | ratio |
|---|---|---|---|---|---|
| mxfp8 | 4.732 | 2.810 | **1.277** | 0.768 | 0.60x |
| mxfp4 | 3.674 | 2.363 | **1.991** | 0.660 | 0.33x |
| mxk8v4 | 4.422 | 2.616 | **1.408** | 1.575 | **1.11x** |

Lanes (mxfp8): staging 0.09 (m0-m1; was 0.30 — the permuted
destinations also COALESCE the scatter's SLM byte writes), V
gather+decode 0.38 (m0-m2; was ~0.9+), K chain 0.27 (m0-m6; was 1.17
— the lean decode did for e4m3-K what the scale vector started).

## Next levers

mxfp4/mxk8v4's remaining gap is the e2m1-K staged path (mxfp4 K chain
~1.03 ms): lean branchless e2m1 decode + pair-decode (one byte -> two
values, halving decode calls). Walk base (mode6) ~1.0 ms across
formats — split-overhead/wave-shaping territory after the lanes.
