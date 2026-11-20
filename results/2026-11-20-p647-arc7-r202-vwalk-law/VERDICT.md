# r202 — V-walk form-law arithmetic + V-prefetch hoist

**VERDICT: PASS correctness, NEUTRAL performance (kept for the law lock
+ the diagnostic).**

- Wheel: `arc7mx-r202` md5 `503416b6fa0d91a78f94a67bf58bc22b`.
- Sources: `mx_dequant.hpp` `85b1a13f`, `mx_paged_decode_mainloop.hpp`
  `6a04f56c`.

## What changed

1. `dequant_vslm` walk: per-element `tv_coord` (runtime idx2crd +
   hierarchical layout apply) replaced by the pinned form-(ii) law
   arithmetic — `k_loc = kr[(i>>1)&7]`, `v_loc = t1 + 8*(i&1) +
   16*(i>>4)`, `t1 = (wi>>1)&7`. Bitwise faces PASS: the law is now
   confirmed by direct arithmetic (not just layout evaluation).
2. V prefetch for tile K+1 hoisted next to the r201 K prefetch (the
   old current-tile issue sat at the staging copy's doorstep).

## 32K device ms (r201b -> r202)

| mode | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 0 full | 2.810 -> 2.870 | 2.363 -> 2.308 | 2.616 -> 2.687 |
| 2 const-V | 1.614 -> 1.647 | 1.262 -> 1.258 | 1.615 -> 1.601 |
| 6 const-K | 1.638 -> 1.668 | 1.402 -> 1.393 | 1.523 -> 1.461 |

Gates: synth-v2 ALL_STAGES_OK x2, faces 42/42 VERDICT PASS (bitwise
0/1024 x3). All deltas inside the +-3% noise band.

## Diagnostic (the value of this run)

Per-element tv_coord was NOT a live cost (LLVM CSE'd it), and V
prefetch distance is not the V-lane limiter. The ~0.9 ms V-dequant
slice (m0-m2 minus staging) is therefore dominated by the SLM byte
TRANSACTIONS: 32 scattered byte loads per WI per VV step (form-(ii)
puts the WI's bytes at v-stride 8 across 8 k-rows). The K lane reads
4x fewer transactions per element (16 u32s per 64 elements via the
staged windows) and is proportionally cheaper.

## Next (r203)

V-axis scatter PERMUTATION in the staging frame so each WI's 4 bytes
per k-row land CONTIGUOUS: e4m3 phys(v) = (v%8)*4 + v/8 (WI's v =
t1+8j -> phys 4*t1+j = one u32 per (row, WI), byte select by
(i&1)+2*(i>>4) compile-time); e2m1 phys_byte(b) = (b%4)*4 + b/4
(WI's nibbles share one nibble-phase and land in one u32). Staging
scatter target gets the permutation (byte-granular writes unchanged
in count); dequant reads 8 u32s per VV step instead of 32 byte loads.
Same wheel: lean e4m3 decode (branchless bit assembly; in-cache law
excludes the NaN band) for the shared K/V decode ALU slice.
