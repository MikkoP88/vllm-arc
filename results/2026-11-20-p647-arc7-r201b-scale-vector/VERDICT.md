# r201b — K-lane scale vector (lean form law) + K-prefetch distance

**VERDICT: PASS — mode0 -36..-41% across formats; correctness closed
(bitwise 0/1024 x3, synth-v2 ALL_STAGES_OK x2, faces 42/42).**

- Wheel: `vllm_xpu_kernels-0.1.14.1+arc7mx-r201b-cp312` md5
  `e74878c280b701a5fd8d99f7010af102` (rig `fix/` archive; replaces the
  REGRESSED r201 wheel `a008f49b` — generic 16-slot dedup spilled, 2.5x
  slower, see R201 REGRESSION LAW in the master plan).
- Sources: `mx_dequant.hpp` md5 `6f261da93fb046c7a4c4289e23bce437`,
  `mx_paged_decode_mainloop.hpp` md5 `2497f001c97720c8785d6c63119e0586`.

## What changed

1. **K-lane scale VECTOR** (both K dequant paths — `dequant_reg` e4m3 +
   `dequant_dtile` staged e2m1): replaces the r200g-PRE 4-entry scanned
   scale cache (~75% miss at the QK B-form's 8 rows x 2 groups) with
   kr[8] rows read through `tv_coord` at registers 0..7 + two FLAT
   vectors sc0/sc1[8] (one per scale group), all 16 loads issued upfront.
   QK-vs-PV FORM LAW: the QK B-tile's row is CONSTANT in v1, varies over
   (v0, v2) — slot = (i & 1) + 2*(i >> 4) (the e2m1 staging path's
   bitwise validation proves v1-independence). Flat indexing keeps every
   array access compile-time under the unroll (the r201 spill mechanism
   was runtime-indexed tables); the only runtime bit is the goff 0/1
   select. Guard `kr[slot] != row || goff out of [0,1]` falls back to
   per-element loads (correct, slow) on any form break.
2. **K-prefetch distance** (mainloop): next tile's K prefetch hoisted
   from the loop END (distance ~0, covered nothing) to right after the
   QK D-loop — loads warm through staging+softmax+PV. Surface-crossing
   tiles keep the loop-end warmer (guard `esurf == k_pre_surf`).

## 32K device ms (probe modes 0-8, r200h -> r201b)

| mode | mxfp8 | mxfp4 | mxk8v4 |
|---|---|---|---|
| 0 full | 4.732 -> **2.810** | 3.674 -> **2.363** | 4.422 -> **2.616** |
| 1 skip V staging | 2.507 | 2.146 | 2.436 |
| 2 const-V | 1.614 | 1.262 | 1.615 |
| 6 const-K | 1.843 -> 1.638 | 1.525 -> 1.402 | 1.520 -> 1.523 |
| 7 compute floor | 0.496 | 0.257 | 0.213 |
| 8 gemm floor | 0.148 | 0.157 | 0.174 |

- **K chain (m0-m6): 2.889 -> 1.172 / 2.149 -> 0.961 / 2.902 -> 1.093 ms
  (-59/-55/-62%)** — the r200h diagnosis (K lane = 58-66% of the walk)
  confirmed and cut.
- const-K modes ~unchanged (they sever the changed code) — the
  attribution is clean.
- vs v2 at 32K (0.768/0.660/1.575): native now 0.27x/0.28x/0.60x
  (was 0.16x/0.18x/0.36x).

## Next levers (ladder order)

V lane (m0-m2 ~= 1.20/1.10/1.00 ms), remaining K chain ~1.0-1.2 ms
(prefetch + e4m3 bitcast decode), then split-overhead (B ~= 1.7 ms).
