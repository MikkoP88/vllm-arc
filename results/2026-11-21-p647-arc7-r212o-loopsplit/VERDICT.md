# r212o — loop split (branch hoist): null on the fast path, hypothesis falsified cleanly

**Date**: 2026-11-21 · **Leg**: r212o (arc7 Phase-8 iteration) · **Stack**:
r212o wheel (`vllm_xpu_kernels-0.1.14.1+arc7mx`, md5
`92a66f1f4624fbb5358e7cf957719f24`; r212m3 tree + `p212o_patch.py`,
ninja-resume build `r212o_build1.log` 15:35–15:51 UTC after a 6m38s TU
pre-flight) + `vllm-arc:v0.30.0-b70-cert25zc`, harness
`p204_graphtime.py` at engine geometry · **Run**: `r212o_val1.log`
15:54–16:10 UTC (9 cells: probes {0,2,3,6,8,9} × 32768 + m0 × {1024,
8192} + dbf0 × 32768) — the full telemetry-law shape

## Verdict

**The loop split is a −1% null on the fast path and FALSIFIES the
per-element-branch hypothesis: the runtime `!probe_val && fast_band`
evaluated ~50M times/call was essentially free (predication), the
e2m1 skeleton floor did not move (248.7 vs 246.6), and the m8 lane
held byte-identical source to +0.2% (1323.1) — the e4m3
stability-oracle discipline works. The patch STAYS (strictly cleaner,
branch-free fast loop, −1%), but the ~361 µs decode residual is
something else.**

### The numbers

| cell | r212o | r212n (m3 wheel) | delta |
|---|---|---|---|
| mxfp4 m0 1K/8K/32K | 397.1 / 303.9 / 741.8 | 399.8 / 306.8 / 749.5 | −0.7/−0.9/−1.0% |
| mxfp8 m0 32K | 1323.1 | 1320.7 | +0.2% (byte-identity held) |
| m4 K / Vg / Vs / floor | 170.5 / 193.0 / 18.7 / 248.7 | 178.2 / 196.7 / 16.7 / 246.6 | −4.3/−1.9/…/+0.9% |
| m4 m9 / dec_resid | 380.6 / 361.2 | 379.0 / 370.5 | probe tail unchanged |
| dtype oracle (dbf0) | 911.6 → 741.8 = +18.6% | 829.5 → 749.3 = +9.7% | see below |

The oracle "gain" growing to +18.6% is NOT a fast-path win — the
SLOW loop's own layout dice moved the float path (911.6 vs 829.5,
+10%); the shipping fast path moved −1.0%. Cross-check the m8 oracle:
0.2% — byte-identical paths reproduce exactly.

### What the falsification teaches

The probe-9 cell (380.6) carries a per-element tail HEAVIER than the
fast path's bit ops (float convert + inline `e8m0_to_scale` + mul +
f32→bf16 convert vs ~6 u16 bit ops) yet sits 361.2 µs BELOW the fast
path — so the fast tail's cost is not op count and not branch
machinery. Remaining suspects for the residual: **u8/u16 scalar
bit-surgery codegen** (16-bit GRF op emulation vs native float
converts), **dependency-chain serialization** (the extract→shift→or
chain per element under register-pressure-limited ILP), and the
**per-element `dst(i)` fragment stores** both paths pay. The next
instrument (queued, r212p): a LIGHT probe-9 tail (extract + float
convert + store, no scale/assembly — the original r212m form) to
separate the assembly cost from the store/extract floor before any
tile-width or store-packing surgery.

### Disposition

- r212o wheel is the working state (m3 backup `*.r212m3bak` sources
  + `...whl.r212m3bak` wheel = e50f1c3a preserved).
- e2e: r212o's −1% does not justify a face rerun; the banked r212m4
  faces on the m3-class wheel remain the e2e state of record
  (m4 48.9/56.7/73.1).
- Next per the loop directive: r212p light-probe pricing, then decide
  between store packing (fragment tv archaeology) and tile width.

## Artifacts

`offline/p212o_{m0_1024,m0_8192,m0,dbf0,m2,m3,m6,m8,m9}_32768.json`,
`r212o_build1.log`, `r212o_val1.log`. Rig: `/root/build/p199/out/p212o_*`,
`/root/build/p199/logs/r212o_*`, `/root/build/r212o_{build1,val1}.log`.
Tools: `p212o_patch.py`, `r212o_val.sh` (commit 9fb109e9).
