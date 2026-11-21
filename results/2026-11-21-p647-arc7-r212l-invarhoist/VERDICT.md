# r212l — the invariant-hoist iteration: correctness-clean, performance-NEUTRAL (null result)

**Date**: 2026-11-21 · **Leg**: r212l (arc7 Phase-8, kernel iteration +
offline validation) · **Stack**: r212l wheel (`vllm_xpu_kernels-0.1.14.1
+arc7mx`, md5 `bf4b0e3729e55268d1df753e157b0087`; build21 sources + the
r212l patch, ninja resume build, `r212l_build1.log` 11:29-11:45 UTC) +
`vllm-arc:v0.30.0-b70-cert25zc`, harness `p204_graphtime.py` at engine
geometry (D=256 / HREP=6 / BLOCK=64, graph 32×50, SPLITS=auto) · **Runs**:
`r212l_launch1.log` 11:46-11:58 UTC (7 cells: probe 0 × ctx {1024, 8192,
32768} + probes {2,3,6,8} × ctx 32768, FMTS=mxfp4,mxfp8)

## Verdict

**The r212k-attributed bookkeeping redundancy is NOT on the critical
path — hoisting it is correctness-clean (all canaries 0.00e+00) and
performance-neutral (fulls within the ±6% layout-dice band). The walk
is bound by per-element decode ALU + the memory pipeline, not by
loop-invariant instruction count.**

### The iteration (source: `tools/ar/p147_mxkv/p212l_patch.py`)

Two anchored edits on the build21 tree:

1. **V-lane per-tile invariants**: `dequant_vslm` re-derived its 8 rows
   (8 tv_coord layout evals) and re-read the same 64 scale bytes as 8
   scalar gmem loads EVERY VV step (×8 = 64 evals + 64 loads per
   k-tile per WI). Split into `dequant_vslm_prep` (rows + full scale
   lines once per k-tile, 2 u32 per row; the second u32 predicated on
   the row's group count — at D=128 the scale row is 4 bytes and an
   unpredicated pair read would walk past the pool) + a walk that
   consumes the registers, extracting the group byte per VV step.
2. **K static window law** (e2m1 staged branch): the 16 per-d-tile
   `tv_coord` window-address evals replaced by the arithmetic the
   pinned form already implies (row = `t1 + 8*r`, `w = org_elem/2 +
   8*g` — `t0` floors away in the byte window, `org_elem` is the
   64-aligned d-tile base).

TU pre-flighted with ninja before the wheel (build-death law);
4 pre-existing duplicate-inline warnings, no new ones.

### The result (probe-0 cells vs banked build21 baselines)

| fmt | ctx | new µs | base µs | delta |
|---|---|---|---|---|
| mxfp4 | 1024 | 594.5 | 598.4 | -0.7% |
| mxfp4 | 8192 | 420.6 | 423.0 | -0.6% |
| mxfp4 | 32768 | 1028.3 | 1040.3 | -1.2% |
| mxfp8 | 1024 | 658.3 | 643.1 | +2.4% |
| mxfp8 | 8192 | 541.7 | 535.1 | +1.2% |
| mxfp8 | 32768 | 1359.0 | 1344.9 | +1.0% |

Slice re-decomposition at 32K: mxfp4 V_gather 392.1 → 375.9 (-4%),
K 357.6 → 365.9, Vs 82.1 → 83.9, floor 142.9 → 139.5; mxfp8 V_gather
317.1 → 292.3, V_stage 390.9 → 364.3, K 407.2 → 470.0 — the mxfp8 K
lane is UNCHANGED CODE (dequant_reg, e4m3); its +63 µs is codegen
drift, the binary-layout dice law in slices. Fulls move ≤ 2.4%.

### Why neutral (the lesson)

The hoisted class — loop-invariant address arithmetic and L1-hit
scalar loads inside a fully unrolled VV loop — was already hidden:
IGC CSEs the invariant tv arithmetic across the unrolled iterations,
and back-to-back L1-hit loads pipeline through. Removing ~100
bookkeeping ops per WI per k-tile does nothing when the walk carries
~5000 per-element decode ops plus the SLM round-trip. **Corollary for
the loop: only per-element work reduction (decode ALU, the SLM
write-side) or pipeline-structure changes (tile width) can move the
MX-vs-fp8 walk excess.** The r212k decomposition remains the map:
mxfp4 K 366 + V_gather 376 + V_stage 84 + floor 140 at 32K; mxfp8
K 470 + Vg 292 + Vs 364 + floor 171.

Reference point banked in the r212j2 JSONs: v2 (the certified Triton
MX decoder) costs 1090.2 µs at the same 32K shape — the native kernel
(1040) already beats it; the fp8-FA2 offline denominator is not
instrumented in this harness (the e2e ctrlA24 bar is the standing
reference).

### Disposition

- The patch STAYS in the tree (validated; strictly fewer loads; the
  staged scale lines set up code-domain scale access for the next
  iteration). build21 backups: `*.build21bak` beside the sources;
  build21 wheel: `...whl.build21bak` in `wheels-arc-vxk/`.
- Next lever (r212m): direct-bf16 decode assembly with the e8m0
  exponent folded into the bf16 exponent field — every e2m1/e4m3 ×
  2^k product is exactly representable in bf16 (small-mantissa ×
  power-of-2), so the direct assembly is bit-identical to the current
  f32-multiply-then-convert path while cutting the per-element decode
  from ~10 ops to ~5-6. Targets both K and V walks of both formats.

## Artifacts

`offline/p212l_m0_{1024,8192,32768}.json` (full cells),
`offline/p212l_m{2,3,6,8}_32768.json` (slice cells), `r212l_launch1.log`,
`r212l_build1.log`. Rig: `/root/build/p199/out/p212l_*`, logs
`/root/build/p199/logs/`. Tools: `r212k_probedecomp.sh` (the
decomposition leg, banked with this verdict — it produced the r212k
baselines AND the r212l slices), `r212l_val.sh`, `p212l_patch.py`.
